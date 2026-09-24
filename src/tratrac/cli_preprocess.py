"""Typer entry point for the pre-pass that resolves every geometric transform
(``tratrac-preprocess``).

Two subcommands, one file, one owner: ``tratrac`` and ``tratrac-postprocess``
never estimate ego-motion, resolve GSD scale, or fit a world-projection
homography themselves — they only ever read rows this tool wrote. See
"Detector-free ego-motion" in infrastructure/video/ego_motion_orb.py's module docstring.

``estimate`` (mandatory before every ``tratrac`` run, even a static camera):
walks VIDEO once with masked ORB + RANSAC (only if ``--background-zones`` is
given -- otherwise there is no ego-motion to estimate), resolves the GSD scale
from drone geometry or a direct value, and writes both as rows -- one
ego-motion row and one scale row per frame -- plus the keyframe-anchor PNGs an
operator draws exclusion-zone/calibration correspondences on. No detector, no
tracker, no track record.

``project`` (run after an operator authors ``calibration.json`` against
``estimate``'s anchor PNGs): fits a world-projection homography and appends it
to the same transforms file, materialized as one row per frame already in the
file. Never touches the video -- a fast file-in/file-out step, so re-fitting a
different calibration costs one quick invocation, not a re-walk.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray
from tqdm import tqdm

from tratrac.application.road_graph import plane_id_for_point, to_global_plane_polygons
from tratrac.calibration.drone_specs import known_models, lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.calibration.srt_parser import mean_altitude
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.ports import AnchorSink, CoordinateTransform, EgoMotionEstimator
from tratrac.domain.world import Calibration, Correspondence
from tratrac.infrastructure.anchors.recording import AnchorRecordingEgoMotionEstimator
from tratrac.infrastructure.anchors.sink import AnchorImageSink
from tratrac.infrastructure.background.json import load_background_zones
from tratrac.infrastructure.road_graph.json import load_plane_zones
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.transform.records import (
	HomographyFunction,
	ScaleFunction,
	TransformRow,
	publish,
	read_jsonl,
	staging_path,
	write_jsonl,
)
from tratrac.infrastructure.transform.sink import (
	CANVAS_PAD,
	CoordinateTransformSink,
	read_ego_motion,
	whole_canvas,
)
from tratrac.infrastructure.video.ego_motion_orb import (
	BackgroundZoneMaskSource,
	OrbEgoMotionEstimator,
)
from tratrac.infrastructure.video.opencv import OpenCvVideoSource
from tratrac.infrastructure.world.calibration import compute_homography, load_calibration

app = typer.Typer(
	name="tratrac-preprocess",
	help="Resolve every geometric transform (ego-motion, scale, world projection) "
	"before the main run.",
	no_args_is_help=True,
)


@app.command()
def estimate(
	video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="The clip to walk.")],
	out: Annotated[
		Path,
		typer.Option("--out", "-o", dir_okay=False, help="The transforms sidecar (JSONL)."),
	],
	background_zones: Annotated[
		Path | None,
		typer.Option(
			"--background-zones",
			exists=True,
			dir_okay=False,
			help="Operator-authored background_zones.json (an external tool draws these on "
			"the video, same role as calibration.json/zones.json today — see "
			"infrastructure/video/ego_motion_orb.py's module docstring). Omit for a static camera: no "
			"ego-motion rows are written, only the per-frame scale rows.",
		),
	] = None,
	anchors_dir: Annotated[
		Path | None,
		typer.Option(
			"--anchors-dir",
			file_okay=False,
			help="Directory for keyframe-anchor PNGs (draw --calibration/--exclusion-zones "
			"correspondences on these; poses come from --out by frame index, not from a "
			"separate manifest — see src/tratrac/application/EXCLUSION_ZONES.md). Only "
			"meaningful (and only produced) alongside --background-zones.",
		),
	] = None,
	meters_per_pixel: Annotated[
		float | None,
		typer.Option("--meters-per-pixel", help="Direct GSD metres-per-pixel calibration."),
	] = None,
	drone_model: Annotated[
		str | None,
		typer.Option("--drone-model", help="Drone model (calibration/drone_specs registry)."),
	] = None,
	altitude_m: Annotated[
		float | None, typer.Option("--altitude-m", help="Altitude AGL in metres.")
	] = None,
	srt: Annotated[
		Path | None,
		typer.Option("--srt", exists=True, dir_okay=False, help="DJI .SRT sidecar for altitude."),
	] = None,
	n_features: Annotated[
		int, typer.Option("--n-features", help="ORB keypoints per frame.")
	] = 2000,
	match_ratio: Annotated[
		float, typer.Option("--match-ratio", help="Lowe ratio test, (0, 1); lower = stricter.")
	] = 0.75,
	min_matches: Annotated[
		int, typer.Option("--min-matches", help="Minimum good matches to fit a transform.")
	] = 10,
	ransac_threshold: Annotated[
		float, typer.Option("--ransac-threshold", help="RANSAC reprojection threshold, px.")
	] = 3.0,
	min_anchor_overlap: Annotated[
		float,
		typer.Option(
			"--min-anchor-overlap",
			help="Re-anchor the keyframe when less than this fraction stays in view, (0, 1).",
		),
	] = 0.6,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite existing outputs.")
	] = False,
) -> None:
	"""Walk VIDEO once and write its ego-motion + scale rows (+ anchor PNGs)."""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	if anchors_dir is not None and anchors_dir.exists() and anchors_dir.is_file():
		raise typer.BadParameter(f"{anchors_dir} must be a directory, not a file.")
	zones = None
	if background_zones is not None:
		try:
			zones = load_background_zones(background_zones)
		except (ValueError, OSError) as exc:
			raise typer.BadParameter(str(exc)) from exc

	out.parent.mkdir(parents=True, exist_ok=True)
	try:
		source = OpenCvVideoSource(video)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	with source:
		width, height = source.metadata.width, source.metadata.height
		try:
			scale = _resolve_scale(
				meters_per_pixel=meters_per_pixel,
				drone_model=drone_model,
				altitude_m=altitude_m,
				srt=srt,
				image_width_pixels=width,
			)
		except ValueError as exc:
			raise typer.BadParameter(str(exc)) from exc

		with (
			CoordinateTransformSink(out, width=width, height=height) as sink,
			_anchor_sink(anchors_dir, wanted=zones is not None) as anchor_sink,
		):
			estimator: EgoMotionEstimator | None = None
			if zones is not None:
				anchor_poses: list[Transform2D] = []
				orb = OrbEgoMotionEstimator(
					n_features=n_features,
					match_ratio=match_ratio,
					min_matches=min_matches,
					ransac_threshold=ransac_threshold,
					min_anchor_overlap=min_anchor_overlap,
					mask_source=BackgroundZoneMaskSource(zones),
					anchor_observer=(
						(lambda _index, pose: anchor_poses.append(pose))
						if anchor_sink is not None
						else None
					),
				)
				estimator = RecordingEgoMotionEstimator(orb, sink)
				if anchor_sink is not None:
					estimator = AnchorRecordingEgoMotionEstimator(
						estimator, anchor_poses, anchor_sink
					)
			total = source.metadata.total_frames if source.metadata.total_frames > 0 else None
			n_frames = 0
			zone = whole_canvas(width, height)
			for frame in tqdm(source.frames(), total=total, desc="Estimating", unit="frame"):
				if estimator is not None:
					estimator.estimate(frame)
				sink.record_row(TransformRow(frame.index, zone, ScaleFunction(scale)))
				n_frames += 1

	typer.echo(
		f"Estimated {n_frames} frames -> {out} (scale={scale} m/px"
		f"{f', anchors: {anchors_dir}' if anchors_dir is not None else ''}). "
		"Point tratrac's input.transforms_in at the transforms file."
	)


@app.command()
def project(
	transforms: Annotated[
		Path,
		typer.Option(
			"--transforms", exists=True, dir_okay=False, help="The estimate run's transforms file."
		),
	],
	calibration: Annotated[
		Path,
		typer.Option(
			"--calibration",
			exists=True,
			dir_okay=False,
			help="World-projection calibration JSON (image<->world ground correspondences), "
			"authored against estimate's anchor PNGs.",
		),
	],
	plane_zones: Annotated[
		Path | None,
		typer.Option(
			"--plane-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space elevation-plane polygons (MVP3, "
			"src/tratrac/domain/road_graph.py). Fits one homography per plane (grouping "
			"correspondences by which plane zone they classify into) instead of one for "
			"the whole scene, for scenes with grade separation (bridges, overpasses, ramps).",
		),
	] = None,
) -> None:
	"""Fit a world-projection homography and append it to --transforms."""
	try:
		calibration_data = load_calibration(calibration)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc
	try:
		pose = read_ego_motion(transforms)
		existing = read_jsonl(transforms)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc
	if not existing:
		raise typer.BadParameter(f"{transforms} has no rows to fit against.")
	frame_indices = sorted({row.frame_index for row in existing})
	width, height = _infer_canvas(existing[0].zone)

	try:
		if plane_zones is not None:
			rows = _fit_per_plane(calibration_data, pose, plane_zones, frame_indices)
		else:
			rows = _fit_whole_scene(calibration_data, pose, frame_indices, width, height)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	# Replace, not append: re-running project (e.g. with a tweaked calibration.json)
	# must stay idempotent -- appending would leave the old homography rows in place
	# too, and TransformTable would then see two rows per (frame, zone) and raise
	# "ambiguous" on every read. Scale rows are dropped too, not just homography ones:
	# once a homography is fitted it supersedes the GSD scale for that zone (cli_postprocess
	# uses ScaleFunction(1.0) whenever homography rows are present), and a scale row shares
	# the whole-canvas zone with a whole-scene homography row, so leaving it in place would
	# make every frame ambiguous for exactly the same reason.
	kept = [
		row for row in existing if not isinstance(row.function, HomographyFunction | ScaleFunction)
	]
	staging = staging_path(transforms)
	write_jsonl(staging, [*kept, *rows])
	publish(staging, transforms)
	typer.echo(f"Projected {len(frame_indices)} frames -> {transforms}.")


def _resolve_scale(
	*,
	meters_per_pixel: float | None,
	drone_model: str | None,
	altitude_m: float | None,
	srt: Path | None,
	image_width_pixels: int,
) -> float:
	"""Mirrors the old ``[calibration]`` one-of: a direct value, or a drone model plus
	an altitude source. Raises ``ValueError`` on an invalid/incomplete combination."""
	if meters_per_pixel is not None and drone_model:
		raise ValueError("specify exactly one of --meters-per-pixel or --drone-model, not both.")
	if meters_per_pixel is not None:
		if meters_per_pixel <= 0.0:
			raise ValueError("--meters-per-pixel must be positive.")
		return meters_per_pixel
	if drone_model:
		if drone_model.lower() not in known_models():
			known = ", ".join(known_models())
			raise ValueError(f"--drone-model {drone_model!r} is unknown; known: {known}.")
		spec = lookup(drone_model)
		if altitude_m is not None:
			if altitude_m <= 0.0:
				raise ValueError("--altitude-m must be positive.")
			altitude = altitude_m
		elif srt is not None:
			altitude = mean_altitude(srt)
		else:
			raise ValueError("--drone-model needs --altitude-m (> 0) or --srt.")
		return ground_sample_distance(
			sensor_width_mm=spec.sensor_width_mm,
			focal_length_mm=spec.focal_length_mm,
			altitude_m=altitude,
			image_width_pixels=image_width_pixels,
		)
	raise ValueError("set --meters-per-pixel, or --drone-model with --altitude-m or --srt.")


@contextmanager
def _anchor_sink(out_dir: Path | None, *, wanted: bool) -> Iterator[AnchorSink | None]:
	if out_dir is None or not wanted:
		yield None
		return
	with AnchorImageSink(out_dir) as sink:
		yield sink


def _infer_canvas(sample_zone: tuple[Point2D, ...]) -> tuple[int, int]:
	"""Recover the frame's (width, height) from a whole-canvas zone already in the
	file -- ``estimate`` always writes at least one (its scale rows), so ``project``
	never needs to reopen the video to know the canvas size. ``whole_canvas`` pads
	every side by ``CANVAS_PAD``, so that's subtracted back out here."""
	width = round(max(p.x for p in sample_zone) - CANVAS_PAD)
	height = round(max(p.y for p in sample_zone) - CANVAS_PAD)
	return width, height


def _fit_whole_scene(
	calibration: Calibration,
	pose: CoordinateTransform,
	frame_indices: list[int],
	width: int,
	height: int,
) -> list[TransformRow]:
	"""Group correspondences by anchor ``reference_frame``, fit one homography per
	anchor, then materialize a row for every frame in ``frame_indices`` -- the
	nearest-anchor assignment is resolved here, once, not left for a reader."""
	by_anchor: dict[int, list[Correspondence]] = defaultdict(list)
	for correspondence in calibration.correspondences:
		by_anchor[correspondence.reference_frame].append(correspondence)
	homography_by_anchor = {
		anchor: _fit_homography(group, pose) for anchor, group in by_anchor.items()
	}
	anchor_frames = sorted(homography_by_anchor)
	zone = whole_canvas(width, height)
	return [
		TransformRow(
			frame_index,
			zone,
			_homography_function(homography_by_anchor[_nearest(anchor_frames, frame_index)]),
		)
		for frame_index in frame_indices
	]


def _fit_per_plane(
	calibration: Calibration,
	pose: CoordinateTransform,
	plane_zones_path: Path,
	frame_indices: list[int],
) -> list[TransformRow]:
	"""Group correspondences by which plane zone they classify into, fit one
	homography per plane, materialize a row for every frame per plane -- the row's
	``zone`` is that plane's own global polygon, so no reader ever needs
	``plane_zones.json`` again."""
	plane_zones = load_plane_zones(plane_zones_path)
	global_planes = to_global_plane_polygons(plane_zones, pose)
	zone_by_plane = dict(global_planes)

	by_plane: dict[int, list[Correspondence]] = defaultdict(list)
	for correspondence in calibration.correspondences:
		point = pose.apply(correspondence.image, correspondence.reference_frame)
		by_plane[plane_id_for_point(point, global_planes)].append(correspondence)

	rows: list[TransformRow] = []
	for plane_id, group in by_plane.items():
		zone = zone_by_plane.get(plane_id)
		if zone is None:
			raise ValueError(
				f"correspondence(s) classified into plane {plane_id}, which has no drawn "
				f"zone in {plane_zones_path}; every correspondence must fall inside an "
				"explicitly-drawn plane zone."
			)
		function = _homography_function(_fit_homography(group, pose))
		rows.extend(TransformRow(frame_index, zone, function) for frame_index in frame_indices)
	return rows


def _fit_homography(group: list[Correspondence], pose: CoordinateTransform) -> NDArray[np.float64]:
	image_points = [pose.apply(c.image, c.reference_frame) for c in group]
	world_points = [c.world for c in group]
	return compute_homography(image_points, world_points)


def _homography_function(matrix: NDArray[np.float64]) -> HomographyFunction:
	return HomographyFunction(tuple(float(v) for v in matrix.flatten()))


def _nearest(sorted_frames: list[int], frame_index: int) -> int:
	"""The entry in ``sorted_frames`` closest to ``frame_index`` (same "nearest
	keyframe" rule the ORB stabilizer itself uses to decide when an anchor is
	still representative)."""
	i = bisect.bisect_left(sorted_frames, frame_index)
	if i == 0:
		return sorted_frames[0]
	if i == len(sorted_frames):
		return sorted_frames[-1]
	before, after = sorted_frames[i - 1], sorted_frames[i]
	return before if frame_index - before <= after - frame_index else after


if __name__ == "__main__":
	app()
