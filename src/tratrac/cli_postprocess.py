"""Typer entry point for the post-process pass (``tratrac-postprocess``).

Reads the perception run's track record (Parquet), optionally **filters** out whole tracks
that fall inside exclusion zones, runs the forward+RTS constant-acceleration Kalman smoother
per surviving track, and writes a de-jittered SSAM ``.trj``. Offline and zero-phase — the
second pass of the two-pass design (src/tratrac/application/SMOOTHING.md). Re-running with different ``--pos-noise`` /
``--jerk`` / ``--exclusion-*`` re-tunes with no re-detection.

Exclusion is **track-aware** (src/tratrac/application/EXCLUSION_ZONES.md): a track is dropped when the majority of its
observations fall inside a zone. Zones are authored on the anchor PNGs the run exported
(``--anchors-dir``); pass the run's anchor ``manifest.json`` via ``--anchors`` so each zone's
``reference_frame`` is mapped into the global frame by that anchor's pose. Omit ``--anchors``
for a static (non-ego-motion) run, where every pose is the identity.

With ``--calibration`` (MVP2, src/tratrac/application/WORLD_PROJECTION.md) the trajectories are projected onto the
metric **world** plane before smoothing: one homography is fitted from image↔world ground
correspondences (mapped into the global frame via the same anchor poses), every observation
is rewritten into world metres, and the ``.trj`` carries world coordinates with
``DIMENSIONS.Scale = 1.0``. Without it, coordinates stay image-space (the pre-MVP2 path).

Rendering is a separate step: ``tratrac-render`` on the ``.trj`` (src/tratrac/infrastructure/export/VIDEO_EXPORT.md).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from tratrac.application.exclusion import excluded_track_ids, to_global_polygons
from tratrac.application.road_graph import (
	lane_id_for_point,
	link_id_for_point,
	to_global_lane_polygons,
	to_global_link_polygons,
)
from tratrac.application.track_smoothing import TrackSample, smooth_to_states
from tratrac.application.world_projection import (
	PerAnchorWorldProjector,
	SingleHomographyProjector,
	local_scale_at,
)
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.ports import TrajectoryExporter, WorldProjector
from tratrac.domain.road_graph import LaneZones, LinkZones
from tratrac.domain.vehicle import VehicleState
from tratrac.domain.world import Calibration, Correspondence
from tratrac.infrastructure.anchors.manifest import ReferenceFrame, read_manifest
from tratrac.infrastructure.exclusion.json import load_exclusion_zones
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.road_graph.json import load_lane_zones, load_link_zones
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording, read_tracks
from tratrac.infrastructure.world.calibration import compute_homography, load_calibration

app = typer.Typer(
	name="tratrac-postprocess",
	help="Filter + smooth a track record into a de-jittered SSAM .trj (forward+RTS Kalman).",
	no_args_is_help=True,
)


@app.command()
def postprocess(
	tracks: Annotated[
		Path, typer.Argument(exists=True, dir_okay=False, help="Track record (export.out).")
	],
	out: Annotated[Path, typer.Option("--out", "-o", dir_okay=False, help="Output .trj path.")],
	exclusion_zones: Annotated[
		Path | None,
		typer.Option(
			"--exclusion-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space ROI polygons; tracks mostly inside are dropped.",
		),
	] = None,
	anchors: Annotated[
		Path | None,
		typer.Option(
			"--anchors",
			exists=True,
			dir_okay=False,
			help="The run's anchor manifest.json (from --anchors-dir); maps zones/correspondences "
			"authored on an anchor into the global frame. Omit for a static run (identity poses).",
		),
	] = None,
	link_zones: Annotated[
		Path | None,
		typer.Option(
			"--link-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space Link ID polygons (src/tratrac/domain/road_graph.py); "
			"surviving observations are classified per-frame and stamped onto VehicleState.link_id.",
		),
	] = None,
	lane_zones: Annotated[
		Path | None,
		typer.Option(
			"--lane-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space Lane ID polygons (src/tratrac/domain/road_graph.py); "
			"surviving observations are classified per-frame and stamped onto VehicleState.lane_id.",
		),
	] = None,
	calibration: Annotated[
		Path | None,
		typer.Option(
			"--calibration",
			exists=True,
			dir_okay=False,
			help="World-projection calibration JSON (image<->world ground correspondences). When "
			"given, trajectories are projected to metric world coordinates before smoothing "
			"(MVP2); DIMENSIONS.Scale becomes 1.0. Correspondences spanning more than one "
			"anchor fit one homography per anchor (each needs its own >= 4 correspondences) "
			"instead of pooling into a single one. See src/tratrac/application/WORLD_PROJECTION.md.",
		),
	] = None,
	exclusion_min_fraction: Annotated[
		float,
		typer.Option(
			"--exclusion-min-fraction",
			help="Drop a track when this fraction of its observations are inside a zone, (0, 1].",
		),
	] = 0.5,
	pos_noise: Annotated[
		float,
		typer.Option("--pos-noise", help="Measurement-noise std in px (detector center jitter)."),
	] = 2.0,
	jerk: Annotated[
		float,
		typer.Option("--jerk", help="Process jerk spectral density; higher = more responsive."),
	] = 20.0,
	timestep_precision: Annotated[
		float,
		typer.Option(
			"--timestep-precision",
			help="Min seconds between exported TIMESTEPs; 0 = every frame. Thins only the .trj "
			"output (the record keeps every frame).",
		),
	] = 0.0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite existing outputs.")
	] = False,
) -> None:
	"""Filter (optional) + smooth TRACKS into a de-jittered .trj at --out."""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	if timestep_precision < 0.0:
		raise typer.BadParameter("--timestep-precision must be >= 0 (0 = every frame).")
	if not 0.0 < exclusion_min_fraction <= 1.0:
		raise typer.BadParameter("--exclusion-min-fraction must be in (0, 1].")
	try:
		recording = read_tracks(tracks)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	dropped = 0
	if exclusion_zones is not None:
		recording, dropped = _filter_excluded(
			recording, exclusion_zones, anchors, exclusion_min_fraction
		)

	link_ids: dict[tuple[int, int], int] = {}
	if link_zones is not None:
		link_ids = _assign_labels(
			recording,
			link_zones,
			anchors,
			load_zones=load_link_zones,
			to_global_polygons=to_global_link_polygons,
			label_for_point=link_id_for_point,
		)

	lane_ids: dict[tuple[int, int], int] = {}
	if lane_zones is not None:
		lane_ids = _assign_labels(
			recording,
			lane_zones,
			anchors,
			load_zones=load_lane_zones,
			to_global_polygons=to_global_lane_polygons,
			label_for_point=lane_id_for_point,
		)

	if calibration is not None:
		recording, pos_noise, jerk = _project_to_world(
			recording, calibration, anchors, pos_noise, jerk
		)

	states_by_frame = _smooth_recording(recording, pos_noise=pos_noise, jerk=jerk)
	if link_ids:
		states_by_frame = _apply_link_ids(states_by_frame, link_ids)
	if lane_ids:
		states_by_frame = _apply_lane_ids(states_by_frame, lane_ids)
	out.parent.mkdir(parents=True, exist_ok=True)
	_emit_trj(out, recording, states_by_frame, timestep_precision=timestep_precision)

	notes = []
	if exclusion_zones is not None:
		notes.append(f"dropped {dropped} excluded tracks")
	if link_zones is not None:
		notes.append(f"assigned link ids to {len(link_ids)} observations")
	if lane_zones is not None:
		notes.append(f"assigned lane ids to {len(lane_ids)} observations")
	if calibration is not None:
		notes.append("projected to world coordinates")
	note = f", {', '.join(notes)}" if notes else ""
	typer.echo(
		f"Post-processed {tracks} -> {out} "
		f"({len(states_by_frame)} frames, pos_noise={pos_noise:g}, jerk={jerk:g}{note})."
	)


def _filter_excluded(
	recording: TrackRecording, zones_path: Path, anchors_path: Path | None, min_fraction: float
) -> tuple[TrackRecording, int]:
	"""Drop whole tracks mostly inside an exclusion zone; return the survivors + drop count."""
	try:
		zones = load_exclusion_zones(zones_path)
		references = read_manifest(anchors_path) if anchors_path is not None else None
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	polygons = to_global_polygons(zones, _pose_for(references))
	excluded = excluded_track_ids(
		((o.track_id, Point2D(o.cx, o.cy)) for o in recording.observations),
		polygons,
		min_fraction=min_fraction,
	)
	if not excluded:
		return recording, 0
	kept = [o for o in recording.observations if o.track_id not in excluded]
	filtered = TrackRecording(metadata=recording.metadata, scale=recording.scale, observations=kept)
	return filtered, len(excluded)


_GlobalLabeledPolygons = tuple[tuple[int, tuple[Point2D, ...]], ...]


def _assign_labels[Zones: (LinkZones, LaneZones)](
	recording: TrackRecording,
	zones_path: Path,
	anchors_path: Path | None,
	*,
	load_zones: Callable[[Path], Zones],
	to_global_polygons: Callable[[Zones, Callable[[int], Transform2D]], _GlobalLabeledPolygons],
	label_for_point: Callable[[Point2D, _GlobalLabeledPolygons], int],
) -> dict[tuple[int, int], int]:
	"""Classify each surviving observation into a label (Link ID or Lane ID), keyed by
	``(track_id, frame_index)``.

	Shared by ``--link-zones`` and ``--lane-zones`` (src/tratrac/domain/ARCHITECTURE.md /
	docs/IMPLEMENTATION_PLAN.md Groups C1/C2): per-observation, not per-track, since a vehicle
	can cross links/lanes mid-track and SSAM's Link ID / Lane ID are per-VEHICLE-RECORD fields.
	Runs on image-space coordinates (before ``--calibration`` projects them), the same stage
	exclusion filtering already runs at.
	"""
	try:
		zones = load_zones(zones_path)
		references = read_manifest(anchors_path) if anchors_path is not None else None
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	polygons = to_global_polygons(zones, _pose_for(references))
	return {
		(o.track_id, o.frame_index): label_for_point(Point2D(o.cx, o.cy), polygons)
		for o in recording.observations
	}


def _apply_link_ids(
	states_by_frame: dict[int, list[VehicleState]], link_ids: dict[tuple[int, int], int]
) -> dict[int, list[VehicleState]]:
	"""Stamp each smoothed state's ``link_id`` from the pre-smoothing per-observation assignment."""
	return {
		frame_index: [
			replace(state, link_id=link_ids.get((state.vehicle_id, frame_index), 0))
			for state in states
		]
		for frame_index, states in states_by_frame.items()
	}


def _apply_lane_ids(
	states_by_frame: dict[int, list[VehicleState]], lane_ids: dict[tuple[int, int], int]
) -> dict[int, list[VehicleState]]:
	"""Stamp each smoothed state's ``lane_id`` from the pre-smoothing per-observation assignment."""
	return {
		frame_index: [
			replace(state, lane_id=lane_ids.get((state.vehicle_id, frame_index), 0))
			for state in states
		]
		for frame_index, states in states_by_frame.items()
	}


def _pose_for(references: list[ReferenceFrame] | None) -> Callable[[int], Transform2D]:
	"""Resolve an exclusion zone's reference-frame pose (raw -> global) from the anchor manifest.

	With a manifest (a moving-drone run), each pose comes from the matching anchor; a
	``reference_frame`` that is not an anchor is rejected. Without one (a static run), every
	pose is the identity (global == raw).
	"""
	if references is None:
		return lambda _reference_frame: Transform2D.identity()
	by_index = {ref.frame_index: ref.pose for ref in references}

	def pose(reference_frame: int) -> Transform2D:
		resolved = by_index.get(reference_frame)
		if resolved is None:
			raise typer.BadParameter(
				f"exclusion zone reference_frame {reference_frame} is not an anchor in the "
				"manifest; draw zones on an exported anchor frame."
			)
		return resolved

	return pose


def _project_to_world(
	recording: TrackRecording,
	calibration_path: Path,
	anchors_path: Path | None,
	pos_noise: float,
	jerk: float,
) -> tuple[TrackRecording, float, float]:
	"""Project the recording's image coordinates onto the metric world plane (MVP2).

	Fits one homography from the calibration correspondences (each mapped into the global
	frame via its anchor pose, exactly like exclusion zones), rewrites every observation's
	centroid and bbox into world metres, and stamps ``scale = 1.0`` so the downstream
	smoother/exporter emit world coordinates with ``DIMENSIONS.Scale = 1.0``.

	The projected coordinates are translated to a non-negative origin and the recording's
	``width``/``height`` are replaced with the **world** extent (metres) — not the pixel
	grid. This matters because the SSAM exporter writes those as the DIMENSIONS bounds and
	flips Y about ``height x scale``: leaving the pixel dimensions in place would make an
	external reader see metric coordinates against a pixel-sized canvas, flipped about the
	wrong axis (src/tratrac/infrastructure/export/SSAM_FORMAT.md + src/tratrac/application/WORLD_PROJECTION.md). The translation discards the operator's
	absolute world origin, which is arbitrary in Approach A and irrelevant to the
	translation-invariant conflict analytics. ``pos_noise``/``jerk`` are converted from
	pixels into world units by the homography's local scale, preserving the smoother's
	behaviour.
	"""
	try:
		calibration = load_calibration(calibration_path)
		references = read_manifest(anchors_path) if anchors_path is not None else None
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	pose_for = _pose_for(references)
	projector, scale = _fit_projector(calibration, pose_for)

	projected = [_project_observation(o, projector) for o in recording.observations]
	world_recording = _normalize_world_recording(recording.metadata, projected)
	return world_recording, pos_noise * scale, jerk * scale * scale


def _fit_projector(
	calibration: Calibration, pose_for: Callable[[int], Transform2D]
) -> tuple[WorldProjector, float]:
	"""Fit a ``WorldProjector`` from calibration correspondences, plus a representative
	local scale (metres/pixel) for converting ``pos_noise``/``jerk`` into world units.

	A calibration authored on a single anchor (or a static run, where every correspondence's
	``reference_frame`` collapses to the same pose) fits one global homography
	(``SingleHomographyProjector``) — the original Approach A behaviour, unchanged. A
	calibration spanning multiple anchors (Group C3, ``docs/IMPLEMENTATION_PLAN.md``) fits one
	homography per anchor (``PerAnchorWorldProjector``), grouping correspondences by their
	``reference_frame``; each group needs its own >= 4 correspondences (``compute_homography``
	enforces this and its ``ValueError`` is reported the same way for either path).
	"""
	by_anchor: dict[int, list[Correspondence]] = defaultdict(list)
	for correspondence in calibration.correspondences:
		by_anchor[correspondence.reference_frame].append(correspondence)

	if len(by_anchor) == 1:
		[(anchor, group)] = by_anchor.items()
		matrix = _fit_homography(group, pose_for)
		single: WorldProjector = SingleHomographyProjector(matrix)
		return single, local_scale_at(single, _centroid(group, pose_for), frame_index=anchor)

	homographies: dict[int, NDArray[np.float64]] = {}
	scales: list[float] = []
	for anchor, group in by_anchor.items():
		homographies[anchor] = _fit_homography(group, pose_for)
		per_anchor = SingleHomographyProjector(homographies[anchor])
		scales.append(local_scale_at(per_anchor, _centroid(group, pose_for), frame_index=anchor))
	# One global pos_noise/jerk feeds the smoother for the whole recording (it isn't
	# per-observation), so multiple anchors' local scales are averaged into one
	# representative value rather than tracked per anchor.
	return PerAnchorWorldProjector(homographies), sum(scales) / len(scales)


def _fit_homography(
	group: list[Correspondence], pose_for: Callable[[int], Transform2D]
) -> NDArray[np.float64]:
	image_points = [pose_for(c.reference_frame).apply(c.image) for c in group]
	world_points = [c.world for c in group]
	try:
		return compute_homography(image_points, world_points)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc


def _centroid(group: list[Correspondence], pose_for: Callable[[int], Transform2D]) -> Point2D:
	points = [pose_for(c.reference_frame).apply(c.image) for c in group]
	return Point2D(sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))


def _normalize_world_recording(
	metadata: VideoMetadata, projected: list[TrackObservation]
) -> TrackRecording:
	"""Shift projected observations to a 0-origin and size the metadata to the world extent.

	Pads the bounding box by the largest projected vehicle dimension so the front/rear
	bumpers the smoother derives from each centroid stay inside the DIMENSIONS bounds. An
	empty recording keeps the original (pixel) metadata — there are no coordinates to size.
	"""
	if not projected:
		return TrackRecording(metadata=metadata, scale=1.0, observations=projected)
	pad = max(max(o.width, o.height) for o in projected)
	min_x = min(o.cx for o in projected) - pad
	min_y = min(o.cy for o in projected) - pad
	max_x = max(o.cx for o in projected) + pad
	max_y = max(o.cy for o in projected) + pad
	shifted = [replace(o, cx=o.cx - min_x, cy=o.cy - min_y) for o in projected]
	world_meta = replace(
		metadata,
		width=max(1, math.ceil(max_x - min_x)),
		height=max(1, math.ceil(max_y - min_y)),
	)
	return TrackRecording(metadata=world_meta, scale=1.0, observations=shifted)


def _project_observation(o: TrackObservation, projector: WorldProjector) -> TrackObservation:
	"""Map one observation's centroid + bbox extent into world metres.

	Drops ``angle``/``obb_w``/``obb_h`` (Group A7/A8 OBB fields): they're image-space
	quantities, and a general homography's rotation need not match the image axes, so
	carrying a raw image-space angle into world-space heading selection would be wrong
	rather than merely imprecise. Re-deriving a world-space OBB angle is out of scope
	here (see `src/tratrac/application/WORLD_PROJECTION.md`); the smoother's low-speed
	fallback drops back to the bbox-major-axis heuristic for a projected run instead.
	"""
	center = projector.to_world(Point2D(o.cx, o.cy), o.frame_index)
	left = projector.to_world(Point2D(o.cx - o.width / 2.0, o.cy), o.frame_index)
	right = projector.to_world(Point2D(o.cx + o.width / 2.0, o.cy), o.frame_index)
	top = projector.to_world(Point2D(o.cx, o.cy - o.height / 2.0), o.frame_index)
	bottom = projector.to_world(Point2D(o.cx, o.cy + o.height / 2.0), o.frame_index)
	return replace(
		o,
		cx=center.x,
		cy=center.y,
		width=math.hypot(right.x - left.x, right.y - left.y),
		height=math.hypot(bottom.x - top.x, bottom.y - top.y),
		angle=None,
		obb_w=None,
		obb_h=None,
	)


def _smooth_recording(
	recording: TrackRecording, *, pos_noise: float, jerk: float
) -> dict[int, list[VehicleState]]:
	"""Group observations by track, smooth each, and regroup the states by frame index."""
	by_track: dict[int, list[TrackObservation]] = defaultdict(list)
	for observation in recording.observations:
		by_track[observation.track_id].append(observation)

	from tqdm import tqdm

	fps = recording.metadata.fps
	states_by_frame: dict[int, list[VehicleState]] = defaultdict(list)
	for track_id, observations in tqdm(by_track.items(), desc="Smoothing", unit="track"):
		observations.sort(key=lambda o: o.frame_index)
		samples = [
			TrackSample(
				frame_index=o.frame_index,
				timestamp_seconds=o.frame_index / fps,
				center=Point2D(o.cx, o.cy),
				width=o.width,
				height=o.height,
				angle=o.angle,
				oriented_size=(o.obb_w, o.obb_h)
				if o.obb_w is not None and o.obb_h is not None
				else None,
			)
			for o in observations
		]
		states = smooth_to_states(
			track_id, samples, recording.scale, pos_noise=pos_noise, jerk=jerk
		)
		for sample, state in zip(samples, states, strict=True):
			states_by_frame[sample.frame_index].append(state)
	return states_by_frame


def _emit_trj(
	out: Path,
	recording: TrackRecording,
	states_by_frame: dict[int, list[VehicleState]],
	*,
	timestep_precision: float,
) -> None:
	"""Write the smoothed .trj from the per-frame states, optionally decimating TIMESTEPs."""
	fps = recording.metadata.fps
	exporter: TrajectoryExporter = SsamTrjExporter(out, recording.metadata, scale=recording.scale)
	if timestep_precision > 0.0:
		# Thin the exported TIMESTEP stream; the smoothing still uses every observation.
		exporter = DecimatingTrajectoryExporter(
			exporter, min_interval_seconds=timestep_precision, fps=fps
		)
	with exporter:
		for frame_index in sorted(states_by_frame):
			exporter.emit_frame(frame_index / fps, states_by_frame[frame_index])


if __name__ == "__main__":
	app()
