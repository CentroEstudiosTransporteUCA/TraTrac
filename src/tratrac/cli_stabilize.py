"""Typer entry point for the detector-free ego-motion pre-pass (``tratrac-stabilize``).

Walks VIDEO once with masked ORB + RANSAC (``OrbEgoMotionEstimator``), sourcing the
feature mask from operator-authored background zones instead of live detections, so
the detector never needs to run here at all — see "Detector-free ego-motion" in
src/tratrac/infrastructure/video/EGO_MOTION.md for why this is possible (and why a
detection-fed mask can't just be swapped for a cheaper one) and the operator
workflow this fits into.

Writes the same per-frame transform sidecar (``infrastructure/transform/sink.py``)
and anchor manifest (``infrastructure/anchors/sink.py``) a stabilized ``tratrac``
run would, so ``tratrac --config …`` can read the result back via
``ego_motion.transforms_in`` and run its single detector pass against already-known
ego-motion — no ORB there at all.

No detector, no tracker, no track record: this tool only ever drives the
ego-motion estimator across the whole clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from tratrac.domain.geometry import Transform2D
from tratrac.infrastructure.anchors.recording import AnchorRecordingEgoMotionEstimator
from tratrac.infrastructure.anchors.sink import AnchorManifestSink
from tratrac.infrastructure.background.json import load_background_zones
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.transform.sink import CoordinateTransformSink
from tratrac.infrastructure.video.ego_motion_orb import (
	BackgroundZoneMaskSource,
	OrbEgoMotionEstimator,
)
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac-stabilize",
	help="Detector-free ego-motion pre-pass: build the transforms file before the main run.",
	no_args_is_help=True,
)


@app.command()
def stabilize(
	video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="The clip to walk.")],
	background_zones: Annotated[
		Path,
		typer.Option(
			"--background-zones",
			exists=True,
			dir_okay=False,
			help="Operator-authored background_zones.json (an external tool draws these on "
			"the video, same role as calibration.json/zones.json today — see "
			"src/tratrac/infrastructure/video/EGO_MOTION.md).",
		),
	],
	out: Annotated[
		Path,
		typer.Option(
			"--out", "-o", dir_okay=False, help="Per-frame ego-motion transform sidecar (JSONL)."
		),
	],
	anchors_dir: Annotated[
		Path,
		typer.Option(
			"--anchors-dir",
			file_okay=False,
			help="Directory for keyframe-anchor PNGs + manifest.json (draw --calibration/"
			"--exclusion-zones correspondences on these, same as a stabilized tratrac run's "
			"anchors — see src/tratrac/application/EXCLUSION_ZONES.md).",
		),
	],
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
	"""Walk VIDEO once (no detector) and write the ego-motion transforms file + anchors."""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	if anchors_dir.exists() and anchors_dir.is_file():
		raise typer.BadParameter(f"{anchors_dir} must be a directory, not a file.")
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
		anchor_poses: list[Transform2D] = []
		orb = OrbEgoMotionEstimator(
			n_features=n_features,
			match_ratio=match_ratio,
			min_matches=min_matches,
			ransac_threshold=ransac_threshold,
			min_anchor_overlap=min_anchor_overlap,
			mask_source=BackgroundZoneMaskSource(zones),
			anchor_observer=lambda _index, pose: anchor_poses.append(pose),
		)
		with (
			CoordinateTransformSink(out) as transform_sink,
			AnchorManifestSink(anchors_dir, video_label=str(video)) as anchor_sink,
		):
			estimator = AnchorRecordingEgoMotionEstimator(
				RecordingEgoMotionEstimator(orb, transform_sink), anchor_poses, anchor_sink
			)
			total = source.metadata.total_frames if source.metadata.total_frames > 0 else None
			n_frames = 0
			for frame in tqdm(source.frames(), total=total, desc="Stabilizing", unit="frame"):
				estimator.estimate(frame)
				n_frames += 1

	typer.echo(
		f"Stabilized {n_frames} frames -> {out} (anchors: {anchors_dir}). "
		"Point tratrac's ego_motion.transforms_in at the transforms file to run its "
		"single detector pass against already-known ego-motion."
	)


if __name__ == "__main__":
	app()
