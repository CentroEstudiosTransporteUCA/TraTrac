"""Scout pass: discover a clip's keyframe anchors and persist the ego-motion schedule.

Runs ORB ego-motion only (no detector/tracker) over every frame, recording the
per-frame transform (reusing ``CsvTransformSink``) and, via the anchor callback,
writing each keyframe-anchor frame as a PNG plus a manifest the operator draws
exclusion zones on. The real run later **replays** the transform CSV, so its poses
match the scout exactly. See vault/21_exclusion_zones.md.

ORB sees vehicles unmasked here (there are no detections to mask): RANSAC already
rejects moving-vehicle matches, and moving-drone footage is ground-dominated, so
this is acceptable. The detector could be added later if a clip needs it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import VideoSource
from tratrac.infrastructure.scout.manifest import ReferenceFrame, write_manifest
from tratrac.infrastructure.transform.csv import CsvTransformSink
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.video.ego_motion_orb import OrbEgoMotionEstimator

ImageWriter = Callable[[Path, NDArray[np.uint8]], None]


def _write_png(path: Path, pixels: NDArray[np.uint8]) -> None:
	cv2.imwrite(str(path), pixels)


def run_scout(
	source: VideoSource,
	out_dir: Path,
	*,
	n_features: int,
	match_ratio: float,
	min_matches: int,
	ransac_threshold: float,
	min_anchor_overlap: float,
	video_label: str,
	transforms_name: str = "transforms.csv",
	manifest_name: str = "refs_manifest.json",
	image_writer: ImageWriter = _write_png,
) -> list[ReferenceFrame]:
	"""Run the scout over an (already-open) ``source``, writing into ``out_dir``.

	Returns the discovered reference frames (also written to the manifest). The
	transform CSV (``out_dir/transforms_name``) carries every frame's pose for the
	real run to replay.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	references: list[ReferenceFrame] = []
	# The estimator calls this synchronously during estimate(frame); we drain it
	# right after, while ``frame`` is still the frame that became the anchor.
	pending: list[tuple[int, Transform2D]] = []
	estimator = OrbEgoMotionEstimator(
		n_features=n_features,
		match_ratio=match_ratio,
		min_matches=min_matches,
		ransac_threshold=ransac_threshold,
		min_anchor_overlap=min_anchor_overlap,
		anchor_observer=lambda index, pose: pending.append((index, pose)),
	)
	with CsvTransformSink(out_dir / transforms_name) as sink:
		recording = RecordingEgoMotionEstimator(estimator, sink)
		for frame in source.frames():
			pending.clear()
			recording.estimate(frame)
			for index, pose in pending:
				image_name = f"frame_{index}.png"
				image_writer(out_dir / image_name, frame.pixels)
				references.append(ReferenceFrame(index, pose, image_name))
	write_manifest(
		out_dir / manifest_name, references, video=video_label, transforms_name=transforms_name
	)
	return references
