"""Coordinate stabilization: map a raw-frame detection into the stabilized frame.

MVP1.9 stabilizes *coordinates*, not pixels (see vault/05_75_mvp1_9.md). The
detector and tracker work on the raw, full-resolution frame — so no car is ever
cropped into a black border — and each detection's box is then transformed into
the keyframe-anchored global frame before tracking, so the exported trajectory is
free of drone ego-motion.

A detection is a box, not just a point. We transform its centre exactly and scale
its size by the transform's uniform scale factor. The box stays axis-aligned: the
similarity's rotation is not re-fitted onto the box. This is exact for the centroid
trajectory (what velocity and heading are derived from) and correctly zoom-
normalises length/width; it only approximates the bounding box's *shape* under
rotation, which downstream uses solely for the never-moved-track fallback heading.
"""

from __future__ import annotations

from tratrac.domain.detection import Detection
from tratrac.domain.geometry import BoundingBox, Transform2D


def apply_transform(detection: Detection, transform: Transform2D) -> Detection:
	"""Return ``detection`` with its bounding box mapped through ``transform``.

	The box centre is transformed exactly; width and height are scaled by the
	transform's uniform scale factor. Score and class are unchanged.
	"""
	box = detection.bbox
	centre = transform.apply(box.center)
	scale = transform.scale
	width = box.width * scale
	height = box.height * scale
	return Detection(
		bbox=BoundingBox(
			x=centre.x - width / 2.0,
			y=centre.y - height / 2.0,
			width=width,
			height=height,
		),
		score=detection.score,
		vehicle_class=detection.vehicle_class,
	)


class EgoMotionStabilizer:
	"""``DetectionStabilizer`` that maps each detection into the global frame via the pose.

	The concrete stabilize step for a `--stabilize` run. A first-class collaborator (rather
	than an inline loop) so it sits behind the ``DetectionStabilizer`` port and is timeable
	like the other steps (vault/15_step_timing.md)."""

	def stabilize(self, detections: list[Detection], transform: Transform2D) -> list[Detection]:
		return [apply_transform(detection, transform) for detection in detections]


class NullDetectionStabilizer:
	"""``DetectionStabilizer`` Null Object: pass detections through unchanged (no ego-motion)."""

	def stabilize(self, detections: list[Detection], transform: Transform2D) -> list[Detection]:
		del transform  # Null Object: no pose to apply
		return detections
