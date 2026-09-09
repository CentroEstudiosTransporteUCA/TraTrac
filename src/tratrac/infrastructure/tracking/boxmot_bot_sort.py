"""boxmot BoT-SORT adapter (IoU-only for MVP1; ReID arrives in MVP5).

boxmot 19.x exposes tracker classes under ``boxmot.trackers``. The top-level
``Boxmot`` orchestrator is intentionally avoided — we want fine-grained control
over per-frame updates inside the application pipeline.

OBB support (Group A5, `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md`): ``boxmot``'s
``BotSort`` natively tracks oriented boxes given a 7-column
``(cx, cy, w, h, angle, conf, cls)`` det array instead of the 6-column AABB one, confirmed
against ``boxmot.trackers.detection_layout``. The tracker infers this from the *first*
non-empty det array's column count and locks it in for the run — an empty first frame would
silently lock in AABB mode even for an OBB-capable detector — so ``is_obb`` is an explicit
constructor flag (decided from the run's detector, not sniffed per frame) rather than
relying on that auto-detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from boxmot.trackers import BotSort

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox

_VEHICLE_CLASS_TO_COCO_ID: dict[VehicleClass, int] = {
	VehicleClass.CAR: 2,
	VehicleClass.MOTORCYCLE: 3,
	VehicleClass.BUS: 5,
	VehicleClass.TRUCK: 7,
}


class BoxmotBotSortTracker:
	"""Wraps ``boxmot.trackers.BotSort`` behind the ``Tracker`` port."""

	def __init__(
		self,
		metadata: VideoMetadata,
		*,
		det_thresh: float,
		compensate_camera_motion: bool = True,
		is_obb: bool = False,
	) -> None:
		# det_thresh below the detector's threshold so the detector is the sole
		# gatekeeper. Aerial-domain detections often peak in the 0.25-0.4 range,
		# and BotSort's stock 0.3 would suppress legitimate low-confidence cars.
		#
		# When coordinate stabilization is on (src/tratrac/infrastructure/video/EGO_MOTION.md), detections are
		# already mapped into the stabilized frame before they reach the tracker, so
		# BoT-SORT must NOT also compensate camera motion — doing so would double-
		# correct. cmc_method=None disables its internal CMC. Left on for raw
		# (unstabilized) runs so a moving camera is still handled for association.
		cmc_method = "ecc" if compensate_camera_motion else None
		self._is_obb = is_obb
		self._tracker: Any = BotSort(
			reid_model=None,
			with_reid=False,
			frame_rate=round(metadata.fps),
			det_thresh=det_thresh,
			cmc_method=cmc_method,
			is_obb=is_obb,
		)

	def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedDetection]:
		dets_array = self._detections_to_array(detections)
		results = self._tracker.update(dets_array, frame.pixels)

		tracked: list[TrackedDetection] = []
		if results.size == 0:
			return tracked

		for row in results:
			if self._is_obb:
				# OBB layout: cx, cy, w, h, angle, id, conf, cls, det_ind.
				det_ind = int(row[8])
				track_id, score = int(row[5]), float(row[6])
			else:
				# AABB layout: x1, y1, x2, y2, id, conf, cls, det_ind.
				det_ind = int(row[7])
				track_id, score = int(row[4]), float(row[5])
			# det_ind == -1 means the track is being held by motion prediction
			# with no fresh detection this frame. MVP1 skips these — long-term
			# track survival is MVP5 work.
			if det_ind < 0 or det_ind >= len(detections):
				continue
			original = detections[det_ind]
			if self._is_obb:
				cx, cy, w, h, angle = (float(v) for v in row[:5])
				detection = Detection(
					bbox=_obb_to_aabb(cx, cy, w, h, angle),
					score=score,
					vehicle_class=original.vehicle_class,
					angle=angle,
					oriented_size=(w, h),
				)
			else:
				x1, y1, x2, y2 = (float(v) for v in row[:4])
				detection = Detection(
					bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
					score=score,
					vehicle_class=original.vehicle_class,
				)
			tracked.append(TrackedDetection(track_id=track_id, detection=detection))
		return tracked

	def _detections_to_array(self, detections: Sequence[Detection]) -> np.ndarray:
		if not detections:
			return np.empty((0, 7 if self._is_obb else 6), dtype=np.float32)
		if self._is_obb:
			return np.array(
				[
					[
						d.bbox.center.x,
						d.bbox.center.y,
						*(
							d.oriented_size
							if d.oriented_size is not None
							else (d.bbox.width, d.bbox.height)
						),
						d.angle if d.angle is not None else 0.0,
						d.score,
						_VEHICLE_CLASS_TO_COCO_ID[d.vehicle_class],
					]
					for d in detections
				],
				dtype=np.float32,
			)
		return np.array(
			[
				[
					d.bbox.x,
					d.bbox.y,
					d.bbox.x + d.bbox.width,
					d.bbox.y + d.bbox.height,
					d.score,
					_VEHICLE_CLASS_TO_COCO_ID[d.vehicle_class],
				]
				for d in detections
			],
			dtype=np.float32,
		)


def _obb_to_aabb(cx: float, cy: float, w: float, h: float, angle: float) -> BoundingBox:
	"""The axis-aligned enclosing box of a rotated ``(cx, cy, w, h, angle)`` rectangle.

	``Detection.bbox`` stays the AABB unconditionally (ORB masking, IoU association) even
	for an OBB detection — see ``domain/detection.py``.
	"""
	cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
	half_w = (w * cos_a + h * sin_a) / 2.0
	half_h = (w * sin_a + h * cos_a) / 2.0
	return BoundingBox(x=cx - half_w, y=cy - half_h, width=2.0 * half_w, height=2.0 * half_h)
