"""boxmot BoT-SORT adapter (IoU-only for MVP1; ReID arrives in MVP5).

boxmot 19.x exposes tracker classes under ``boxmot.trackers``. The top-level
``Boxmot`` orchestrator is intentionally avoided — we want fine-grained control
over per-frame updates inside the application pipeline.
"""

from __future__ import annotations

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
	) -> None:
		# det_thresh below the detector's threshold so the detector is the sole
		# gatekeeper. Aerial-domain detections often peak in the 0.25-0.4 range,
		# and BotSort's stock 0.3 would suppress legitimate low-confidence cars.
		#
		# When coordinate stabilization is on (vault/05_75_mvp1_9.md), detections are
		# already mapped into the stabilized frame before they reach the tracker, so
		# BoT-SORT must NOT also compensate camera motion — doing so would double-
		# correct. cmc_method=None disables its internal CMC. Left on for raw
		# (unstabilized) runs so a moving camera is still handled for association.
		cmc_method = "ecc" if compensate_camera_motion else None
		self._tracker: Any = BotSort(
			reid_model=None,
			with_reid=False,
			frame_rate=round(metadata.fps),
			det_thresh=det_thresh,
			cmc_method=cmc_method,
		)

	def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedDetection]:
		dets_array = self._detections_to_array(detections)
		results = self._tracker.update(dets_array, frame.pixels)

		tracked: list[TrackedDetection] = []
		if results.size == 0:
			return tracked

		# AABB layout: x1, y1, x2, y2, id, conf, cls, det_ind.
		for row in results:
			det_ind = int(row[7])
			# det_ind == -1 means the track is being held by motion prediction
			# with no fresh detection this frame. MVP1 skips these — long-term
			# track survival is MVP5 work.
			if det_ind < 0 or det_ind >= len(detections):
				continue
			x1, y1, x2, y2 = (float(v) for v in row[:4])
			track_id = int(row[4])
			score = float(row[5])
			original = detections[det_ind]
			tracked.append(
				TrackedDetection(
					track_id=track_id,
					detection=Detection(
						bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
						score=score,
						vehicle_class=original.vehicle_class,
					),
				)
			)
		return tracked

	@staticmethod
	def _detections_to_array(detections: Sequence[Detection]) -> np.ndarray:
		if not detections:
			return np.empty((0, 6), dtype=np.float32)
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
