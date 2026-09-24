"""YOLO-OBB detector adapter (MVP1.5, replanned — Group A4, GitHub Issues).

Wraps `ultralytics`'s OBB task (`result.obb`, distinct from the plain-detect `result.boxes`
the MVP1 `yolov8_visdrone.py` adapter uses) so each detection carries the model's own
oriented-box angle and size instead of only an axis-aligned box. See
`src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` for the full MVP1.5 plan.

**Not yet the default detector, and not yet backed by a fine-tuned checkpoint** — `checkpoint`
must be pointed at a UAV-OBB-fine-tuned `yolo11-obb`/`yolo26-obb` weights file once one exists
(GitHub Issues Group A2, blocked on a GPU + the UAV-OBB dataset, neither of
which this adapter's construction depends on). Until then this class is exercised only by its
unit tests (a real checkpoint's class order and angle convention are unverified — see the two
open questions in `DETECTOR_CHOICE.md`).
"""

from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import oriented_box_to_aabb

# UAV-OBB's six classes (Group A2, GitHub Issues), mapped into TraTrac's
# VehicleClass. "bike" buckets into MOTORCYCLE (TraTrac has no separate bicycle/moped class);
# "other_vehicle"/"taxi" bucket into CAR, mirroring the "van -> car bucket" precedent already
# used by the MVP1 YOLOv8-VisDrone adapter. Keyed by lowercased label name, not class index —
# a checkpoint's index order isn't guaranteed stable across training runs, but its label
# strings (read from the model itself) are.
_UAV_OBB_LABEL_TO_VEHICLE_CLASS: dict[str, VehicleClass] = {
	"bike": VehicleClass.MOTORCYCLE,
	"bus": VehicleClass.BUS,
	"car": VehicleClass.CAR,
	"other_vehicle": VehicleClass.CAR,
	"taxi": VehicleClass.CAR,
	"truck": VehicleClass.TRUCK,
}


class YoloObbDetector:
	"""Wraps an `ultralytics` YOLO-OBB checkpoint behind the `Detector` port."""

	def __init__(self, checkpoint: str, device: str, score_threshold: float) -> None:
		if not 0.0 <= score_threshold <= 1.0:
			raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}.")
		self._model: Any = YOLO(checkpoint)
		self._device = device
		self._score_threshold = score_threshold

	def detect(self, frame: Frame) -> list[Detection]:
		# Ultralytics accepts BGR ndarrays directly (frame.pixels is BGR from OpenCV).
		results = self._model.predict(
			source=frame.pixels,
			conf=self._score_threshold,
			device=self._device,
			verbose=False,
		)
		if not results:
			return []
		first = results[0]
		obb = first.obb
		if obb is None or obb.shape[0] == 0:
			return []

		names: dict[int, str] = first.names
		xywhr = obb.xywhr.cpu().numpy()
		confs = obb.conf.cpu().numpy()
		cls_ids = obb.cls.cpu().numpy().astype(int)

		detections: list[Detection] = []
		for (cx, cy, w, h, angle), conf, cls_id in zip(xywhr, confs, cls_ids, strict=True):
			vehicle_class = _UAV_OBB_LABEL_TO_VEHICLE_CLASS.get(names[int(cls_id)].lower())
			if vehicle_class is None:
				continue
			cx_f, cy_f, w_f, h_f, angle_f = (float(v) for v in (cx, cy, w, h, angle))
			detections.append(
				Detection(
					bbox=oriented_box_to_aabb(cx_f, cy_f, w_f, h_f, angle_f),
					score=float(conf),
					vehicle_class=vehicle_class,
					angle=angle_f,
					oriented_size=(w_f, h_f),
				)
			)
		return detections
