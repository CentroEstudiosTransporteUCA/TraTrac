"""YOLOv8-VisDrone detector adapter.

**MVP1 emergency adapter.** The vault picks RT-DETR over YOLO for long-term
aerial robustness (see `vault/03_tech_stack.md`), but the COCO-pretrained
RT-DETR-R18 is unusable on aerial inputs and we can't fine-tune in the MVP1
timebox. This adapter wraps the community-trained `Mahadih534/YoloV8-VisDrone`
checkpoint on HuggingFace (YOLOv8 fine-tuned on VisDrone: 10K aerial images,
2.6M boxes) so MVP1 has a detector that actually sees aerial cars.

To remove later (once RT-DETR fine-tuning lands):
	1. Delete this file.
	2. Drop the `yolov8_visdrone` option from `DetectorChoice` in `cli.py`.
	3. `uv remove ultralytics dill`.
"""

from __future__ import annotations

from typing import Any

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import BoundingBox

# VisDrone class IDs the model emits, mapped to TraTrac's VehicleClass.
# Excluded: 0 pedestrian, 1 people, 2 bicycle, 6 tricycle, 7 awning-tricycle
# (not relevant to vehicle trajectory analytics for MVP1).
_VISDRONE_ID_TO_VEHICLE_CLASS: dict[int, VehicleClass] = {
	3: VehicleClass.CAR,
	4: VehicleClass.CAR,  # van -> car bucket
	5: VehicleClass.TRUCK,
	8: VehicleClass.BUS,
	9: VehicleClass.MOTORCYCLE,
}


class YoloV8VisDroneDetector:
	"""Wraps `Mahadih534/YoloV8-VisDrone` behind the `Detector` port."""

	def __init__(
		self,
		repo_id: str,
		filename: str,
		device: str,
		score_threshold: float,
	) -> None:
		if not 0.0 <= score_threshold <= 1.0:
			raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}.")
		weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
		self._model: Any = YOLO(weights_path)
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
		boxes = first.boxes
		if boxes is None or boxes.shape[0] == 0:
			return []

		detections: list[Detection] = []
		xyxy = boxes.xyxy.cpu().numpy()
		confs = boxes.conf.cpu().numpy()
		cls_ids = boxes.cls.cpu().numpy().astype(int)

		for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids, strict=True):
			vehicle_class = _VISDRONE_ID_TO_VEHICLE_CLASS.get(int(cls_id))
			if vehicle_class is None:
				continue
			detections.append(
				Detection(
					bbox=BoundingBox(
						x=float(x1),
						y=float(y1),
						width=float(x2 - x1),
						height=float(y2 - y1),
					),
					score=float(conf),
					vehicle_class=vehicle_class,
				)
			)
		return detections
