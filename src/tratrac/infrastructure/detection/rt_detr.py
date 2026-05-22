"""HuggingFace transformers RT-DETR detector adapter.

Defaults to ``PekingU/rtdetr_r18vd`` — smaller checkpoint, tractable on CPU for
MVP1 dev. Swap to ``PekingU/rtdetr_r50vd_coco_o365`` for stronger detections
once a GPU is available.
"""

from __future__ import annotations

from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrForObjectDetection

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import BoundingBox

_COCO_LABEL_TO_VEHICLE_CLASS: dict[str, VehicleClass] = {
	"car": VehicleClass.CAR,
	"motorcycle": VehicleClass.MOTORCYCLE,
	"bus": VehicleClass.BUS,
	"truck": VehicleClass.TRUCK,
}


class RtDetrDetector:
	"""Wraps a HuggingFace RT-DETR model behind the ``Detector`` port."""

	def __init__(
		self,
		checkpoint: str = "PekingU/rtdetr_r18vd",
		device: str = "cpu",
		score_threshold: float = 0.25,
	) -> None:
		if not 0.0 <= score_threshold <= 1.0:
			raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}.")
		self._processor: Any = AutoImageProcessor.from_pretrained(checkpoint)
		self._model: Any = RTDetrForObjectDetection.from_pretrained(checkpoint).to(device)
		self._model.eval()
		self._device = device
		self._score_threshold = score_threshold
		self._id2label: dict[int, str] = self._model.config.id2label

	def detect(self, frame: Frame) -> list[Detection]:
		rgb = cv2.cvtColor(frame.pixels, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)

		inputs = self._processor(images=image, return_tensors="pt").to(self._device)
		with torch.no_grad():
			outputs = self._model(**inputs)

		target_size = torch.tensor([[image.height, image.width]], device=self._device)
		processed = self._processor.post_process_object_detection(
			outputs, target_sizes=target_size, threshold=self._score_threshold
		)[0]

		detections: list[Detection] = []
		for score, label_id, box in zip(
			processed["scores"], processed["labels"], processed["boxes"], strict=True
		):
			label = self._id2label[int(label_id.item())]
			vehicle_class = _COCO_LABEL_TO_VEHICLE_CLASS.get(label)
			if vehicle_class is None:
				continue
			x1, y1, x2, y2 = (float(c) for c in box.tolist())
			detections.append(
				Detection(
					bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
					score=float(score.item()),
					vehicle_class=vehicle_class,
				)
			)
		return detections
