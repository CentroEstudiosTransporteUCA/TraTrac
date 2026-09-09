"""Tests for the YOLO-OBB detector adapter: class mapping, score validation, and empty results.
``ultralytics.YOLO`` is mocked out — no checkpoint download, no real inference."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tratrac.domain.detection import VehicleClass
from tratrac.domain.frame import Frame
from tratrac.infrastructure.detection.yolo_obb import YoloObbDetector


class _Tensor:
	"""Minimal stand-in for a torch.Tensor: only the .cpu().numpy() chain this adapter uses."""

	def __init__(self, array: np.ndarray) -> None:
		self._array = array

	def cpu(self) -> _Tensor:
		return self

	def numpy(self) -> np.ndarray:
		return self._array


class _FakeObb:
	def __init__(self, xywhr: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
		self.shape = xywhr.shape
		self.xywhr = _Tensor(xywhr)
		self.conf = _Tensor(conf)
		self.cls = _Tensor(cls)


class _FakeResult:
	def __init__(self, obb: _FakeObb | None, names: dict[int, str]) -> None:
		self.obb = obb
		self.names = names


def _frame() -> Frame:
	return Frame(index=0, pixels=np.zeros((100, 100, 3), dtype=np.uint8))


def _detector_with_results(results: list[_FakeResult]) -> tuple[YoloObbDetector, MagicMock]:
	with patch("tratrac.infrastructure.detection.yolo_obb.YOLO") as mock_yolo:
		model = MagicMock()
		model.predict.return_value = results
		mock_yolo.return_value = model
		detector = YoloObbDetector("dummy-checkpoint.pt", device="cpu", score_threshold=0.3)
	return detector, model


class TestConstruction:
	def test_rejects_out_of_range_threshold(self) -> None:
		with (
			patch("tratrac.infrastructure.detection.yolo_obb.YOLO"),
			pytest.raises(ValueError, match="score_threshold"),
		):
			YoloObbDetector("dummy.pt", device="cpu", score_threshold=1.5)


class TestDetect:
	def test_maps_known_labels_and_carries_orientation(self) -> None:
		names = {0: "car", 1: "truck"}
		xywhr = np.array([[50.0, 40.0, 10.0, 5.0, 0.3], [20.0, 20.0, 8.0, 4.0, 0.0]])
		obb = _FakeObb(xywhr=xywhr, conf=np.array([0.9, 0.8]), cls=np.array([0, 1]))
		detector, model = _detector_with_results([_FakeResult(obb, names)])

		detections = detector.detect(_frame())

		assert len(detections) == 2
		car = detections[0]
		assert car.vehicle_class is VehicleClass.CAR
		assert car.score == pytest.approx(0.9)
		assert car.angle == pytest.approx(0.3)
		assert car.oriented_size == pytest.approx((10.0, 5.0))
		truck = detections[1]
		assert truck.vehicle_class is VehicleClass.TRUCK

		model.predict.assert_called_once()
		assert model.predict.call_args.kwargs["conf"] == pytest.approx(0.3)
		assert model.predict.call_args.kwargs["device"] == "cpu"

	def test_unmapped_label_is_dropped(self) -> None:
		names = {0: "pedestrian"}
		xywhr = np.array([[50.0, 40.0, 10.0, 5.0, 0.0]])
		obb = _FakeObb(xywhr=xywhr, conf=np.array([0.9]), cls=np.array([0]))
		detector, _ = _detector_with_results([_FakeResult(obb, names)])
		assert detector.detect(_frame()) == []

	def test_bucket_classes_map_to_car(self) -> None:
		names = {0: "taxi", 1: "other_vehicle"}
		xywhr = np.array([[1.0, 1.0, 2.0, 1.0, 0.0], [3.0, 3.0, 2.0, 1.0, 0.0]])
		obb = _FakeObb(xywhr=xywhr, conf=np.array([0.5, 0.5]), cls=np.array([0, 1]))
		detector, _ = _detector_with_results([_FakeResult(obb, names)])
		assert all(d.vehicle_class is VehicleClass.CAR for d in detector.detect(_frame()))

	def test_bike_maps_to_motorcycle(self) -> None:
		names = {0: "bike"}
		xywhr = np.array([[1.0, 1.0, 2.0, 1.0, 0.0]])
		obb = _FakeObb(xywhr=xywhr, conf=np.array([0.5]), cls=np.array([0]))
		detector, _ = _detector_with_results([_FakeResult(obb, names)])
		[detection] = detector.detect(_frame())
		assert detection.vehicle_class is VehicleClass.MOTORCYCLE

	def test_no_obb_on_result_returns_empty(self) -> None:
		detector, _ = _detector_with_results([_FakeResult(None, {})])
		assert detector.detect(_frame()) == []

	def test_empty_obb_returns_empty(self) -> None:
		obb = _FakeObb(xywhr=np.empty((0, 5)), conf=np.empty((0,)), cls=np.empty((0,)))
		detector, _ = _detector_with_results([_FakeResult(obb, {})])
		assert detector.detect(_frame()) == []

	def test_no_results_returns_empty(self) -> None:
		detector, _ = _detector_with_results([])
		assert detector.detect(_frame()) == []

	def test_bbox_is_the_enclosing_aabb_of_the_oriented_box(self) -> None:
		names = {0: "car"}
		xywhr = np.array([[10.0, 10.0, 4.0, 2.0, math.pi / 2]])
		obb = _FakeObb(xywhr=xywhr, conf=np.array([0.9]), cls=np.array([0]))
		detector, _ = _detector_with_results([_FakeResult(obb, names)])
		[detection] = detector.detect(_frame())
		# Rotated 90deg: the enclosing box swaps width/height relative to the raw w/h.
		assert detection.bbox.width == pytest.approx(2.0, abs=1e-6)
		assert detection.bbox.height == pytest.approx(4.0, abs=1e-6)
