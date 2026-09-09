"""Tests for the boxmot BoT-SORT adapter: AABB (MVP1) and OBB (Group A5) det-array shapes,
output-row parsing, and det_ind handling. ``boxmot.trackers.BotSort`` is mocked out — this
tests the adapter's array building/parsing, not boxmot's own association logic."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.tracking.boxmot_bot_sort import BoxmotBotSortTracker, _obb_to_aabb

_META = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=900)


def _frame() -> Frame:
	return Frame(index=0, pixels=np.zeros((1080, 1920, 3), dtype=np.uint8))


def _det(
	x: float = 10.0, y: float = 20.0, w: float = 4.0, h: float = 2.0, score: float = 0.8
) -> Detection:
	return Detection(
		bbox=BoundingBox(x=x, y=y, width=w, height=h), score=score, vehicle_class=VehicleClass.CAR
	)


def _obb_det(
	cx: float = 12.0,
	cy: float = 21.0,
	w: float = 4.2,
	h: float = 2.1,
	angle: float = 0.3,
	score: float = 0.8,
) -> Detection:
	return Detection(
		bbox=BoundingBox(x=cx - w / 2, y=cy - h / 2, width=w, height=h),
		score=score,
		vehicle_class=VehicleClass.CAR,
		angle=angle,
		oriented_size=(w, h),
	)


@patch("tratrac.infrastructure.tracking.boxmot_bot_sort.BotSort")
def _tracker(
	mock_botsort: MagicMock, *, is_obb: bool = False
) -> tuple[BoxmotBotSortTracker, MagicMock]:
	instance = MagicMock()
	mock_botsort.return_value = instance
	tracker = BoxmotBotSortTracker(_META, det_thresh=0.1, is_obb=is_obb)
	return tracker, instance


class TestConstruction:
	def test_passes_is_obb_through_to_botsort(self) -> None:
		with patch("tratrac.infrastructure.tracking.boxmot_bot_sort.BotSort") as mock_botsort:
			BoxmotBotSortTracker(_META, det_thresh=0.1, is_obb=True)
			assert mock_botsort.call_args.kwargs["is_obb"] is True

	def test_disables_cmc_when_stabilization_compensates_already(self) -> None:
		with patch("tratrac.infrastructure.tracking.boxmot_bot_sort.BotSort") as mock_botsort:
			BoxmotBotSortTracker(_META, det_thresh=0.1, compensate_camera_motion=False)
			assert mock_botsort.call_args.kwargs["cmc_method"] is None


class TestAabbPath:
	def test_builds_six_column_array(self) -> None:
		tracker, instance = _tracker()
		instance.update.return_value = np.empty((0, 8), dtype=np.float32)
		tracker.update(_frame(), [_det(x=10.0, y=20.0, w=4.0, h=2.0, score=0.8)])
		dets_array = instance.update.call_args[0][0]
		assert dets_array.shape == (1, 6)
		assert dets_array[0].tolist() == pytest.approx([10.0, 20.0, 14.0, 22.0, 0.8, 2.0])

	def test_empty_detections_gives_empty_six_column_array(self) -> None:
		tracker, instance = _tracker()
		instance.update.return_value = np.empty((0, 8), dtype=np.float32)
		tracker.update(_frame(), [])
		dets_array = instance.update.call_args[0][0]
		assert dets_array.shape == (0, 6)

	def test_parses_output_row_into_tracked_detection(self) -> None:
		tracker, instance = _tracker()
		# x1,y1,x2,y2,id,conf,cls,det_ind
		instance.update.return_value = np.array([[10.0, 20.0, 14.0, 22.0, 7.0, 0.75, 2.0, 0.0]])
		[tracked] = tracker.update(_frame(), [_det()])
		assert tracked.track_id == 7
		assert tracked.detection.score == pytest.approx(0.75)
		assert tracked.detection.bbox == BoundingBox(x=10.0, y=20.0, width=4.0, height=2.0)
		assert tracked.detection.angle is None

	def test_negative_det_ind_is_skipped(self) -> None:
		tracker, instance = _tracker()
		instance.update.return_value = np.array([[10.0, 20.0, 14.0, 22.0, 7.0, 0.75, 2.0, -1.0]])
		assert tracker.update(_frame(), [_det()]) == []

	def test_no_results_returns_empty(self) -> None:
		tracker, instance = _tracker()
		instance.update.return_value = np.empty((0, 8), dtype=np.float32)
		assert tracker.update(_frame(), [_det()]) == []


class TestObbPath:
	def test_builds_seven_column_array(self) -> None:
		tracker, instance = _tracker(is_obb=True)
		instance.update.return_value = np.empty((0, 9), dtype=np.float32)
		tracker.update(_frame(), [_obb_det(cx=12.0, cy=21.0, w=4.2, h=2.1, angle=0.3, score=0.8)])
		dets_array = instance.update.call_args[0][0]
		assert dets_array.shape == (1, 7)
		assert dets_array[0].tolist() == pytest.approx([12.0, 21.0, 4.2, 2.1, 0.3, 0.8, 2.0])

	def test_empty_detections_gives_empty_seven_column_array(self) -> None:
		tracker, instance = _tracker(is_obb=True)
		instance.update.return_value = np.empty((0, 9), dtype=np.float32)
		tracker.update(_frame(), [])
		dets_array = instance.update.call_args[0][0]
		assert dets_array.shape == (0, 7)

	def test_parses_output_row_with_angle_and_oriented_size(self) -> None:
		tracker, instance = _tracker(is_obb=True)
		# cx,cy,w,h,angle,id,conf,cls,det_ind
		instance.update.return_value = np.array([[12.0, 21.0, 4.2, 2.1, 0.3, 9.0, 0.9, 2.0, 0.0]])
		[tracked] = tracker.update(_frame(), [_obb_det()])
		assert tracked.track_id == 9
		assert tracked.detection.score == pytest.approx(0.9)
		assert tracked.detection.angle == pytest.approx(0.3)
		assert tracked.detection.oriented_size == pytest.approx((4.2, 2.1))

	def test_negative_det_ind_is_skipped(self) -> None:
		tracker, instance = _tracker(is_obb=True)
		instance.update.return_value = np.array([[12.0, 21.0, 4.2, 2.1, 0.3, 9.0, 0.9, 2.0, -1.0]])
		assert tracker.update(_frame(), [_obb_det()]) == []

	def test_missing_angle_falls_back_to_zero_in_the_det_array(self) -> None:
		# Defensive: a mixed-detector frame shouldn't crash the array build.
		tracker, instance = _tracker(is_obb=True)
		instance.update.return_value = np.empty((0, 9), dtype=np.float32)
		tracker.update(_frame(), [_det()])  # AABB-shaped Detection, no angle/oriented_size
		dets_array = instance.update.call_args[0][0]
		assert dets_array[0][4] == pytest.approx(0.0)  # angle column defaults to 0.0


class TestObbToAabb:
	def test_zero_angle_is_the_plain_box(self) -> None:
		box = _obb_to_aabb(cx=10.0, cy=10.0, w=4.0, h=2.0, angle=0.0)
		assert box == BoundingBox(x=8.0, y=9.0, width=4.0, height=2.0)

	def test_quarter_turn_swaps_extents(self) -> None:
		box = _obb_to_aabb(cx=10.0, cy=10.0, w=4.0, h=2.0, angle=math.pi / 2)
		assert box.width == pytest.approx(2.0, abs=1e-6)
		assert box.height == pytest.approx(4.0, abs=1e-6)

	def test_forty_five_degrees_grows_the_enclosing_box(self) -> None:
		box = _obb_to_aabb(cx=0.0, cy=0.0, w=2.0, h=2.0, angle=math.pi / 4)
		diag = 2.0 * math.sqrt(2.0)
		assert box.width == pytest.approx(diag, abs=1e-6)
		assert box.height == pytest.approx(diag, abs=1e-6)
