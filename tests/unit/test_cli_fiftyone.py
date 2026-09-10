"""Tests for the pure record -> FiftyOne-shaped-box conversion (`cli_fiftyone.py`).

Only the conversion functions are covered — `_build_dataset` needs a live `fiftyone` (and
its MongoDB backing store), which this environment can't run; see the module docstring.
"""

from __future__ import annotations

import math

import pytest

from tratrac.cli_fiftyone import record_frame_detections, smoothed_frame_detections
from tratrac.domain.detection import VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording
from tratrac.infrastructure.tracks.smoothed_parquet import SmoothedObservation, SmoothedRecording

_META = VideoMetadata(width=200, height=100, fps=10.0, total_frames=10)


class TestRecordFrameDetections:
	def test_normalizes_box_by_frame_dimensions(self) -> None:
		recording = TrackRecording(
			metadata=_META,
			scale=1.0,
			observations=[
				TrackObservation(
					frame_index=0,
					track_id=1,
					cx=50.0,
					cy=25.0,
					width=20.0,
					height=10.0,
					vehicle_class=VehicleClass.CAR,
					score=0.9,
				)
			],
		)
		by_frame = record_frame_detections(recording)
		assert list(by_frame) == [0]
		(det,) = by_frame[0]
		assert det.track_id == 1
		assert det.x == pytest.approx(0.2)  # (50 - 10) / 200
		assert det.y == pytest.approx(0.2)  # (25 - 5) / 100
		assert det.width == pytest.approx(0.1)  # 20 / 200
		assert det.height == pytest.approx(0.1)  # 10 / 100
		assert det.label == "car"
		assert det.confidence == pytest.approx(0.9)

	def test_groups_multiple_observations_by_frame(self) -> None:
		recording = TrackRecording(
			metadata=_META,
			scale=1.0,
			observations=[
				TrackObservation(0, 1, 10.0, 10.0, 4.0, 4.0, VehicleClass.CAR, 0.5),
				TrackObservation(0, 2, 20.0, 20.0, 4.0, 4.0, VehicleClass.TRUCK, 0.5),
				TrackObservation(1, 1, 12.0, 10.0, 4.0, 4.0, VehicleClass.CAR, 0.5),
			],
		)
		by_frame = record_frame_detections(recording)
		assert {t.track_id for t in by_frame[0]} == {1, 2}
		assert {t.track_id for t in by_frame[1]} == {1}


class TestSmoothedFrameDetections:
	def test_axis_aligned_heading_matches_plain_bbox(self) -> None:
		recording = SmoothedRecording(
			metadata=_META,
			observations=[
				SmoothedObservation(
					frame_index=0,
					track_id=7,
					cx=100.0,
					cy=50.0,
					angle=0.0,  # facing +x
					length=20.0,
					width=10.0,
				)
			],
		)
		by_frame = smoothed_frame_detections(recording)
		(det,) = by_frame[0]
		assert det.track_id == 7
		# Facing +x: length (20) extends along x, width (10) along y.
		assert det.x == pytest.approx((100.0 - 10.0) / 200.0)
		assert det.y == pytest.approx((50.0 - 5.0) / 100.0)
		assert det.width == pytest.approx(20.0 / 200.0)
		assert det.height == pytest.approx(10.0 / 100.0)
		assert det.label == "vehicle"

	def test_groups_by_frame_index_directly(self) -> None:
		recording = SmoothedRecording(
			metadata=_META,
			observations=[
				SmoothedObservation(0, 1, 50.0, 50.0, 0.0, 10.0, 10.0),
				SmoothedObservation(30, 1, 60.0, 50.0, 0.0, 10.0, 10.0),
			],
		)
		by_frame = smoothed_frame_detections(recording)
		assert list(by_frame) == [0, 30]

	def test_perpendicular_heading_swaps_extents(self) -> None:
		recording = SmoothedRecording(
			metadata=_META,
			observations=[
				SmoothedObservation(
					frame_index=0,
					track_id=1,
					cx=100.0,
					cy=50.0,
					angle=math.pi / 2,  # facing +y
					length=20.0,
					width=10.0,
				)
			],
		)
		by_frame = smoothed_frame_detections(recording)
		(det,) = by_frame[0]
		# Facing +y: length (20) now extends along y, width (10) along x.
		assert det.width == pytest.approx(10.0 / 200.0, abs=1e-6)
		assert det.height == pytest.approx(20.0 / 100.0, abs=1e-6)
