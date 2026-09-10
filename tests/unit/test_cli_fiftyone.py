"""Tests for the pure record/.trj -> FiftyOne-shaped-box conversion (`cli_fiftyone.py`).

Only the conversion functions are covered — `_build_dataset` needs a live `fiftyone` (and
its MongoDB backing store), which this environment can't run; see the module docstring.
"""

from __future__ import annotations

import math

import pytest

from tratrac.cli_fiftyone import record_frame_detections, trj_frame_detections
from tratrac.domain.detection import VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.ssam_trj import TrjFrame, TrjRecording
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording

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


class TestTrjFrameDetections:
	def test_axis_aligned_heading_matches_plain_bbox(self) -> None:
		trj = TrjRecording(
			scale=1.0,
			width=200,
			height=100,
			frames=[
				TrjFrame(
					timestamp_seconds=0.0,
					states=[
						VehicleState(
							vehicle_id=7,
							timestamp_seconds=0.0,
							centroid=Point2D(100.0, 50.0),
							heading=Heading(1.0, 0.0),
							dimensions=Dimensions(length=20.0, width=10.0),
							velocity=Vector2D(5.0, 0.0),
							acceleration=0.0,
						)
					],
				)
			],
		)
		by_frame = trj_frame_detections(trj, fps=10.0)
		# timestamp 0.0 * fps 10.0 -> frame 0.
		(det,) = by_frame[0]
		assert det.track_id == 7
		# Facing +x: length (20) extends along x, width (10) along y.
		assert det.x == pytest.approx((100.0 - 10.0) / 200.0)
		assert det.y == pytest.approx((50.0 - 5.0) / 100.0)
		assert det.width == pytest.approx(20.0 / 200.0)
		assert det.height == pytest.approx(10.0 / 100.0)

	def test_frame_index_from_timestamp_and_fps(self) -> None:
		trj = TrjRecording(
			scale=1.0,
			width=200,
			height=100,
			frames=[
				TrjFrame(
					timestamp_seconds=1.0,
					states=[
						VehicleState(
							vehicle_id=1,
							timestamp_seconds=1.0,
							centroid=Point2D(50.0, 50.0),
							heading=Heading(0.0, 1.0),
							dimensions=Dimensions(length=10.0, width=10.0),
							velocity=Vector2D(0.0, 1.0),
							acceleration=0.0,
						)
					],
				)
			],
		)
		by_frame = trj_frame_detections(trj, fps=30.0)
		assert list(by_frame) == [30]  # round(1.0 * 30.0)

	def test_perpendicular_heading_swaps_extents(self) -> None:
		trj = TrjRecording(
			scale=1.0,
			width=200,
			height=100,
			frames=[
				TrjFrame(
					timestamp_seconds=0.0,
					states=[
						VehicleState(
							vehicle_id=1,
							timestamp_seconds=0.0,
							centroid=Point2D(100.0, 50.0),
							heading=Heading.from_angle(math.pi / 2),  # facing +y
							dimensions=Dimensions(length=20.0, width=10.0),
							velocity=Vector2D(0.0, 5.0),
							acceleration=0.0,
						)
					],
				)
			],
		)
		by_frame = trj_frame_detections(trj, fps=10.0)
		(det,) = by_frame[0]
		# Facing +y: length (20) now extends along y, width (10) along x.
		assert det.width == pytest.approx(10.0 / 200.0, abs=1e-6)
		assert det.height == pytest.approx(20.0 / 100.0, abs=1e-6)
