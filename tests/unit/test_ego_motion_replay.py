"""Tests for ReplayEgoMotionEstimator and the transform-CSV round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.csv import CsvTransformSink, read_transforms
from tratrac.infrastructure.video.ego_motion_replay import ReplayEgoMotionEstimator

_PIXELS = np.zeros((4, 4, 3), dtype=np.uint8)  # replay never reads pixels


def _frame(index: int) -> Frame:
	return Frame(index=index, pixels=_PIXELS)


class TestReadTransforms:
	def test_round_trips_a_sink_file(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.csv"
		first = Transform2D(a=1.0, b=0.0, tx=2.0, c=0.0, d=1.0, ty=3.0)
		second = Transform2D(a=0.5, b=0.1, tx=-4.0, c=-0.1, d=0.5, ty=5.0)
		with CsvTransformSink(path) as sink:
			sink.record(FrameTransform(frame_index=0, transform=first))
			sink.record(FrameTransform(frame_index=9, transform=second))
		assert read_transforms(path) == {0: first, 9: second}

	def test_malformed_row_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.csv"
		path.write_text("frame,a,b,c,d,tx,ty\n0,oops,0,0,1,0,0\n")
		with pytest.raises(ValueError, match="not a valid transform CSV"):
			read_transforms(path)


class TestReplayEgoMotionEstimator:
	def test_serves_recorded_pose_by_frame_index(self) -> None:
		pose = Transform2D(a=1.0, b=0.0, tx=7.0, c=0.0, d=1.0, ty=-2.0)
		estimator = ReplayEgoMotionEstimator({0: Transform2D.identity(), 5: pose})
		assert estimator.estimate(_frame(0)) == Transform2D.identity()
		assert estimator.estimate(_frame(5)) == pose
		assert estimator.current_transform == pose

	def test_missing_frame_raises(self) -> None:
		estimator = ReplayEgoMotionEstimator({0: Transform2D.identity()})
		with pytest.raises(ValueError, match="no recorded transform for frame 3"):
			estimator.estimate(_frame(3))
