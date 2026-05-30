"""Tests for the per-frame transform sink and recording decorator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.csv import CsvTransformSink
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator


class _RecordingSink:
	def __init__(self) -> None:
		self.records: list[FrameTransform] = []

	def record(self, frame_transform: FrameTransform) -> None:
		self.records.append(frame_transform)


class _StubEstimator:
	"""Returns the supplied transforms in order, one per ``estimate`` call."""

	def __init__(self, transforms: list[Transform2D]) -> None:
		self._transforms = transforms
		self.calls: list[int] = []

	def estimate(self, frame: Frame) -> Transform2D:
		self.calls.append(frame.index)
		return self._transforms[len(self.calls) - 1]


def _frame(index: int = 0) -> Frame:
	return Frame(index=index, pixels=np.zeros((4, 4, 3), dtype=np.uint8))


class TestCsvTransformSink:
	def test_writes_header_and_one_row_per_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.csv"
		with CsvTransformSink(path) as sink:
			sink.record(FrameTransform(0, Transform2D.identity()))
			sink.record(FrameTransform(1, Transform2D(a=1.0, b=2.0, tx=5.0, c=3.0, d=4.0, ty=6.0)))
		lines = path.read_text().splitlines()
		assert lines[0] == "frame,a,b,c,d,tx,ty"
		assert lines[1] == "0,1.0,0.0,0.0,1.0,0.0,0.0"
		# Column order is (a, b, c, d, tx, ty) — linear part then translation.
		assert lines[2] == "1,1.0,2.0,3.0,4.0,5.0,6.0"

	def test_record_outside_context_manager_raises(self, tmp_path: Path) -> None:
		sink = CsvTransformSink(tmp_path / "x.csv")
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(FrameTransform(0, Transform2D.identity()))


class TestRecordingEgoMotionEstimator:
	def test_forwards_transform_and_records_it_with_frame_index(self) -> None:
		t0 = Transform2D.identity()
		t1 = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		inner = _StubEstimator([t0, t1])
		sink = _RecordingSink()
		recorder = RecordingEgoMotionEstimator(inner, sink)

		assert recorder.estimate(_frame(0)) is t0
		assert recorder.estimate(_frame(7)) is t1

		assert inner.calls == [0, 7]  # forwarded unchanged
		assert sink.records == [FrameTransform(0, t0), FrameTransform(7, t1)]
