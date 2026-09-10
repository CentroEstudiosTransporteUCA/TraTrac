"""Tests for the JSONL per-frame transform sink + the recording decorator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.transform.sink import CoordinateTransformSink, read_transforms


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


class TestCoordinateTransformSink:
	def test_writes_one_jsonl_record_per_frame_and_publishes_on_clean_exit(
		self, tmp_path: Path
	) -> None:
		path = tmp_path / "transforms.jsonl"
		with CoordinateTransformSink(path) as sink:
			sink.record(FrameTransform(0, Transform2D.identity()))
			sink.record(FrameTransform(1, Transform2D(a=1.0, b=2.0, tx=5.0, c=3.0, d=4.0, ty=6.0)))

		assert path.exists()
		assert not path.with_name(path.name + ".partial").exists()
		lines = [json.loads(line) for line in path.read_text().splitlines()]
		assert lines[0] == {
			"type": "similarity",
			"frame_index": 0,
			"a": 1.0,
			"b": 0.0,
			"c": 0.0,
			"d": 1.0,
			"tx": 0.0,
			"ty": 0.0,
		}
		assert lines[1]["frame_index"] == 1
		assert lines[1]["tx"] == 5.0

	def test_staging_file_is_left_behind_on_a_failed_run(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.jsonl"

		def _write_then_fail() -> None:
			with CoordinateTransformSink(path) as sink:
				sink.record(FrameTransform(0, Transform2D.identity()))
				raise RuntimeError("boom")

		with pytest.raises(RuntimeError):
			_write_then_fail()

		assert not path.exists()
		assert path.with_name(path.name + ".partial").exists()

	def test_record_outside_context_manager_raises(self, tmp_path: Path) -> None:
		sink = CoordinateTransformSink(tmp_path / "x.jsonl")
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(FrameTransform(0, Transform2D.identity()))


class TestReadTransforms:
	def test_round_trips_through_a_per_frame_transform(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.jsonl"
		t1 = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		with CoordinateTransformSink(path) as sink:
			sink.record(FrameTransform(0, Transform2D.identity()))
			sink.record(FrameTransform(7, t1))

		table = read_transforms(path)
		assert table.at(0) == Transform2D.identity()
		assert table.at(7) == t1
		assert table.at(99) is None


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
