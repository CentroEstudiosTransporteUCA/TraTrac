"""Tests for the JSONL transform sink + readers + the recording decorator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tratrac.application.coordinate_transforms import TransformTable
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.transform.records import SimilarityFunction, TransformRow
from tratrac.infrastructure.transform.sink import (
	CoordinateTransformSink,
	PrecomputedEgoMotionEstimator,
	read_ego_motion,
	whole_canvas,
)


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
		with CoordinateTransformSink(path, width=4, height=4) as sink:
			sink.record(FrameTransform(0, Transform2D.identity()))
			sink.record(FrameTransform(1, Transform2D(a=1.0, b=2.0, tx=5.0, c=3.0, d=4.0, ty=6.0)))

		assert path.exists()
		assert not path.with_name(path.name + ".partial").exists()
		lines = [json.loads(line) for line in path.read_text().splitlines()]
		assert lines[0] == {
			"frame_index": 0,
			"type": "ego_motion",
			"zone": [[p.x, p.y] for p in whole_canvas(4, 4)],
			"function": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0},
		}
		assert lines[1]["frame_index"] == 1
		assert lines[1]["function"]["tx"] == 5.0

	def test_staging_file_is_left_behind_on_a_failed_run(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.jsonl"

		def _write_then_fail() -> None:
			with CoordinateTransformSink(path, width=4, height=4) as sink:
				sink.record(FrameTransform(0, Transform2D.identity()))
				raise RuntimeError("boom")

		with pytest.raises(RuntimeError):
			_write_then_fail()

		assert not path.exists()
		assert path.with_name(path.name + ".partial").exists()

	def test_record_outside_context_manager_raises(self, tmp_path: Path) -> None:
		sink = CoordinateTransformSink(tmp_path / "x.jsonl", width=4, height=4)
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(FrameTransform(0, Transform2D.identity()))

	def test_record_row_writes_an_arbitrary_row(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.jsonl"
		zone = whole_canvas(4, 4)
		with CoordinateTransformSink(path, width=4, height=4) as sink:
			sink.record_row(TransformRow(0, zone, SimilarityFunction(Transform2D.identity())))

		table = read_ego_motion(path)
		assert table.function_at(0) == SimilarityFunction(Transform2D.identity())


class TestReadEgoMotion:
	def test_round_trips_through_a_transform_table(self, tmp_path: Path) -> None:
		path = tmp_path / "transforms.jsonl"
		t1 = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		with CoordinateTransformSink(path, width=4, height=4) as sink:
			sink.record(FrameTransform(0, Transform2D.identity()))
			sink.record(FrameTransform(7, t1))

		table = read_ego_motion(path)
		assert table.function_at(0) == SimilarityFunction(Transform2D.identity())
		assert table.function_at(7) == SimilarityFunction(t1)
		assert table.function_at(99) is None


class TestPrecomputedEgoMotionEstimator:
	def test_estimate_looks_up_the_exact_frame(self) -> None:
		t1 = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		zone = whole_canvas(2, 2)
		table = TransformTable(
			[
				TransformRow(0, zone, SimilarityFunction(Transform2D.identity())),
				TransformRow(7, zone, SimilarityFunction(t1)),
			]
		)
		estimator = PrecomputedEgoMotionEstimator(table)
		assert estimator.estimate(Frame(index=0, pixels=np.zeros((2, 2, 3), dtype=np.uint8))) == (
			Transform2D.identity()
		)
		assert estimator.estimate(Frame(index=7, pixels=np.zeros((2, 2, 3), dtype=np.uint8))) == t1

	def test_missing_frame_raises(self) -> None:
		zone = whole_canvas(2, 2)
		table = TransformTable([TransformRow(0, zone, SimilarityFunction(Transform2D.identity()))])
		estimator = PrecomputedEgoMotionEstimator(table)
		with pytest.raises(KeyError, match="frame 5"):
			estimator.estimate(Frame(index=5, pixels=np.zeros((2, 2, 3), dtype=np.uint8)))

	def test_empty_table_returns_identity(self) -> None:
		estimator = PrecomputedEgoMotionEstimator(TransformTable([]))
		result = estimator.estimate(Frame(index=5, pixels=np.zeros((2, 2, 3), dtype=np.uint8)))
		assert result == Transform2D.identity()


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
