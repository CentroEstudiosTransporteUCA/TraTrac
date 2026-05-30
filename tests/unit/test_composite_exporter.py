"""Tests for the composite trajectory exporter (fan-out over the export port)."""

from __future__ import annotations

from types import TracebackType

import numpy as np
import pytest

from tratrac.domain.frame import Frame
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.composite import CompositeTrajectoryExporter

_FRAME = Frame(index=0, pixels=np.zeros((1, 1, 3), dtype=np.uint8))


class _RecordingChild:
	"""Records lifecycle and emit calls into a shared event log."""

	def __init__(self, name: str, log: list[str], *, fail_on: str | None = None) -> None:
		self._name = name
		self._log = log
		self._fail_on = fail_on

	def _maybe_fail(self, event: str) -> None:
		if self._fail_on == event:
			raise RuntimeError(f"{self._name} failed on {event}")

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None:
		self._log.append(f"{self._name}:emit")

	def __enter__(self) -> _RecordingChild:
		self._log.append(f"{self._name}:enter")
		self._maybe_fail("enter")
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self._log.append(f"{self._name}:exit")
		self._maybe_fail("exit")


class TestConstruction:
	def test_rejects_empty(self) -> None:
		with pytest.raises(ValueError, match="at least one"):
			CompositeTrajectoryExporter([])


class TestFanOut:
	def test_emit_forwards_to_every_child_in_order(self) -> None:
		log: list[str] = []
		composite = CompositeTrajectoryExporter(
			[_RecordingChild("a", log), _RecordingChild("b", log)]
		)
		composite.emit_frame(1.0, [], _FRAME)
		assert log == ["a:emit", "b:emit"]

	def test_enter_then_exit_lifecycle_order(self) -> None:
		log: list[str] = []
		composite = CompositeTrajectoryExporter(
			[_RecordingChild("a", log), _RecordingChild("b", log)]
		)
		with composite:
			composite.emit_frame(0.0, [], _FRAME)
		# Children open in order, close in reverse order.
		assert log == ["a:enter", "b:enter", "a:emit", "b:emit", "b:exit", "a:exit"]


class TestErrorHandling:
	def test_enter_failure_unwinds_already_opened_children(self) -> None:
		log: list[str] = []
		composite = CompositeTrajectoryExporter(
			[_RecordingChild("a", log), _RecordingChild("b", log, fail_on="enter")]
		)
		with pytest.raises(RuntimeError, match="b failed on enter"):
			composite.__enter__()
		# a opened, b tried and failed, a was rolled back.
		assert log == ["a:enter", "b:enter", "a:exit"]

	def test_exit_closes_all_children_even_when_one_raises(self) -> None:
		log: list[str] = []
		composite = CompositeTrajectoryExporter(
			[_RecordingChild("a", log), _RecordingChild("b", log, fail_on="exit")]
		)
		composite.__enter__()
		with pytest.raises(RuntimeError, match="b failed on exit"):
			composite.__exit__(None, None, None)
		# b (reverse-first) raised, but a was still closed before the error propagated.
		assert log == ["a:enter", "b:enter", "b:exit", "a:exit"]
