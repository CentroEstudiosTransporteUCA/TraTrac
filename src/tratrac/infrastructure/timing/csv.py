"""CSV timing sink: one wide row per frame (frame, detect, track, orient, export).

Buffers a frame's step timings and flushes the row when the frame ordinal
advances — safe because every step runs once per frame, so a frame's records
arrive consecutively. The last frame (and any partial frame from a mid-step
crash) flushes on close. See vault/15_step_timing.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import TextIO

from tratrac.domain.timing import PipelineStep, StepTiming

_STEP_ORDER = (
	PipelineStep.DETECT,
	PipelineStep.TRACK,
	PipelineStep.ORIENT,
	PipelineStep.EXPORT,
)


class CsvTimingSink:
	"""Writes per-frame step timings to a CSV file. Use as a context manager."""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._file: TextIO | None = None
		self._ordinal: int | None = None
		self._row: dict[PipelineStep, float] = {}

	def __enter__(self) -> CsvTimingSink:
		self._file = self._path.open("w", newline="")
		csv.writer(self._file).writerow(["frame", *(step.value for step in _STEP_ORDER)])
		return self

	def record(self, timing: StepTiming) -> None:
		if self._ordinal is not None and timing.frame_ordinal != self._ordinal:
			self._flush()
		self._ordinal = timing.frame_ordinal
		self._row[timing.step] = timing.seconds

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self._flush()
		if self._file is not None:
			self._file.close()
			self._file = None

	def _flush(self) -> None:
		if self._file is None or self._ordinal is None or not self._row:
			return
		row: list[object] = [self._ordinal, *(self._row.get(step, "") for step in _STEP_ORDER)]
		csv.writer(self._file).writerow(row)
		self._row = {}
