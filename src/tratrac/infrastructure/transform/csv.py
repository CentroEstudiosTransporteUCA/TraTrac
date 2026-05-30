"""CSV transform sink: one row per frame (frame, a, b, c, d, tx, ty).

Persists the per-frame ego-motion transform (current frame -> global) so a
downstream tool can invert each row to map stabilized coordinates back onto the
raw frame they were derived from. Unlike the wide-row timing CSV, each record is
a single self-contained row, so it is written immediately with no buffering. See
vault/05_75_mvp1_9.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import TextIO

from tratrac.domain.stabilization import FrameTransform

# Linear part (a, b, c, d) then translation (tx, ty). Consumers read by name, so
# the column order is free; this groups the 2x2 ahead of the offset.
_HEADER = ("frame", "a", "b", "c", "d", "tx", "ty")


class CsvTransformSink:
	"""Writes per-frame ego-motion transforms to a CSV file. Use as a context manager."""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._file: TextIO | None = None

	def __enter__(self) -> CsvTransformSink:
		self._file = self._path.open("w", newline="")
		csv.writer(self._file).writerow(_HEADER)
		return self

	def record(self, frame_transform: FrameTransform) -> None:
		file = self._require_file()
		t = frame_transform.transform
		csv.writer(file).writerow([frame_transform.frame_index, t.a, t.b, t.c, t.d, t.tx, t.ty])

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._file is not None:
			self._file.close()
			self._file = None

	def _require_file(self) -> TextIO:
		if self._file is None:
			raise RuntimeError("CsvTransformSink must be used as a context manager.")
		return self._file
