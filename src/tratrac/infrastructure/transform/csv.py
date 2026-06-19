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

from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform

# Linear part (a, b, c, d) then translation (tx, ty). Consumers read by name, so
# the column order is free; this groups the 2x2 ahead of the offset.
_HEADER = ("frame", "a", "b", "c", "d", "tx", "ty")


def read_transforms(path: Path) -> dict[int, Transform2D]:
	"""Read a transform CSV back into a ``{frame_index: Transform2D}`` map.

	The inverse of ``CsvTransformSink``: lets a replay estimator and the exclusion
	mask reuse the schedule a scout pass recorded. Raises ``FileNotFoundError`` if
	absent and ``ValueError`` on a malformed row (re-wrapped with the file path).
	"""
	transforms: dict[int, Transform2D] = {}
	with path.open(newline="") as handle:
		for row in csv.DictReader(handle):
			try:
				transforms[int(row["frame"])] = Transform2D(
					a=float(row["a"]),
					b=float(row["b"]),
					tx=float(row["tx"]),
					c=float(row["c"]),
					d=float(row["d"]),
					ty=float(row["ty"]),
				)
			except (KeyError, TypeError, ValueError) as exc:
				raise ValueError(f"{path} is not a valid transform CSV: {exc}") from exc
	return transforms


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
