"""Footprint sidecar: per-observation occupancy polygon, from post-hoc segmentation.

Group D1 (GitHub Issues, MVP4 remainder): a segmentation stage (SAM 3, not
built yet — needs a GPU, see ``src/tratrac/application/FOOTPRINT.md``) would write one polygon
per track observation here, and ``tratrac-postprocess --footprint`` reads it back to replace
bbox/OBB-derived ``Dimensions`` with mask-derived ones before smoothing — the real accuracy
payoff segmentation buys over even an oriented box.

This *storage format* does not depend on SAM 3's specific output shape — "a polygon per
``(track_id, frame_index)``" is a general geometric concept, independent of which model
produced it, the same way ``infrastructure/reid/json.py``'s schema doesn't depend on DINOv3
specifically. That's what let this land without the segmentation stage that would populate it.

Mirrors ``infrastructure/tracks/parquet.py``'s writer/reader shape (buffered row-group writes,
one ``TrackXSink``-style context manager, a plain read function) rather than inventing a new
pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tratrac.domain.geometry import Point2D, Polygon

_ROW_GROUP_ROWS = 10_000

_SCHEMA = pa.schema(
	[
		("frame", pa.int64()),
		("track_id", pa.int64()),
		("xs", pa.list_(pa.float64())),
		("ys", pa.list_(pa.float64())),
	]
)

# One buffered footprint row, column order matching ``_SCHEMA``.
_Row = tuple[int, int, list[float], list[float]]


@dataclass(frozen=True, slots=True)
class FootprintObservation:
	"""One track's occupancy polygon at one frame, in the same coordinate frame (pixels,
	stabilized when ego-motion is on) as the matching ``TrackObservation``."""

	frame_index: int
	track_id: int
	polygon: Polygon


class FootprintParquetSink:
	"""Writes the footprint sidecar as Parquet. Use as a context manager."""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._writer: Any = None
		self._buffer: list[_Row] = []

	def __enter__(self) -> FootprintParquetSink:
		self._writer = pq.ParquetWriter(self._path, _SCHEMA)
		self._buffer = []
		return self

	def record(self, frame_index: int, track_id: int, polygon: Polygon) -> None:
		self._require_writer()
		xs = [v.x for v in polygon.vertices]
		ys = [v.y for v in polygon.vertices]
		self._buffer.append((frame_index, track_id, xs, ys))
		if len(self._buffer) >= _ROW_GROUP_ROWS:
			self._flush()

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._writer is not None:
			self._flush()
			self._writer.close()
			self._writer = None

	def _flush(self) -> None:
		if not self._buffer:
			return
		frames, track_ids, xs, ys = zip(*self._buffer, strict=True)
		table = pa.table(
			{"frame": frames, "track_id": track_ids, "xs": xs, "ys": ys}, schema=_SCHEMA
		)
		self._writer.write_table(table)
		self._buffer = []

	def _require_writer(self) -> None:
		if self._writer is None:
			raise RuntimeError("FootprintParquetSink must be used as a context manager.")


def read_footprints(path: Path) -> list[FootprintObservation]:
	"""Read a footprint sidecar back into a flat list of observations.

	Raises ``ValueError`` (re-wrapped with the path) on a missing/malformed file. Each row's
	``xs``/``ys`` must have equal, non-zero length and at least 3 points (a ``Polygon``
	requirement) — a malformed row raises rather than silently dropping the observation.
	"""
	try:
		table = pq.read_table(path)
	except (OSError, ValueError) as exc:
		raise ValueError(f"{path} is not a readable Parquet footprint sidecar: {exc}") from exc
	columns = table.to_pydict()
	try:
		observations = [
			FootprintObservation(
				frame_index=int(columns["frame"][i]),
				track_id=int(columns["track_id"][i]),
				polygon=Polygon(
					vertices=tuple(
						Point2D(float(x), float(y))
						for x, y in zip(columns["xs"][i], columns["ys"][i], strict=True)
					)
				),
			)
			for i in range(table.num_rows)
		]
	except (KeyError, ValueError) as exc:
		raise ValueError(f"{path} has a malformed footprint row: {exc}") from exc
	return observations
