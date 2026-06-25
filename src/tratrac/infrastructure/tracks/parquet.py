"""Parquet track record: the canonical "export B" raw-measurement file.

Pass 1 (the perception run) writes one row per tracked detection per frame — the raw
centroid + bbox + class the offline ``tratrac-smooth`` pass needs to run the Kalman/RTS
smoother (see vault/22_smoothing.md). The video metadata + metric scale live in the
Parquet **schema metadata** so the post-pass is self-contained (no video needed) and can
reconstruct a metric ``.trj``. Centroids are in the tracker's coordinate frame —
stabilized pixels when ego-motion is on.

Columnar storage is the canonical record (Step 2 of the export inversion); it replaced a
line-oriented CSV behind the same ``TrackSink`` / ``read_tracks`` seam. ``pyarrow`` is the
only Parquet dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tratrac.domain.detection import TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata

# Buffered rows are flushed as one Parquet row group, so the writer stays streaming
# without emitting a tiny row group per frame.
_ROW_GROUP_ROWS = 10_000

_COLUMNS = ("frame", "track_id", "cx", "cy", "w", "h", "vehicle_class", "score")
_SCHEMA = pa.schema(
	[
		("frame", pa.int64()),
		("track_id", pa.int64()),
		("cx", pa.float64()),
		("cy", pa.float64()),
		("w", pa.float64()),
		("h", pa.float64()),
		("vehicle_class", pa.string()),
		("score", pa.float64()),
	]
)

# One buffered observation row, column order matching ``_SCHEMA``.
_Row = tuple[int, int, float, float, float, float, str, float]


@dataclass(frozen=True, slots=True)
class TrackObservation:
	"""One tracked detection at one frame, as read back from the record."""

	frame_index: int
	track_id: int
	cx: float
	cy: float
	width: float
	height: float
	vehicle_class: VehicleClass
	score: float


@dataclass(frozen=True, slots=True)
class TrackRecording:
	"""The full track record: the run's metadata, scale, and observations."""

	metadata: VideoMetadata
	scale: float
	observations: list[TrackObservation]


class ParquetTrackSink:
	"""Writes the track record as Parquet. Use as a context manager."""

	def __init__(self, path: Path, metadata: VideoMetadata, *, scale: float) -> None:
		self._path = path
		self._metadata = metadata
		self._scale = scale
		# pyarrow is treated as untyped at the third-party seam (mypy follow_imports=skip),
		# so the writer is Any; None until the context manager is entered.
		self._writer: Any = None
		self._schema: Any = None
		self._buffer: list[_Row] = []

	def __enter__(self) -> ParquetTrackSink:
		meta = self._metadata
		self._schema = _SCHEMA.with_metadata(
			{
				b"fps": str(meta.fps).encode(),
				b"width": str(meta.width).encode(),
				b"height": str(meta.height).encode(),
				b"total_frames": str(meta.total_frames).encode(),
				b"meters_per_pixel": str(self._scale).encode(),
			}
		)
		self._writer = pq.ParquetWriter(self._path, self._schema)
		self._buffer = []
		return self

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		self._require_writer()
		for item in tracked:
			box = item.detection.bbox
			center = box.center
			self._buffer.append(
				(
					frame_index,
					item.track_id,
					center.x,
					center.y,
					box.width,
					box.height,
					item.detection.vehicle_class.value,
					item.detection.score,
				)
			)
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
		columns = list(zip(*self._buffer, strict=True))
		table = pa.table(dict(zip(_COLUMNS, columns, strict=True)), schema=self._schema)
		self._writer.write_table(table)
		self._buffer = []

	def _require_writer(self) -> None:
		if self._writer is None:
			raise RuntimeError("ParquetTrackSink must be used as a context manager.")


def read_tracks(path: Path) -> TrackRecording:
	"""Read a Parquet track record back into metadata + observations.

	Raises ``ValueError`` (re-wrapped with the path) on a missing/malformed file or a
	record without the expected schema metadata.
	"""
	try:
		table = pq.read_table(path)
	except (OSError, ValueError) as exc:
		raise ValueError(f"{path} is not a readable Parquet track record: {exc}") from exc
	metadata, scale = _parse_schema_metadata(table.schema.metadata, path)
	columns = table.to_pydict()
	observations = [
		TrackObservation(
			frame_index=int(columns["frame"][i]),
			track_id=int(columns["track_id"][i]),
			cx=float(columns["cx"][i]),
			cy=float(columns["cy"][i]),
			width=float(columns["w"][i]),
			height=float(columns["h"][i]),
			vehicle_class=VehicleClass(columns["vehicle_class"][i]),
			score=float(columns["score"][i]),
		)
		for i in range(table.num_rows)
	]
	return TrackRecording(metadata=metadata, scale=scale, observations=observations)


def _parse_schema_metadata(
	raw: dict[bytes, bytes] | None, path: Path
) -> tuple[VideoMetadata, float]:
	if not raw or b"fps" not in raw:
		raise ValueError(f"{path} is missing its track-record schema metadata (fps, scale, ...).")
	try:
		metadata = VideoMetadata(
			width=int(raw[b"width"]),
			height=int(raw[b"height"]),
			fps=float(raw[b"fps"]),
			total_frames=int(raw[b"total_frames"]),
		)
		return metadata, float(raw[b"meters_per_pixel"])
	except (KeyError, ValueError) as exc:
		raise ValueError(f"{path} has malformed track-record metadata: {exc}") from exc
