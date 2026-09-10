"""Smoothed track record: the de-jittered trajectory, always in image-space pixels.

``tratrac-postprocess --smoothed-record`` (``src/tratrac/application/SMOOTHING.md``'s
"Dual-space export" section) writes this alongside, or instead of, the SSAM ``.trj`` — both
come from the exact same single Kalman/RTS smoothing pass, just expressed in different
coordinate spaces: ``.trj`` is world-space metric when ``--calibration`` projects to it (MVP2's
invariant, ``application/WORLD_PROJECTION.md``), this record is always raw image pixels,
suited to overlaying on the actual video frame (e.g. ``tratrac-fiftyone``) regardless of
whether the run was calibrated.

Deliberately **not** a full ``VehicleState`` mirror: velocity/acceleration aren't included.
There's no general, honest way to convert a world-space smoothed velocity into an image-space
one without differentiating the (possibly per-frame) inverse homography, and nothing consumes
it yet — see ``application/track_smoothing.py``'s ``invert_state_to_image``, which only
reconstructs what a homography inverse actually recovers cleanly: position, heading,
dimensions, all by inverting geometric points, not by converting a rate.

Mirrors ``infrastructure/tracks/footprint_parquet.py``'s writer/reader shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tratrac.domain.frame import VideoMetadata

_ROW_GROUP_ROWS = 10_000

_SCHEMA = pa.schema(
	[
		("frame", pa.int64()),
		("track_id", pa.int64()),
		("cx", pa.float64()),
		("cy", pa.float64()),
		("angle", pa.float64()),
		("length", pa.float64()),
		("width", pa.float64()),
		("link_id", pa.int64()),
		("lane_id", pa.int64()),
	]
)

# One buffered row, column order matching ``_SCHEMA``.
_Row = tuple[int, int, float, float, float, float, float, int, int]


@dataclass(frozen=True, slots=True)
class SmoothedObservation:
	"""One vehicle's de-jittered pose at one frame, always in raw image-space pixels.

	``angle`` is the heading in radians (``Heading.from_angle`` convention, matching
	``Detection.angle``) — a direction, not a rate, so it survives coordinate-space inversion
	cleanly, unlike velocity.
	"""

	frame_index: int
	track_id: int
	cx: float
	cy: float
	angle: float
	length: float
	width: float
	link_id: int = 0
	lane_id: int = 0


@dataclass(frozen=True, slots=True)
class SmoothedRecording:
	"""The full smoothed record: the run's metadata and its image-space observations."""

	metadata: VideoMetadata
	observations: list[SmoothedObservation]


class SmoothedTrackParquetSink:
	"""Writes the smoothed record as Parquet. Use as a context manager."""

	def __init__(self, path: Path, metadata: VideoMetadata) -> None:
		self._path = path
		self._metadata = metadata
		self._writer: Any = None
		self._schema: Any = None
		self._buffer: list[_Row] = []

	def __enter__(self) -> SmoothedTrackParquetSink:
		meta = self._metadata
		self._schema = _SCHEMA.with_metadata(
			{
				b"fps": str(meta.fps).encode(),
				b"width": str(meta.width).encode(),
				b"height": str(meta.height).encode(),
				b"total_frames": str(meta.total_frames).encode(),
			}
		)
		self._writer = pq.ParquetWriter(self._path, self._schema)
		self._buffer = []
		return self

	def record(self, observation: SmoothedObservation) -> None:
		self._require_writer()
		self._buffer.append(
			(
				observation.frame_index,
				observation.track_id,
				observation.cx,
				observation.cy,
				observation.angle,
				observation.length,
				observation.width,
				observation.link_id,
				observation.lane_id,
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
		names = [f.name for f in _SCHEMA]
		table = pa.table(dict(zip(names, columns, strict=True)), schema=self._schema)
		self._writer.write_table(table)
		self._buffer = []

	def _require_writer(self) -> None:
		if self._writer is None:
			raise RuntimeError("SmoothedTrackParquetSink must be used as a context manager.")


def read_smoothed_tracks(path: Path) -> SmoothedRecording:
	"""Read a smoothed record back. Raises ``ValueError`` (re-wrapped) on a malformed file."""
	try:
		table = pq.read_table(path)
	except (OSError, ValueError) as exc:
		raise ValueError(f"{path} is not a readable Parquet smoothed record: {exc}") from exc
	meta = table.schema.metadata or {}
	try:
		metadata = VideoMetadata(
			width=int(meta[b"width"]),
			height=int(meta[b"height"]),
			fps=float(meta[b"fps"]),
			total_frames=int(meta[b"total_frames"]),
		)
	except (KeyError, ValueError) as exc:
		raise ValueError(f"{path} is missing required run metadata: {exc}") from exc
	columns = table.to_pydict()
	try:
		observations = [
			SmoothedObservation(
				frame_index=int(columns["frame"][i]),
				track_id=int(columns["track_id"][i]),
				cx=float(columns["cx"][i]),
				cy=float(columns["cy"][i]),
				angle=float(columns["angle"][i]),
				length=float(columns["length"][i]),
				width=float(columns["width"][i]),
				link_id=int(columns["link_id"][i]),
				lane_id=int(columns["lane_id"][i]),
			)
			for i in range(table.num_rows)
		]
	except (KeyError, ValueError) as exc:
		raise ValueError(f"{path} has a malformed row: {exc}") from exc
	return SmoothedRecording(metadata=metadata, observations=observations)
