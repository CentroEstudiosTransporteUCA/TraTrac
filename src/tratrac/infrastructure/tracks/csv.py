"""CSV track-observation sidecar: the "export B" raw-measurement file.

Pass 1 (the streaming pipeline) writes one row per tracked detection per frame —
the raw centroid + bbox + class the offline ``tratrac-smooth`` pass needs to run the
Kalman/RTS smoother (see vault/22_smoothing.md). A header line carries the video
metadata + metric scale so the post-pass is self-contained (no video needed) and can
reconstruct a metric ``.trj``. Centroids are in the tracker's coordinate frame —
stabilized pixels when ego-motion is on. Re-running the smoother on this file re-tunes
the filter with no re-detection.

Sibling of ``transform/csv.py``: a single self-contained row per record, written
immediately with no buffering.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

from tratrac.domain.detection import TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata

_HEADER = ("frame", "track_id", "cx", "cy", "w", "h", "vehicle_class", "score")


@dataclass(frozen=True, slots=True)
class TrackObservation:
	"""One tracked detection at one frame, as read back from the sidecar."""

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
	"""The full track-observation file: the run's metadata, scale, and observations."""

	metadata: VideoMetadata
	scale: float
	observations: list[TrackObservation]


class CsvTrackSink:
	"""Writes the track-observation sidecar. Use as a context manager."""

	def __init__(self, path: Path, metadata: VideoMetadata, *, scale: float) -> None:
		self._path = path
		self._metadata = metadata
		self._scale = scale
		self._file: TextIO | None = None

	def __enter__(self) -> CsvTrackSink:
		self._file = self._path.open("w", newline="")
		meta = self._metadata
		self._file.write(
			f"# fps={meta.fps} width={meta.width} height={meta.height} "
			f"total_frames={meta.total_frames} meters_per_pixel={self._scale}\n"
		)
		csv.writer(self._file).writerow(_HEADER)
		return self

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		writer = csv.writer(self._require_file())
		for item in tracked:
			box = item.detection.bbox
			center = box.center
			writer.writerow(
				[
					frame_index,
					item.track_id,
					center.x,
					center.y,
					box.width,
					box.height,
					item.detection.vehicle_class.value,
					item.detection.score,
				]
			)

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
			raise RuntimeError("CsvTrackSink must be used as a context manager.")
		return self._file


def read_tracks(path: Path) -> TrackRecording:
	"""Read a track-observation sidecar back into metadata + observations.

	Raises ``FileNotFoundError`` if absent and ``ValueError`` on a malformed header or
	row (re-wrapped with the file path).
	"""
	with path.open(newline="") as handle:
		metadata, scale = _parse_header(handle.readline(), path)
		try:
			observations = [_parse_row(row) for row in csv.DictReader(handle)]
		except (KeyError, TypeError, ValueError) as exc:
			raise ValueError(f"{path} is not a valid track file: {exc}") from exc
	return TrackRecording(metadata=metadata, scale=scale, observations=observations)


def _parse_header(line: str, path: Path) -> tuple[VideoMetadata, float]:
	if not line.startswith("#"):
		raise ValueError(f"{path} is missing its '# fps=... meters_per_pixel=...' header line.")
	fields: dict[str, str] = {}
	for token in line.lstrip("#").split():
		key, _, value = token.partition("=")
		fields[key] = value
	try:
		metadata = VideoMetadata(
			width=int(fields["width"]),
			height=int(fields["height"]),
			fps=float(fields["fps"]),
			total_frames=int(fields["total_frames"]),
		)
		return metadata, float(fields["meters_per_pixel"])
	except (KeyError, ValueError) as exc:
		raise ValueError(f"{path} has a malformed header line: {exc}") from exc


def _parse_row(row: dict[str, str]) -> TrackObservation:
	return TrackObservation(
		frame_index=int(row["frame"]),
		track_id=int(row["track_id"]),
		cx=float(row["cx"]),
		cy=float(row["cy"]),
		width=float(row["w"]),
		height=float(row["h"]),
		vehicle_class=VehicleClass(row["vehicle_class"]),
		score=float(row["score"]),
	)
