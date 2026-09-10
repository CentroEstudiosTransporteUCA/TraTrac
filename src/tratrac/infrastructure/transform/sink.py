"""JSONL transform sink: persists the per-frame ego-motion transform (current frame ->
global) so a downstream tool can invert each one to map stabilized coordinates back onto
the raw frame they were derived from. Renamed from ``csv.py`` — the wire format moved from
a bespoke CSV to the shared ``records.py`` JSON-Lines shape every transform sidecar now
uses. See src/tratrac/infrastructure/video/EGO_MOTION.md and
src/tratrac/infrastructure/transform/TRANSFORM_SINK.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TextIO

from tratrac.application.coordinate_transforms import PerFrameTransform
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.records import (
	SimilarityRecord,
	publish,
	read_jsonl,
	staging_path,
	to_json,
)


def read_transforms(path: Path) -> PerFrameTransform:
	"""Read a transform sidecar back into a ``PerFrameTransform``.

	The inverse of ``CoordinateTransformSink``: lets a reader (``tratrac-render``, the
	exclusion/calibration anchor-pose lookup) apply or reverse a frame's transform by
	exact ``frame_index``. Raises ``ValueError`` (path-wrapped) on a missing/malformed
	file or a record that isn't a similarity transform.
	"""
	records = read_jsonl(path)
	transforms: dict[int, Transform2D] = {}
	for record in records:
		if not isinstance(record, SimilarityRecord):
			raise ValueError(
				f"{path}: expected similarity-transform records, found {type(record).__name__}."
			)
		transforms[record.frame_index] = record.transform
	return PerFrameTransform(transforms)


class PrecomputedEgoMotionEstimator:
	"""``EgoMotionEstimator`` reading an already-known transform table instead of
	estimating live — the `tratrac` side of `ego_motion.transforms_in`: a
	`tratrac-stabilize` run's output, consumed so the live run's detector pass never
	needs to also run ORB (see "Detector-free ego-motion",
	src/tratrac/infrastructure/video/EGO_MOTION.md).

	``estimate`` is an exact `PerFrameTransform` lookup by `frame.index` — no
	feature matching. Raises ``KeyError`` (with a clearer message) if the table
	doesn't cover a frame the run processes; the table is expected to be complete
	for the whole clip.
	"""

	def __init__(self, transforms: PerFrameTransform) -> None:
		self._transforms = transforms

	def estimate(self, frame: Frame) -> Transform2D:
		found = self._transforms.at(frame.index)
		if found is None:
			raise KeyError(
				f"no pre-built transform for frame {frame.index}; ego_motion.transforms_in "
				"must cover every frame this run processes."
			)
		return found


class CoordinateTransformSink:
	"""Writes per-frame ego-motion transforms to a JSONL sidecar. Use as a context manager.

	Streams to a staging path with no buffering (each record is a self-contained
	line, written immediately), then atomically publishes to the canonical path only
	on a clean exit (``infrastructure/transform/records.py``'s ``staging_path``/
	``publish``) — a reader of the canonical path never observes a half-written
	file: either it isn't there yet (run still going, or it crashed — the
	``.partial`` file stays inspectable) or it's the complete table.
	"""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._staging = staging_path(path)
		self._file: TextIO | None = None

	def __enter__(self) -> CoordinateTransformSink:
		self._file = self._staging.open("w")
		return self

	def record(self, frame_transform: FrameTransform) -> None:
		file = self._require_file()
		record = SimilarityRecord(frame_transform.frame_index, frame_transform.transform)
		file.write(json.dumps(to_json(record)))
		file.write("\n")

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._file is not None:
			self._file.close()
			self._file = None
		if exc_type is None:
			publish(self._staging, self._path)

	def _require_file(self) -> TextIO:
		if self._file is None:
			raise RuntimeError("CoordinateTransformSink must be used as a context manager.")
		return self._file
