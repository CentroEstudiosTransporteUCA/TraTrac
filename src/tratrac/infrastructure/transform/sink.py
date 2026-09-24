"""JSONL transform sink + readers for the unified transforms file.

Every geometric transform (ego-motion, scale, homography) is a row in one
file (`infrastructure/transform/records.py`). This module is the I/O seam:
`CoordinateTransformSink` streams rows out during `tratrac-preprocess`'s live
passes; `read_transform_table`/`read_ego_motion` read them back into a
`TransformTable` or a plain per-frame `Transform2D` lookup, filtered to
whichever kind(s) a caller needs. See
`infrastructure/video/ego_motion_orb.py`'s module docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TextIO

from tratrac.application.coordinate_transforms import TransformTable
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.stabilization import FrameTransform
from tratrac.infrastructure.transform.records import (
	SimilarityFunction,
	TransformFunction,
	TransformRow,
	encode_row,
	publish,
	read_jsonl,
	staging_path,
)

# Exported so a reader that needs to recover (width, height) from an already-padded
# whole-canvas zone (e.g. `cli_preprocess.py`'s `_infer_canvas`) can subtract it back out.
CANVAS_PAD = 1.0


def whole_canvas(width: int, height: int) -> tuple[Point2D, ...]:
	"""The full-frame rectangle -- the concrete ``zone`` a row uses when it applies
	everywhere spatially (never a null/optional zone, per the row model).

	Padded by ``CANVAS_PAD`` on every side: ``point_in_polygon`` (``domain/geometry.py``)
	treats a polygon's own boundary as *outside* it (shapely's ``contains``, not
	``covers``), so a zone flush with the frame edges would reject a perfectly
	normal detection or calibration correspondence sitting exactly at pixel
	``(0, 0)`` or the frame's far edge. The pad only has to clear that boundary
	case, not model anything physical.
	"""
	return (
		Point2D(-CANVAS_PAD, -CANVAS_PAD),
		Point2D(float(width) + CANVAS_PAD, -CANVAS_PAD),
		Point2D(float(width) + CANVAS_PAD, float(height) + CANVAS_PAD),
		Point2D(-CANVAS_PAD, float(height) + CANVAS_PAD),
	)


def read_transform_table(
	path: Path, *, kinds: tuple[type[TransformFunction], ...]
) -> TransformTable:
	"""Read ``path``'s rows, keep only functions matching ``kinds``, build a ``TransformTable``.

	The two calling shapes: the ego-motion stage (``kinds=(SimilarityFunction,)``,
	used by ``tratrac``'s pipeline and zone/correspondence pose resolution) and the
	projection stage (``kinds=(ScaleFunction, HomographyFunction)``, used by
	``tratrac-postprocess`` to map already-global observations to their final unit).
	"""
	rows = read_jsonl(path)
	return TransformTable(row for row in rows if isinstance(row.function, kinds))


def read_ego_motion(path: Path) -> TransformTable:
	"""The ego-motion-only stage: raw-frame pixel -> global-frame pixel."""
	return read_transform_table(path, kinds=(SimilarityFunction,))


class PrecomputedEgoMotionEstimator:
	"""``EgoMotionEstimator`` reading an already-known ego-motion table instead of
	estimating live -- the `tratrac` side of `ego_motion.transforms_in`: a
	`tratrac-preprocess` run's output, consumed so `tratrac` never estimates
	ego-motion itself (see "Detector-free ego-motion",
	infrastructure/video/ego_motion_orb.py's module docstring).

	``estimate`` is an exact per-frame lookup, no feature matching. An entirely
	empty ego-motion stage (a static-camera run) returns the identity for every
	frame. Raises ``ValueError`` if a frame's row is present but isn't an
	ego-motion function (a malformed or mismatched file), or ``KeyError`` if the
	table is non-empty but doesn't cover a frame the run processes.
	"""

	def __init__(self, ego_motion: TransformTable) -> None:
		self._ego_motion = ego_motion

	def estimate(self, frame: Frame) -> Transform2D:
		if self._ego_motion.is_empty:
			return Transform2D.identity()
		function = self._ego_motion.function_at(frame.index)
		if function is None:
			raise KeyError(
				f"no ego-motion transform recorded for frame {frame.index}; "
				"input.transforms_in must cover every frame this run processes."
			)
		if not isinstance(function, SimilarityFunction):
			raise ValueError(
				f"frame {frame.index}'s ego-motion row is a {type(function).__name__}, "
				"not a similarity transform."
			)
		return function.transform


class CoordinateTransformSink:
	"""Writes rows to a JSONL sidecar. Use as a context manager.

	Streams to a staging path with no buffering (each row is a self-contained
	line, written immediately), then atomically publishes to the canonical path
	only on a clean exit (``infrastructure/transform/records.py``'s
	``staging_path``/``publish``) — a reader of the canonical path never
	observes a half-written file: either it isn't there yet (run still going,
	or it crashed — the ``.partial`` file stays inspectable) or it's the
	complete table.

	``record`` (the ``TransformSink`` port `RecordingEgoMotionEstimator` drives
	during the ORB walk) writes one ego-motion row per frame, using ``width``/
	``height`` (fixed for the whole run) as that row's whole-canvas zone.
	``record_row`` is the lower-level escape hatch `tratrac-preprocess estimate`
	uses to also write the per-frame scale rows in the same pass -- scale isn't
	a ``FrameTransform``, so it doesn't fit the narrower ``TransformSink`` port.
	"""

	def __init__(self, path: Path, *, width: int, height: int) -> None:
		self._path = path
		self._staging = staging_path(path)
		self._zone = whole_canvas(width, height)
		self._file: TextIO | None = None

	def __enter__(self) -> CoordinateTransformSink:
		self._file = self._staging.open("w")
		return self

	def record(self, frame_transform: FrameTransform) -> None:
		self.record_row(
			TransformRow(
				frame_index=frame_transform.frame_index,
				zone=self._zone,
				function=SimilarityFunction(frame_transform.transform),
			)
		)

	def record_row(self, row: TransformRow) -> None:
		file = self._require_file()
		file.write(json.dumps(encode_row(row)))
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


__all__ = [
	"CANVAS_PAD",
	"CoordinateTransformSink",
	"PrecomputedEgoMotionEstimator",
	"read_ego_motion",
	"read_transform_table",
	"whole_canvas",
]
