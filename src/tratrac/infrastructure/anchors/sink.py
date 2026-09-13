"""AnchorImageSink: writes each keyframe anchor's frame as a PNG.

The ``AnchorSink`` adapter the run uses to export the frames an operator draws
exclusion zones on (see src/tratrac/application/EXCLUSION_ZONES.md). cv2 lives
behind an injected ``image_writer`` seam so the orchestration (filenames,
lifecycle) is testable without a codec.

No manifest is written here: an anchor's pose is already in the per-frame
transform sidecar (``infrastructure/transform/sink.py``) under the same
``frame.index``, so a separate ``(frame_index, pose, image)`` record would only
duplicate it. The PNG's filename (``frame_<i>.png``) is self-describing — the
only thing a downstream reader needs beyond that is the transform sidecar.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.frame import Frame

ImageWriter = Callable[[Path, NDArray[np.uint8]], None]


def _cv2_write(path: Path, pixels: NDArray[np.uint8]) -> None:
	import cv2  # lazy: keep the module import-light

	cv2.imwrite(str(path), pixels)


class AnchorImageSink:
	"""Writes ``frame_<i>.png`` per anchor. Use as a context manager."""

	def __init__(
		self,
		out_dir: Path,
		*,
		image_writer: ImageWriter = _cv2_write,
	) -> None:
		self._out_dir = out_dir
		self._image_writer = image_writer

	def __enter__(self) -> AnchorImageSink:
		self._out_dir.mkdir(parents=True, exist_ok=True)
		return self

	def record(self, frame: Frame) -> None:
		image_name = f"frame_{frame.index}.png"
		self._image_writer(self._out_dir / image_name, frame.pixels)

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		del exc_type, exc_val, exc_tb
