"""AnchorManifestSink: writes each keyframe anchor as a PNG + a manifest on close.

The ``AnchorSink`` adapter the run uses to export the frames an operator draws exclusion
zones on (see vault/21_exclusion_zones.md). cv2 lives behind an injected ``image_writer``
seam so the orchestration (filenames, manifest accumulation, lifecycle) is testable without
a codec.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.infrastructure.anchors.manifest import ReferenceFrame, write_manifest

ImageWriter = Callable[[Path, NDArray[np.uint8]], None]


def _cv2_write(path: Path, pixels: NDArray[np.uint8]) -> None:
	import cv2  # lazy: keep the module import-light

	cv2.imwrite(str(path), pixels)


class AnchorManifestSink:
	"""Writes ``frame_<i>.png`` per anchor and ``manifest.json`` on exit. Use as a context
	manager."""

	def __init__(
		self,
		out_dir: Path,
		*,
		video_label: str,
		manifest_name: str = "manifest.json",
		image_writer: ImageWriter = _cv2_write,
	) -> None:
		self._out_dir = out_dir
		self._video_label = video_label
		self._manifest_name = manifest_name
		self._image_writer = image_writer
		self._references: list[ReferenceFrame] = []

	def __enter__(self) -> AnchorManifestSink:
		self._out_dir.mkdir(parents=True, exist_ok=True)
		self._references = []
		return self

	def record(self, frame: Frame, pose: Transform2D) -> None:
		image_name = f"frame_{frame.index}.png"
		self._image_writer(self._out_dir / image_name, frame.pixels)
		self._references.append(ReferenceFrame(frame.index, pose, image_name))

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		write_manifest(
			self._out_dir / self._manifest_name, self._references, video=self._video_label
		)
