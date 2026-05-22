"""OpenCV-backed VideoSource adapter. MVP1 reads frames sequentially on the main thread."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2

from tratrac.domain.frame import Frame, VideoMetadata


class OpenCvVideoSource:
	"""Reads frames from a video container via ``cv2.VideoCapture``."""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._capture: Any = None
		self._metadata: VideoMetadata | None = None

	@property
	def metadata(self) -> VideoMetadata:
		if self._metadata is None:
			raise RuntimeError("VideoSource must be used as a context manager.")
		return self._metadata

	def frames(self) -> Iterator[Frame]:
		capture = self._require_capture()
		index = 0
		while True:
			ok, pixels = capture.read()
			if not ok:
				return
			yield Frame(index=index, pixels=pixels)
			index += 1

	def __enter__(self) -> OpenCvVideoSource:
		capture = cv2.VideoCapture(str(self._path))
		if not capture.isOpened():
			raise RuntimeError(f"Cannot open video: {self._path}")
		self._capture = capture
		self._metadata = VideoMetadata(
			width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
			height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
			fps=float(capture.get(cv2.CAP_PROP_FPS)),
			total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
		)
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._capture is not None:
			self._capture.release()
			self._capture = None
		self._metadata = None

	def _require_capture(self) -> Any:
		if self._capture is None:
			raise RuntimeError("VideoSource must be used as a context manager.")
		return self._capture
