"""OpenCV-backed VideoSource adapter. MVP1 reads frames sequentially on the main thread."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2

from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.infrastructure.cadence import DecimationGrid
from tratrac.infrastructure.video.window import FrameWindow


class OpenCvVideoSource:
	"""Reads frames from a video container via ``cv2.VideoCapture``.

	An optional ``[start_seconds, end_seconds]`` window trims the analyzed range:
	the source seeks to the start frame and stops after the (inclusive) end frame.
	Frame indices stay absolute, so exported TIMESTEPs remain on the source
	video's clock. See vault/17_time_window.md.

	An optional ``process_fps`` caps the processing cadence: frames off the target
	grid are skipped with ``grab`` (no decode) so detection/tracking run on fewer
	frames — a deliberate speed-for-quality trade (more BoT-SORT ID switches). It is
	independent of, and stacks before, export-side timestep decimation. Skipped
	frames keep absolute indices, so timestamps and any replay schedule still line
	up. See vault/18_timestep_precision.md.
	"""

	def __init__(
		self,
		path: Path,
		*,
		start_seconds: float | None = None,
		end_seconds: float | None = None,
		process_fps: float | None = None,
	) -> None:
		if start_seconds is not None and start_seconds < 0:
			raise ValueError(f"start_seconds must be non-negative, got {start_seconds}.")
		if end_seconds is not None and end_seconds <= 0:
			raise ValueError(f"end_seconds must be positive, got {end_seconds}.")
		if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
			raise ValueError(
				f"end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds})."
			)
		if process_fps is not None and process_fps <= 0:
			raise ValueError(f"process_fps must be positive, got {process_fps}.")
		self._path = path
		self._start_seconds = start_seconds
		self._end_seconds = end_seconds
		self._process_fps = process_fps
		self._capture: Any = None
		self._metadata: VideoMetadata | None = None
		self._window: FrameWindow | None = None

	@property
	def metadata(self) -> VideoMetadata:
		if self._metadata is None:
			raise RuntimeError("VideoSource must be used as a context manager.")
		return self._metadata

	def frames(self) -> Iterator[Frame]:
		capture = self._require_capture()
		window = self._require_window()
		fps = self.metadata.fps
		grid = self._make_grid(fps)
		index = window.start_frame
		while window.includes(index):
			if grid is None or grid.accepts(index / fps):
				ok, pixels = capture.read()
				if not ok:
					return
				yield Frame(index=index, pixels=pixels)
			elif not capture.grab():
				# Advance without decoding; cheap skip of an off-grid frame.
				return
			index += 1

	def _make_grid(self, fps: float) -> DecimationGrid | None:
		"""A fresh processing-cadence grid for this pass, or ``None`` (every frame)."""
		if self._process_fps is None:
			return None
		return DecimationGrid(min_interval_seconds=1.0 / self._process_fps, fps=fps)

	def _processed_count(self, window: FrameWindow, fps: float, window_total: int) -> int:
		"""How many frames this run will actually process under the cadence grid.

		The full window count when not decimating (or when the container reported no
		total); otherwise the grid-accepted count, computed with the *same* grid that
		drives ``frames`` so the progress denominator can't drift from reality."""
		if self._process_fps is None or window.frame_count is None:
			return window_total
		grid = DecimationGrid(min_interval_seconds=1.0 / self._process_fps, fps=fps)
		start = window.start_frame
		return sum(1 for i in range(start, start + window.frame_count) if grid.accepts(i / fps))

	def __enter__(self) -> OpenCvVideoSource:
		capture = cv2.VideoCapture(str(self._path))
		if not capture.isOpened():
			raise RuntimeError(f"Cannot open video: {self._path}")
		try:
			fps = float(capture.get(cv2.CAP_PROP_FPS))
			real_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
			window = FrameWindow.from_seconds(
				fps=fps,
				total_frames=real_total,
				start_seconds=self._start_seconds,
				end_seconds=self._end_seconds,
			)
			window_total = window.frame_count if window.frame_count is not None else real_total
			metadata = VideoMetadata(
				width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
				height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
				fps=fps,
				# total_frames is "frames this run will process": when we decimate, count
				# the grid-accepted frames so progress has the right denominator.
				total_frames=self._processed_count(window, fps, window_total),
			)
			if window.start_frame > 0:
				# Keyframe-only codecs may seek to a nearby keyframe; we still label
				# frames from the requested start so timestamps stay on the clip clock.
				capture.set(cv2.CAP_PROP_POS_FRAMES, float(window.start_frame))
		except Exception:
			capture.release()
			raise
		self._capture = capture
		self._window = window
		self._metadata = metadata
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
		self._window = None

	def _require_capture(self) -> Any:
		if self._capture is None:
			raise RuntimeError("VideoSource must be used as a context manager.")
		return self._capture

	def _require_window(self) -> FrameWindow:
		if self._window is None:
			raise RuntimeError("VideoSource must be used as a context manager.")
		return self._window
