"""Frame and video metadata value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VideoMetadata:
	"""Static properties of a video, read once when the source opens."""

	width: int
	height: int
	fps: float
	total_frames: int

	def __post_init__(self) -> None:
		if self.width <= 0 or self.height <= 0:
			raise ValueError(f"Invalid frame dimensions: {self.width}x{self.height}.")
		if self.fps <= 0:
			raise ValueError(f"FPS must be positive, got {self.fps}.")


@dataclass(frozen=True, slots=True)
class Frame:
	"""A single decoded video frame, BGR uint8 per OpenCV convention."""

	index: int
	pixels: NDArray[np.uint8]

	def timestamp_seconds(self, fps: float) -> float:
		return self.index / fps
