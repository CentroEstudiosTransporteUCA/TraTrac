"""Progress events emitted while a video is processed.

A sealed-by-convention family of value objects describing *what happened*
during a run, never *how to render it*. The pipeline sends these to a
``ProgressReporter`` (see ``tratrac.domain.ports``) as a single stream of
messages; each reporter interprets the subset it cares about and ignores the
rest. New event types can be added here without breaking reporters that only
handle the older ones.

See ``vault/14_progress_reporting.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.frame import VideoMetadata


class ProgressEvent:
	"""Marker base for the progress-event family. Carries no state itself."""

	__slots__ = ()


@dataclass(frozen=True, slots=True)
class ProcessingStarted(ProgressEvent):
	"""Emitted once, before the first frame is read."""

	metadata: VideoMetadata


@dataclass(frozen=True, slots=True)
class FrameProcessed(ProgressEvent):
	"""Emitted after each frame has been processed and exported."""

	frame_index: int  # absolute zero-based frame number in the source video (provenance)
	frames_done: int  # count of frames processed so far this run (1-based)
	total_frames: int  # frames this run will process (windowed); may be 0 if unknown
	timestamp_seconds: float
	active_tracks: int

	@property
	def fraction(self) -> float:
		"""Completed fraction in [0, 1]; ``0.0`` when the total is unknown.

		Uses ``frames_done`` (count processed this run), not ``frame_index``: with
		an analysis window the index is absolute (e.g. 10659) while the total is
		the windowed count, so the index is the wrong numerator.
		"""
		if self.total_frames <= 0:
			return 0.0
		# frames_done is 1-based (the frame just finished), so the last frame reads 1.0.
		# Clamp: OpenCV's frame count can under-report, yielding more frames than total.
		return min(1.0, self.frames_done / self.total_frames)

	@property
	def percent(self) -> float:
		return self.fraction * 100.0


@dataclass(frozen=True, slots=True)
class ProcessingFinished(ProgressEvent):
	"""Emitted once, after the last frame and after the exporter is closed."""

	frames_processed: int


@dataclass(frozen=True, slots=True)
class ProcessingFailed(ProgressEvent):
	"""Emitted when a frame raises; the error is re-raised after emission."""

	frame_index: int
	error: str
