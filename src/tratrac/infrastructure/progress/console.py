"""Console progress reporter: renders progress events to a text stream.

An infrastructure adapter for the ``ProgressReporter`` port. Writes a single
in-place updating line to stderr (so it never pollutes stdout, where the CLI
prints its final summary), throttled to stay readable on long videos.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
	ProgressEvent,
)


class ConsoleProgressReporter:
	"""Renders progress to a stream, updating one line in place.

	``min_interval_seconds`` throttles per-frame redraws by wall-clock time; the
	final frame (``fraction >= 1.0``) and the start/finish/failure events always
	render.
	"""

	def __init__(self, *, stream: TextIO | None = None, min_interval_seconds: float = 0.1) -> None:
		if min_interval_seconds < 0.0:
			raise ValueError(f"min_interval_seconds must be >= 0, got {min_interval_seconds}.")
		self._stream = stream if stream is not None else sys.stderr
		self._min_interval = min_interval_seconds
		# -inf guarantees the first frame always draws regardless of the clock.
		self._last_draw = float("-inf")
		self._line_open = False

	def receive(self, event: ProgressEvent) -> None:
		match event:
			case ProcessingStarted():
				self._on_started(event)
			case FrameProcessed():
				self._on_frame(event)
			case ProcessingFinished():
				self._on_finished(event)
			case ProcessingFailed():
				self._on_failed(event)
			case _:
				# Unknown future event: a console reporter safely ignores it.
				pass

	def _on_started(self, event: ProcessingStarted) -> None:
		meta = event.metadata
		self._write(
			f"Processing {meta.total_frames} frames "
			f"({meta.width}x{meta.height} @ {meta.fps:.1f} fps)\n"
		)

	def _on_frame(self, event: FrameProcessed) -> None:
		now = time.monotonic()
		if event.fraction < 1.0 and now - self._last_draw < self._min_interval:
			return
		self._last_draw = now
		self._write(
			f"\r{event.percent:5.1f}% | frame {event.frame_index + 1}/{event.total_frames} "
			f"| t={event.timestamp_seconds:7.1f}s | {event.active_tracks} tracked"
		)
		self._line_open = True

	def _on_finished(self, event: ProcessingFinished) -> None:
		self._close_line()
		self._write(f"Done: {event.frames_processed} frames processed.\n")

	def _on_failed(self, event: ProcessingFailed) -> None:
		self._close_line()
		self._write(f"Failed at frame {event.frame_index}: {event.error}\n")

	def _close_line(self) -> None:
		if self._line_open:
			self._write("\n")
			self._line_open = False

	def _write(self, text: str) -> None:
		self._stream.write(text)
		self._stream.flush()
