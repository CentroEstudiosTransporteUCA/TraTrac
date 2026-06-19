"""Console progress reporter: renders progress events as a tqdm progress bar.

An infrastructure adapter for the ``ProgressReporter`` port. Drives a single tqdm
bar from the pipeline's event stream — created on ``ProcessingStarted`` (its total
is the number of frames this run will process), advanced on each ``FrameProcessed``
with the active-track count in the postfix, and closed on finish/failure. Renders
to stderr by default so it never pollutes stdout, where the CLI prints its final
summary. tqdm throttles its own redraws, so no manual rate-limiting is needed.

The tqdm bar is created through an injected ``bar_factory`` seam so the reporter is
unit-testable without driving real terminal output. See vault/14_progress_reporting.md.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Protocol, TextIO

from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
	ProgressEvent,
)


class _Bar(Protocol):
	"""The slice of the tqdm API this reporter drives (the seam for testing)."""

	def update(self, n: int) -> Any: ...

	def set_postfix(self, *args: Any, **kwargs: Any) -> Any: ...

	def close(self) -> Any: ...


BarFactory = Callable[..., _Bar]


def _tqdm_bar(**kwargs: Any) -> _Bar:
	"""Default factory: a real tqdm bar. Imported lazily so tests can inject a fake."""
	from tqdm import tqdm

	bar: _Bar = tqdm(**kwargs)
	return bar


class ConsoleProgressReporter:
	"""Renders progress as a tqdm bar driven by the progress-event stream."""

	def __init__(
		self, *, stream: TextIO | None = None, bar_factory: BarFactory = _tqdm_bar
	) -> None:
		self._stream = stream if stream is not None else sys.stderr
		self._bar_factory = bar_factory
		self._bar: _Bar | None = None
		self._advanced = 0  # frames already pushed into the bar (to compute the delta)

	def receive(self, event: ProgressEvent) -> None:
		match event:
			case ProcessingStarted():
				self._on_started(event)
			case FrameProcessed():
				self._on_frame(event)
			case ProcessingFinished():
				self._close()
			case ProcessingFailed():
				self._on_failed(event)
			case _:
				# Unknown future event: a console reporter safely ignores it.
				pass

	def _on_started(self, event: ProcessingStarted) -> None:
		meta = event.metadata
		self._advanced = 0
		self._bar = self._bar_factory(
			total=meta.total_frames if meta.total_frames > 0 else None,
			desc="Processing",
			unit="frame",
			file=self._stream,
			dynamic_ncols=True,
		)

	def _on_frame(self, event: FrameProcessed) -> None:
		if self._bar is None:
			return
		delta = event.frames_done - self._advanced
		if delta <= 0:
			return
		self._advanced = event.frames_done
		# Postfix without an immediate redraw; update() does the throttled refresh.
		self._bar.set_postfix(tracks=event.active_tracks, refresh=False)
		self._bar.update(delta)

	def _on_failed(self, event: ProcessingFailed) -> None:
		self._close()
		self._stream.write(f"Failed at frame {event.frame_index}: {event.error}\n")
		self._stream.flush()

	def _close(self) -> None:
		if self._bar is not None:
			self._bar.close()
			self._bar = None
