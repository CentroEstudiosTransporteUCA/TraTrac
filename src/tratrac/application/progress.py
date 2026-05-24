"""Default (silent) progress reporter for the application layer.

A Null Object so ``TrajectoryPipeline`` always has a reporter to message and
never needs to guard against ``None``. Infrastructure provides the reporters
that actually render (e.g. ``ConsoleProgressReporter``).
"""

from __future__ import annotations

from tratrac.domain.progress import ProgressEvent


class NullProgressReporter:
	"""Discards every event. The pipeline's silent default reporter."""

	def receive(self, event: ProgressEvent) -> None:
		del event  # intentionally ignored
