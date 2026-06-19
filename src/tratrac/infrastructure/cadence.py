"""DecimationGrid: an anchored cadence grid for thinning a frame/timestamp stream.

Shared, pure mechanism behind two decimation features (see
vault/18_timestep_precision.md): the export-side `DecimatingTrajectoryExporter`
and the decode-side processing cap in `OpenCvVideoSource`. Both ask the same
question — "given this frame's timestamp, is it on the grid?" — so the policy
lives here once.

The first timestamp is always accepted and **anchors** the grid at
``anchor + k * interval``. A later timestamp is accepted once it reaches the next
grid point, within half a frame (``0.5 / fps``) so spacing snaps to the nearest
available frame instead of always rounding up. An interval at or below the frame
duration degrades cleanly to accepting every frame. No drift: the grid (not a
running counter) is the reference.
"""

from __future__ import annotations


class DecimationGrid:
	"""Decides, per timestamp, whether it falls on the anchored emission grid.

	Stateful and single-pass: call ``accepts`` once per frame in stream order.
	``reset`` re-anchors so the grid can be reused across passes.
	"""

	def __init__(self, *, min_interval_seconds: float, fps: float) -> None:
		if min_interval_seconds <= 0.0:
			raise ValueError(f"min_interval_seconds must be positive, got {min_interval_seconds}.")
		if fps <= 0.0:
			raise ValueError(f"fps must be positive, got {fps}.")
		self._interval = min_interval_seconds
		# Half a frame: lets a frame microscopically before a grid point still count,
		# so realized spacing tracks the requested interval rather than rounding up.
		self._epsilon = 0.5 / fps
		self._next_at: float | None = None

	def accepts(self, timestamp_seconds: float) -> bool:
		"""Whether ``timestamp_seconds`` is on the grid; advances the grid if so."""
		if self._next_at is None:
			self._next_at = timestamp_seconds + self._interval
			return True
		if timestamp_seconds >= self._next_at - self._epsilon:
			# Advance past this timestamp. The loop (not a single step) keeps the
			# schedule correct when the interval is finer than the frame spacing,
			# where it collapses to accepting every frame.
			while self._next_at <= timestamp_seconds + self._epsilon:
				self._next_at += self._interval
			return True
		return False

	def reset(self) -> None:
		"""Re-anchor the grid (forget the previous pass)."""
		self._next_at = None
