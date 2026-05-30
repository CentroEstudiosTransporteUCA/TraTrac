"""Timestep-decimating exporter decorator: thins the exported TIMESTEP stream.

Wraps any ``TrajectoryExporter`` and forwards ``emit_frame`` only once a minimum
interval has elapsed since the last forwarded timestep. Detection, tracking, and
orientation still run on every frame upstream (the pipeline is untouched); only
the *output* cadence is downsampled. This keeps BoT-SORT's per-frame association
quality while letting the ``.trj`` carry coarser TIMESTEPs. See
vault/18_timestep_precision.md.

A decorator (like ``TimedExporter``) so the policy stays out of the pipeline loop
and the concrete writer stays dumb.
"""

from __future__ import annotations

from types import TracebackType

from tratrac.domain.frame import Frame
from tratrac.domain.ports import TrajectoryExporter
from tratrac.domain.vehicle import VehicleState


class DecimatingTrajectoryExporter:
	"""``TrajectoryExporter`` wrapper that emits at most one timestep per interval.

	The first frame is always forwarded; it anchors the emission grid. Subsequent
	frames are forwarded when their timestamp reaches the next grid point
	(``anchor + k * interval``), within half a frame so spacing snaps to the
	nearest available frame rather than the next one. An interval at or below the
	frame duration degrades to forwarding every frame.
	"""

	def __init__(
		self,
		inner: TrajectoryExporter,
		*,
		min_interval_seconds: float,
		fps: float,
	) -> None:
		if min_interval_seconds <= 0.0:
			raise ValueError(f"min_interval_seconds must be positive, got {min_interval_seconds}.")
		if fps <= 0.0:
			raise ValueError(f"fps must be positive, got {fps}.")
		self._inner = inner
		self._interval = min_interval_seconds
		# Half a frame: lets a frame microscopically before a grid point still count,
		# so the emitted spacing tracks the requested interval rather than rounding up.
		self._epsilon = 0.5 / fps
		self._next_emit_at: float | None = None

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None:
		if self._next_emit_at is None:
			self._inner.emit_frame(timestamp_seconds, states, frame)
			self._next_emit_at = timestamp_seconds + self._interval
			return
		if timestamp_seconds >= self._next_emit_at - self._epsilon:
			self._inner.emit_frame(timestamp_seconds, states, frame)
			# Advance the grid past this timestamp. The loop (not a single step)
			# keeps the schedule correct when the interval is finer than the frame
			# spacing, where it collapses to emitting every frame.
			while self._next_emit_at <= timestamp_seconds + self._epsilon:
				self._next_emit_at += self._interval

	def __enter__(self) -> DecimatingTrajectoryExporter:
		self._inner.__enter__()
		# Reset the grid so the decorator is reusable across context-manager uses.
		self._next_emit_at = None
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self._inner.__exit__(exc_type, exc_val, exc_tb)
