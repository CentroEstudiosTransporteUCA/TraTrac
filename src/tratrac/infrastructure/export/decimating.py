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

from tratrac.domain.ports import TrajectoryExporter
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.cadence import DecimationGrid


class DecimatingTrajectoryExporter:
	"""``TrajectoryExporter`` wrapper that emits at most one timestep per interval.

	The first frame is always forwarded; it anchors the emission grid. Subsequent
	frames are forwarded when their timestamp reaches the next grid point, within
	half a frame so spacing snaps to the nearest available frame. An interval at or
	below the frame duration degrades to forwarding every frame. The grid math is
	the shared ``DecimationGrid`` (vault/18_timestep_precision.md).
	"""

	def __init__(
		self,
		inner: TrajectoryExporter,
		*,
		min_interval_seconds: float,
		fps: float,
	) -> None:
		self._inner = inner
		self._grid = DecimationGrid(min_interval_seconds=min_interval_seconds, fps=fps)

	def emit_frame(self, timestamp_seconds: float, states: list[VehicleState]) -> None:
		if self._grid.accepts(timestamp_seconds):
			self._inner.emit_frame(timestamp_seconds, states)

	def __enter__(self) -> DecimatingTrajectoryExporter:
		self._inner.__enter__()
		# Reset the grid so the decorator is reusable across context-manager uses.
		self._grid.reset()
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self._inner.__exit__(exc_type, exc_val, exc_tb)
