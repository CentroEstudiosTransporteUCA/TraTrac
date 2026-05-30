"""CompositeTrajectoryExporter: fans one trajectory stream out to several exporters.

A GoF Composite over the ``TrajectoryExporter`` port: it *is* a
``TrajectoryExporter`` and forwards every call to each child in order. This is how
"write the .trj and the overlay video in one run" is expressed — compose an
``SsamTrjExporter`` (optionally wrapped in ``DecimatingTrajectoryExporter`` to
thin only the .trj) with an ``OverlayVideoExporter``. The pipeline still sees a
single exporter and is unchanged. See vault/20_video_export.md.

Lifecycle is all-or-nothing on enter (if a child fails to enter, the ones already
entered are exited) and best-effort on exit (every child is exited even if one
raises; the first error propagates).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

from tratrac.domain.frame import Frame
from tratrac.domain.ports import TrajectoryExporter
from tratrac.domain.vehicle import VehicleState


class CompositeTrajectoryExporter:
	"""Broadcasts ``emit_frame`` and the context-manager lifecycle to each child."""

	def __init__(self, exporters: Sequence[TrajectoryExporter]) -> None:
		if not exporters:
			raise ValueError("CompositeTrajectoryExporter needs at least one exporter.")
		self._exporters = list(exporters)

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None:
		for exporter in self._exporters:
			exporter.emit_frame(timestamp_seconds, states, frame)

	def __enter__(self) -> CompositeTrajectoryExporter:
		entered: list[TrajectoryExporter] = []
		try:
			for exporter in self._exporters:
				exporter.__enter__()
				entered.append(exporter)
		except Exception as exc:
			# Unwind the children that did open so none leak an unclosed file/writer.
			for opened in reversed(entered):
				opened.__exit__(type(exc), exc, exc.__traceback__)
			raise
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		first: BaseException | None = None
		# Reverse order so children close in the opposite order they opened.
		for exporter in reversed(self._exporters):
			try:
				exporter.__exit__(exc_type, exc_val, exc_tb)
			except Exception as exc:  # keep closing the rest, re-raise the first later
				if first is None:
					first = exc
		if first is not None:
			raise first
