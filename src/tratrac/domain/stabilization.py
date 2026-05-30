"""Stabilization value objects emitted while a video is processed.

A ``FrameTransform`` pairs a processed frame's index with the ego-motion transform
estimated for it (current frame -> global stabilization frame). It is the record a
``TransformSink`` persists, so a downstream tool can invert each one to map
stabilized coordinates back onto the raw frame they were derived from. Sibling of
``domain/timing.py``'s ``StepTiming``. See vault/05_75_mvp1_9.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Transform2D


@dataclass(frozen=True, slots=True)
class FrameTransform:
	"""The ego-motion transform estimated for one processed frame.

	``frame_index`` is the source ``Frame.index`` (absolute, even under a trimmed
	analysis window). ``transform`` maps that frame's pixel coordinates into the
	continuous global stabilization frame; its inverse maps global coordinates back
	onto the raw frame.
	"""

	frame_index: int
	transform: Transform2D
