"""Per-step timing value objects emitted while a video is processed.

Each describes how long one pipeline step took for one frame. They are the
vocabulary the timing decorators speak and the timing sinks render. See
vault/15_step_timing.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PipelineStep(StrEnum):
	"""The per-frame pipeline steps that can be timed."""

	DETECT = "detect"
	OBSERVE = "observe"
	EGOMOTION = "ego_motion"
	STABILIZE = "stabilize"
	TRACK = "track"
	RECORD = "record"


@dataclass(frozen=True, slots=True)
class StepTiming:
	"""How long one pipeline step took for one frame.

	``frame_ordinal`` is the zero-based position of the frame in the processed
	stream. Each timing decorator counts its own calls; because every step runs
	exactly once per frame, the ordinals stay aligned across steps.
	"""

	step: PipelineStep
	frame_ordinal: int
	seconds: float
