"""Detection types: what the detector emits and what the tracker labels with identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tratrac.domain.geometry import BoundingBox


class VehicleClass(Enum):
	"""Vehicle categories MVP1 cares about (COCO-aligned)."""

	CAR = "car"
	MOTORCYCLE = "motorcycle"
	BUS = "bus"
	TRUCK = "truck"


@dataclass(frozen=True, slots=True)
class Detection:
	"""A single-frame bbox detection, untracked."""

	bbox: BoundingBox
	score: float
	vehicle_class: VehicleClass

	def __post_init__(self) -> None:
		if not 0.0 <= self.score <= 1.0:
			raise ValueError(f"score must be in [0, 1], got {self.score}.")


@dataclass(frozen=True, slots=True)
class TrackedDetection:
	"""A detection that has been assigned a stable cross-frame identity by the tracker."""

	track_id: int
	detection: Detection

	def __post_init__(self) -> None:
		if self.track_id < 0:
			raise ValueError(f"track_id must be non-negative, got {self.track_id}.")
