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
	"""A single-frame bbox detection, untracked.

	``bbox`` stays the axis-aligned box unconditionally — ORB's vehicle-masking
	and the IoU-based tracking path both depend on it, even for an OBB detector.
	``angle``/``oriented_size`` are populated only by an OBB-capable detector
	(``None`` otherwise, e.g. the MVP1 YOLOv8-VisDrone/RT-DETR AABB adapters);
	see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` MVP1.5.
	"""

	bbox: BoundingBox
	score: float
	vehicle_class: VehicleClass
	# Oriented-box heading in radians, standard math convention (0 = +x axis,
	# counter-clockwise positive in image coordinates). ``None`` when the
	# detector reports only an axis-aligned box.
	angle: float | None = None
	# The rotated box's own (length, width) in pixels, distinct from ``bbox``'s
	# axis-aligned (width, height) — the real accuracy payoff OBB buys for
	# vehicle sizing (`docs/IMPLEMENTATION_PLAN.md` Group A7). ``None`` unless
	# ``angle`` is also set.
	oriented_size: tuple[float, float] | None = None

	def __post_init__(self) -> None:
		if not 0.0 <= self.score <= 1.0:
			raise ValueError(f"score must be in [0, 1], got {self.score}.")
		if (self.angle is None) != (self.oriented_size is None):
			raise ValueError("angle and oriented_size must be set together (both or neither).")
		if self.oriented_size is not None and (
			self.oriented_size[0] <= 0 or self.oriented_size[1] <= 0
		):
			raise ValueError(f"oriented_size must be positive, got {self.oriented_size}.")


@dataclass(frozen=True, slots=True)
class TrackedDetection:
	"""A detection that has been assigned a stable cross-frame identity by the tracker."""

	track_id: int
	detection: Detection

	def __post_init__(self) -> None:
		if self.track_id < 0:
			raise ValueError(f"track_id must be non-negative, got {self.track_id}.")
