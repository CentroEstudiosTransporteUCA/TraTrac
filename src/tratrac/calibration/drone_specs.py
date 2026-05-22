"""Per-drone-model sensor + focal-length registry.

Values come from each manufacturer's published spec sheet. **Verify before
shipping for a new model** — the GSD formula is unforgiving of bad sensor or
focal numbers.

The focal length is the *real* focal length (often called "actual" or
"native"), NOT the 35 mm-equivalent value photographers quote.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DroneSpec:
	"""Camera-relevant specs for a drone model. All lengths in millimetres."""

	sensor_width_mm: float
	focal_length_mm: float

	def __post_init__(self) -> None:
		if self.sensor_width_mm <= 0:
			raise ValueError(f"sensor_width_mm must be positive, got {self.sensor_width_mm}.")
		if self.focal_length_mm <= 0:
			raise ValueError(f"focal_length_mm must be positive, got {self.focal_length_mm}.")


# Add new models here. Keys are the strings users pass to --drone-model.
_REGISTRY: dict[str, DroneSpec] = {
	# DJI Mavic 3 / 3 Cine / 3 Classic (Hasselblad L2D-20c, 4/3 CMOS).
	"mavic_3": DroneSpec(sensor_width_mm=17.3, focal_length_mm=12.29),
	# DJI Mavic 2 Pro (Hasselblad L1D-20c, 1" CMOS).
	"mavic_2_pro": DroneSpec(sensor_width_mm=13.2, focal_length_mm=10.26),
	# DJI Air 2S (1" CMOS).
	"air_2s": DroneSpec(sensor_width_mm=13.2, focal_length_mm=8.4),
	# DJI Mini 3 Pro / Mini 4 Pro (1/1.3" CMOS).
	"mini_3_pro": DroneSpec(sensor_width_mm=6.4, focal_length_mm=6.7),
	"mini_4_pro": DroneSpec(sensor_width_mm=6.4, focal_length_mm=6.7),
}


def lookup(model: str) -> DroneSpec:
	"""Return specs for a drone model. Case-insensitive; raises if unknown."""
	key = model.lower()
	if key not in _REGISTRY:
		known = ", ".join(sorted(_REGISTRY))
		raise KeyError(f"Unknown drone model {model!r}. Known: {known}.")
	return _REGISTRY[key]


def known_models() -> list[str]:
	"""Sorted list of registered drone-model keys."""
	return sorted(_REGISTRY)
