"""Ground Sample Distance computation for aerial photogrammetry.

See vault/05_5_mvp1_75.md for the derivation and MVP context.
"""

from __future__ import annotations


def ground_sample_distance(
	*,
	sensor_width_mm: float,
	focal_length_mm: float,
	altitude_m: float,
	image_width_pixels: int,
) -> float:
	"""Compute metres per pixel from camera geometry and flight altitude.

	GSD = (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_pixels)

	Inputs must all be strictly positive; the result is in metres per pixel.
	"""
	if sensor_width_mm <= 0:
		raise ValueError(f"sensor_width_mm must be positive, got {sensor_width_mm}.")
	if focal_length_mm <= 0:
		raise ValueError(f"focal_length_mm must be positive, got {focal_length_mm}.")
	if altitude_m <= 0:
		raise ValueError(f"altitude_m must be positive, got {altitude_m}.")
	if image_width_pixels <= 0:
		raise ValueError(f"image_width_pixels must be positive, got {image_width_pixels}.")
	return (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_pixels)
