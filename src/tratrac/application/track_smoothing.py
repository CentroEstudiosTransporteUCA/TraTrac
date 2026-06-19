"""Turn a track's raw observations into smoothed VehicleStates (the post-pass core).

Pure application logic for ``tratrac-smooth`` (vault/22_smoothing.md): runs the
forward+RTS Kalman smoother (``application.kalman.smooth_track``) on a track's measured
centroids, then reads position/velocity/acceleration out of the smoothed state — never
finite-differencing noisy position. Measurements are in pixels; outputs are scaled to
metric (by ``meters_per_pixel``) exactly as the EMA estimator does, so the result feeds
``SsamTrjExporter`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.application.kalman import smooth_track
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState

# Below this speed the velocity direction is pure jitter; fall back to the last good
# heading or the bbox major axis (mirrors EmaOrientationEstimator).
_VELOCITY_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class TrackSample:
	"""One raw observation in a track: when, the measured centre, and the bbox size (px)."""

	frame_index: int
	timestamp_seconds: float
	center: Point2D
	width: float
	height: float


def smooth_to_states(
	track_id: int,
	samples: list[TrackSample],
	scale: float,
	*,
	pos_noise: float,
	jerk: float,
) -> list[VehicleState]:
	"""Smooth one track's observations into per-frame ``VehicleState``s (aligned to ``samples``).

	``samples`` must be in frame order. ``scale`` is metres-per-pixel. Returns one state
	per sample; an empty input yields an empty list.
	"""
	if not samples:
		return []
	smoothed = smooth_track(
		[s.center.x for s in samples],
		[s.center.y for s in samples],
		[s.timestamp_seconds for s in samples],
		pos_noise=pos_noise,
		jerk=jerk,
	)
	states: list[VehicleState] = []
	last_heading: Heading | None = None
	for sample, point in zip(samples, smoothed, strict=True):
		velocity = Vector2D(point.vx * scale, point.vy * scale)
		speed = velocity.magnitude
		if speed >= _VELOCITY_EPSILON:
			last_heading = velocity.normalized()
			heading = last_heading
		else:
			heading = last_heading or _major_axis_heading(sample.width, sample.height)
		# Longitudinal acceleration = d|v|/dt = (v·a)/|v| (the SSAM Acceleration field).
		accel_x, accel_y = point.ax * scale, point.ay * scale
		acceleration = (
			(velocity.dx * accel_x + velocity.dy * accel_y) / speed
			if speed >= _VELOCITY_EPSILON
			else 0.0
		)
		states.append(
			VehicleState(
				vehicle_id=track_id,
				timestamp_seconds=sample.timestamp_seconds,
				centroid=Point2D(point.px * scale, point.py * scale),
				heading=heading,
				dimensions=Dimensions(
					length=max(sample.width, sample.height) * scale,
					width=min(sample.width, sample.height) * scale,
				),
				velocity=velocity,
				acceleration=acceleration,
			)
		)
	return states


def _major_axis_heading(width: float, height: float) -> Heading:
	"""Fallback heading from bbox shape when speed is too low to trust velocity."""
	return Heading(1.0, 0.0) if width >= height else Heading(0.0, 1.0)
