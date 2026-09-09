"""Turn a track's raw observations into smoothed VehicleStates (the post-pass core).

Pure application logic for ``tratrac-smooth`` (src/tratrac/application/SMOOTHING.md): runs the
forward+RTS Kalman smoother (``application.kalman.smooth_track``) on a track's measured
centroids, then reads position/velocity/acceleration out of the smoothed state — never
finite-differencing noisy position. Measurements are in pixels; outputs are scaled to
metric (by ``meters_per_pixel``) exactly as the EMA estimator does, so the result feeds
``SsamTrjExporter`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.application.kalman import SmoothedSample, smooth_track
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
	# OBB fields (Group A7, docs/IMPLEMENTATION_PLAN.md): the detector's own oriented-box
	# angle (radians) and (length, width) in pixels, when available. ``None`` for an
	# AABB-only run — ``width``/``height`` above (the bbox) are always populated regardless.
	angle: float | None = None
	oriented_size: tuple[float, float] | None = None


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
	for sample, kinematics in zip(samples, smoothed, strict=True):
		state, last_heading = build_state(
			track_id=track_id,
			timestamp_seconds=sample.timestamp_seconds,
			kinematics=kinematics,
			width=sample.width,
			height=sample.height,
			scale=scale,
			last_heading=last_heading,
			angle=sample.angle,
			oriented_size=sample.oriented_size,
		)
		states.append(state)
	return states


def build_state(
	*,
	track_id: int,
	timestamp_seconds: float,
	kinematics: SmoothedSample,
	width: float,
	height: float,
	scale: float,
	last_heading: Heading | None,
	angle: float | None = None,
	oriented_size: tuple[float, float] | None = None,
) -> tuple[VehicleState, Heading | None]:
	"""Reconstruct a ``VehicleState`` from one smoothed sample + the source bbox size.

	Shared by the offline post-pass and the inline forward filter. Returns the state and
	the heading to remember for the next frame's low-speed fallback (only updated while
	the vehicle is actually moving). Pixel kinematics are scaled to metric by ``scale``.

	``angle``/``oriented_size`` are the detector's own OBB fields (Group A7), when
	available. ``angle`` only ever replaces the **low-speed fallback** heading, never the
	primary RTS-smoothed-velocity path: trusting a raw, per-frame, unsmoothed OBB angle
	over the two-pass Kalman result while moving would reintroduce the jitter the smoother
	exists to remove. ``oriented_size``, when present, replaces the bbox as the source of
	``Dimensions`` — the real accuracy payoff OBB buys for vehicle sizing.
	"""
	velocity = Vector2D(kinematics.vx * scale, kinematics.vy * scale)
	speed = velocity.magnitude
	if speed >= _VELOCITY_EPSILON:
		heading: Heading = velocity.normalized()
		remembered: Heading | None = heading
	elif angle is not None:
		candidate = Heading.from_angle(angle)
		heading = (
			candidate
			if last_heading is None or candidate.dot(last_heading) >= 0.0
			else candidate.reversed()
		)
		remembered = last_heading
	else:
		heading = last_heading or _major_axis_heading(width, height)
		remembered = last_heading
	# Longitudinal acceleration = d|v|/dt = (v·a)/|v| (the SSAM Acceleration field).
	accel_x, accel_y = kinematics.ax * scale, kinematics.ay * scale
	acceleration = (
		(velocity.dx * accel_x + velocity.dy * accel_y) / speed
		if speed >= _VELOCITY_EPSILON
		else 0.0
	)
	size_length, size_width = oriented_size if oriented_size is not None else (width, height)
	state = VehicleState(
		vehicle_id=track_id,
		timestamp_seconds=timestamp_seconds,
		centroid=Point2D(kinematics.px * scale, kinematics.py * scale),
		heading=heading,
		dimensions=Dimensions(
			length=max(size_length, size_width) * scale,
			width=min(size_length, size_width) * scale,
		),
		velocity=velocity,
		acceleration=acceleration,
	)
	return state, remembered


def _major_axis_heading(width: float, height: float) -> Heading:
	"""Fallback heading from bbox shape when speed is too low to trust velocity."""
	return Heading(1.0, 0.0) if width >= height else Heading(0.0, 1.0)
