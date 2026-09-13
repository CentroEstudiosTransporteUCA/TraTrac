"""Turn a track's raw observations into smoothed VehicleStates (the post-pass core).

Pure application logic for ``tratrac-smooth`` (src/tratrac/application/SMOOTHING.md): runs the
forward+RTS Kalman smoother (``application.kalman.smooth_track``) on a track's measured
centroids, then reads position/velocity/acceleration out of the smoothed state — never
finite-differencing noisy position. Measurements are in pixels; outputs are scaled to
metric via a ``ScaleFunction`` (the GSD calibration, ``infrastructure/transform/records.py``)
exactly as the EMA estimator does, so the result feeds ``SsamTrjExporter`` unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tratrac.application.kalman import SmoothedSample, smooth_track
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.ports import InvertibleCoordinateTransform
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.transform.records import ScaleFunction

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
	scale: ScaleFunction,
	*,
	pos_noise: float,
	jerk: float,
) -> list[VehicleState]:
	"""Smooth one track's observations into per-frame ``VehicleState``s (aligned to ``samples``).

	``samples`` must be in frame order. ``scale`` is the GSD metres-per-pixel calibration.
	Returns one state per sample; an empty input yields an empty list.
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
	scale: ScaleFunction,
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
	velocity = Vector2D(kinematics.vx * scale.factor, kinematics.vy * scale.factor)
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
	accel_x, accel_y = kinematics.ax * scale.factor, kinematics.ay * scale.factor
	acceleration = (
		(velocity.dx * accel_x + velocity.dy * accel_y) / speed
		if speed >= _VELOCITY_EPSILON
		else 0.0
	)
	size_length, size_width = oriented_size if oriented_size is not None else (width, height)
	state = VehicleState(
		vehicle_id=track_id,
		timestamp_seconds=timestamp_seconds,
		centroid=Point2D(kinematics.px * scale.factor, kinematics.py * scale.factor),
		heading=heading,
		dimensions=Dimensions(
			length=max(size_length, size_width) * scale.factor,
			width=min(size_length, size_width) * scale.factor,
		),
		velocity=velocity,
		acceleration=acceleration,
	)
	return state, remembered


def _major_axis_heading(width: float, height: float) -> Heading:
	"""Fallback heading from bbox shape when speed is too low to trust velocity."""
	return Heading(1.0, 0.0) if width >= height else Heading(0.0, 1.0)


def invert_state_to_image(
	state: VehicleState, projector: InvertibleCoordinateTransform, frame_index: int
) -> tuple[Point2D, float, Dimensions]:
	"""Map a world-smoothed ``VehicleState`` back to image-space pixels via ``projector``.

	Returns ``(centroid, heading_angle_radians, dimensions)`` — deliberately not a
	``VehicleState`` (see ``infrastructure/tracks/smoothed_parquet.py``'s module docstring for
	why velocity/acceleration don't make the trip).

	Inverts **four points** independently — front/rear bumpers (recovering centroid, heading,
	and length) and left/right side points (recovering width) — rather than transforming
	``centroid``/``heading``/``dimensions`` directly. This is the same reason the SSAM ``.trj``
	format itself stores front/rear bumper points instead of centroid+heading+length: a point
	transforms correctly under an arbitrary coordinate change (a homography, here), a
	direction+magnitude pair does not. This is a genuine geometric inverse of the one smoothing
	pass that already ran — not a second smoothing pass in a different space (see
	``application/SMOOTHING.md``'s "Dual-space export" section for why that would be a real
	problem: perspective-varying jitter looks different in each space, so smoothing twice risks
	damping real motion in whichever space wasn't the one physically justified).
	"""
	half_width = state.dimensions.width / 2.0
	perpendicular = Heading(-state.heading.dy, state.heading.dx)

	front_img = projector.reverse(state.front_bumper, frame_index)
	rear_img = projector.reverse(state.rear_bumper, frame_index)
	left_img = projector.reverse(
		state.centroid.translate_by(perpendicular.as_vector_with_magnitude(half_width)),
		frame_index,
	)
	right_img = projector.reverse(
		state.centroid.translate_by(perpendicular.reversed().as_vector_with_magnitude(half_width)),
		frame_index,
	)

	axis = rear_img.displacement_to(front_img)
	heading = axis.normalized() if axis.magnitude > 0.0 else Heading(1.0, 0.0)
	length = max(axis.magnitude, 1e-6)
	width = max(left_img.displacement_to(right_img).magnitude, 1e-6)
	centroid = Point2D((front_img.x + rear_img.x) / 2.0, (front_img.y + rear_img.y) / 2.0)
	angle = math.atan2(heading.dy, heading.dx)
	return centroid, angle, Dimensions(length=length, width=width)


def unscale_state_to_image(
	state: VehicleState, scale: ScaleFunction
) -> tuple[Point2D, float, Dimensions]:
	"""Undo ``build_state``'s metric scaling (no homography involved) to recover pixels.

	Used when the transforms file has scale rows, not homography rows: the run's
	``ScaleFunction`` (the GSD metric scale,
	``src/tratrac/calibration/GSD_CALIBRATION.md`` — config-only, zero-defaults
	means it's essentially never exactly ``1.0``) still applied, so ``state``'s
	position/dimensions are metric, not raw pixels, even without a homography.
	"""
	return (
		Point2D(state.centroid.x / scale.factor, state.centroid.y / scale.factor),
		math.atan2(state.heading.dy, state.heading.dx),
		Dimensions(
			length=state.dimensions.length / scale.factor,
			width=state.dimensions.width / scale.factor,
		),
	)
