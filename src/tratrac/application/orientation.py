"""Orientation + kinematics estimator.

Turns a stream of per-frame ``TrackedDetection``s into ``VehicleState``s by
tracking each track's recent centroids and deriving velocity, acceleration,
and heading from that history.

* Heading is a motion-magnitude-weighted EMA of the velocity direction. Fast
  motion fully trusts the velocity vector; slow motion blends toward the
  cached heading so detector-bbox jitter on stationary vehicles doesn't make
  the orientation arrow spin.
* When a track has never moved meaningfully, heading falls back to the bbox
  major-axis direction.
* Unit-aware via the ``meters_per_pixel`` constructor argument (default
  ``1.0`` reproduces MVP1 pixel-as-meter behaviour). When a real GSD is
  supplied (MVP1.75+, see ``vault/05_5_mvp1_75.md``), all metric quantities
  the estimator publishes — centroid, length, width, velocity, acceleration —
  are in metres / m·s⁻¹ / m·s⁻².
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tratrac.domain.detection import TrackedDetection
from tratrac.domain.geometry import BoundingBox, Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState

_VELOCITY_EPSILON = 1e-6
# Speed (in coordinate-frame units / sec) at which we fully trust the velocity
# vector as the heading. Below this we blend with the cached last_good_heading.
# Tuned for image-space pixels: real moving cars run 100-300 px/s; YOLO bbox
# jitter on a stationary vehicle is typically <30 px/s.
_FULL_TRUST_SPEED = 50.0


@dataclass(slots=True)
class _TrackHistory:
	centroids: deque[Point2D]
	timestamps: deque[float]
	last_velocity: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
	last_good_heading: Heading | None = None


class OrientationEstimator:
	"""Builds ``VehicleState``s from per-frame ``TrackedDetection``s."""

	def __init__(
		self,
		*,
		smoothing_window: int = 5,
		meters_per_pixel: float = 1.0,
	) -> None:
		if smoothing_window < 2:
			raise ValueError(f"smoothing_window must be >= 2, got {smoothing_window}.")
		if meters_per_pixel <= 0:
			raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}.")
		self._window = smoothing_window
		self._scale = meters_per_pixel
		self._history: dict[int, _TrackHistory] = {}

	def estimate(self, tracked: TrackedDetection, timestamp_seconds: float) -> VehicleState:
		track_id = tracked.track_id
		bbox = tracked.detection.bbox
		# Convert pixel-space measurements to world units (metres if scale is a
		# real GSD, "pixel-metres" if scale is 1.0). Doing this once at the
		# boundary means velocity and acceleration fall out in the right units
		# automatically.
		scale = self._scale
		centroid_world = Point2D(bbox.center.x * scale, bbox.center.y * scale)

		history = self._history.get(track_id)
		if history is None:
			history = _TrackHistory(
				centroids=deque(maxlen=self._window),
				timestamps=deque(maxlen=self._window),
			)
			self._history[track_id] = history
		history.centroids.append(centroid_world)
		history.timestamps.append(timestamp_seconds)

		velocity = self._compute_velocity(history)
		acceleration = self._compute_acceleration(velocity, history)
		heading = self._compute_heading(velocity, bbox, history)
		history.last_velocity = velocity

		return VehicleState(
			vehicle_id=track_id,
			timestamp_seconds=timestamp_seconds,
			centroid=centroid_world,
			heading=heading,
			dimensions=Dimensions(
				length=bbox.major_axis_length * scale,
				width=bbox.minor_axis_length * scale,
			),
			velocity=velocity,
			acceleration=acceleration,
		)

	def forget(self, track_id: int) -> None:
		"""Drop history for a track that has terminated."""
		self._history.pop(track_id, None)

	@staticmethod
	def _compute_velocity(history: _TrackHistory) -> Vector2D:
		if len(history.centroids) < 2:
			return Vector2D(0.0, 0.0)
		dt = history.timestamps[-1] - history.timestamps[0]
		if dt <= 0.0:
			return Vector2D(0.0, 0.0)
		displacement = history.centroids[0].displacement_to(history.centroids[-1])
		return displacement.scaled_by(1.0 / dt)

	@staticmethod
	def _compute_acceleration(current: Vector2D, history: _TrackHistory) -> Vector2D:
		if len(history.timestamps) < 2:
			return Vector2D(0.0, 0.0)
		dt = history.timestamps[-1] - history.timestamps[-2]
		if dt <= 0.0:
			return Vector2D(0.0, 0.0)
		prior = history.last_velocity
		return Vector2D((current.dx - prior.dx) / dt, (current.dy - prior.dy) / dt)

	@staticmethod
	def _compute_heading(velocity: Vector2D, bbox: BoundingBox, history: _TrackHistory) -> Heading:
		speed = velocity.magnitude

		# True zero (or numerical noise): no info in velocity at all.
		if speed < _VELOCITY_EPSILON:
			if history.last_good_heading is not None:
				return history.last_good_heading
			return _bbox_major_axis_heading(bbox)

		velocity_heading = velocity.normalized()

		# First-ever motion for this track. Adopt direction outright.
		if history.last_good_heading is None:
			history.last_good_heading = velocity_heading
			return velocity_heading

		# Velocity-magnitude-weighted blend with the cached heading. Low speed
		# (jitter) barely moves the arrow; high speed snaps it to the motion.
		alpha = min(1.0, speed / _FULL_TRUST_SPEED)
		last = history.last_good_heading
		blended = Vector2D(
			(1.0 - alpha) * last.dx + alpha * velocity_heading.dx,
			(1.0 - alpha) * last.dy + alpha * velocity_heading.dy,
		)
		if blended.magnitude < _VELOCITY_EPSILON:
			# Anti-parallel cancellation at alpha=0.5 — keep the previous heading.
			return history.last_good_heading
		smoothed = blended.normalized()
		history.last_good_heading = smoothed
		return smoothed


def _bbox_major_axis_heading(bbox: BoundingBox) -> Heading:
	if bbox.width >= bbox.height:
		return Heading(1.0, 0.0)
	return Heading(0.0, 1.0)
