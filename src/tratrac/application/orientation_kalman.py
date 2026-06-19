"""KalmanOrientationEstimator: streaming forward-Kalman kinematics behind the port.

The causal counterpart to the offline ``tratrac-smooth`` pass: a constant-acceleration
forward Kalman filter per track, run inside the streaming pipeline so the pipeline's own
``.trj`` is de-jittered live (the real-time path). It implements the same
``OrientationEstimator`` port as ``EmaOrientationEstimator`` — a config-selected adapter
swap (``orientation.method = kalman``); EMA stays the default. Being causal it has settling
lag, so the zero-phase RTS post-pass is preferred when the whole clip is available. See
vault/22_smoothing.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tratrac.application.kalman import KinematicKalmanFilter
from tratrac.application.track_smoothing import build_state
from tratrac.domain.detection import TrackedDetection
from tratrac.domain.geometry import Heading
from tratrac.domain.vehicle import VehicleState

# Drop a track's filter once it has not been seen for this long, so per-track state does
# not accumulate for the whole run (the EMA estimator's unbounded-history gap). Generous,
# so a filter survives ordinary occlusions and a re-appearing track id continues smoothly.
_EVICTION_HORIZON_SECONDS = 5.0


@dataclass(slots=True)
class _TrackFilter:
	filter: KinematicKalmanFilter
	last_timestamp: float
	last_heading: Heading | None = None


class KalmanOrientationEstimator:
	"""Implements ``OrientationEstimator`` with a per-track forward Kalman filter."""

	def __init__(self, *, meters_per_pixel: float, pos_noise: float, jerk: float) -> None:
		if meters_per_pixel <= 0.0:
			raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}.")
		self._scale = meters_per_pixel
		self._pos_noise = pos_noise
		self._jerk = jerk
		self._tracks: dict[int, _TrackFilter] = {}

	def estimate(
		self, tracked: Sequence[TrackedDetection], timestamp_seconds: float
	) -> list[VehicleState]:
		states = [self._estimate_one(item, timestamp_seconds) for item in tracked]
		self._evict_stale(timestamp_seconds)
		return states

	def _estimate_one(self, item: TrackedDetection, timestamp_seconds: float) -> VehicleState:
		box = item.detection.bbox
		center = box.center
		entry = self._tracks.get(item.track_id)
		if entry is None:
			# First sighting: the filter initialises from the measurement (dt ignored).
			entry = _TrackFilter(
				filter=KinematicKalmanFilter(pos_noise=self._pos_noise, jerk=self._jerk),
				last_timestamp=timestamp_seconds,
			)
			self._tracks[item.track_id] = entry
			kinematics = entry.filter.observe(center.x, center.y, 0.0)
		else:
			# Clamp to a tiny positive step so a repeated/out-of-order timestamp still
			# updates (the filter requires dt > 0 after the first observation).
			dt = max(timestamp_seconds - entry.last_timestamp, 1.0e-6)
			entry.last_timestamp = timestamp_seconds
			kinematics = entry.filter.observe(center.x, center.y, dt)
		state, entry.last_heading = build_state(
			track_id=item.track_id,
			timestamp_seconds=timestamp_seconds,
			kinematics=kinematics,
			width=box.width,
			height=box.height,
			scale=self._scale,
			last_heading=entry.last_heading,
		)
		return state

	def _evict_stale(self, now: float) -> None:
		stale = [
			track_id
			for track_id, entry in self._tracks.items()
			if now - entry.last_timestamp > _EVICTION_HORIZON_SECONDS
		]
		for track_id in stale:
			del self._tracks[track_id]
