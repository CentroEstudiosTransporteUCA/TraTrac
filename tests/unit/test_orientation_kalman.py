"""Tests for the streaming KalmanOrientationEstimator (inline forward filter)."""

from __future__ import annotations

import pytest

from tratrac.application.orientation_kalman import (
	_EVICTION_HORIZON_SECONDS,
	KalmanOrientationEstimator,
)
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.geometry import BoundingBox


def _tracked(
	track_id: int, cx: float, cy: float, w: float = 4.0, h: float = 2.0
) -> TrackedDetection:
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=cx - w / 2, y=cy - h / 2, width=w, height=h),
			score=0.9,
			vehicle_class=VehicleClass.CAR,
		),
	)


def _estimator() -> KalmanOrientationEstimator:
	return KalmanOrientationEstimator(meters_per_pixel=1.0, pos_noise=1.0, jerk=10.0)


class TestKalmanOrientationEstimator:
	def test_first_frame_is_stationary(self) -> None:
		est = _estimator()
		state = est.estimate([_tracked(1, 10.0, 20.0)], 0.0)[0]
		assert state.vehicle_id == 1
		assert state.velocity.magnitude == 0.0
		assert state.acceleration == 0.0

	def test_tracks_constant_velocity(self) -> None:
		est = _estimator()
		v = 5.0
		state = None
		for i in range(40):
			t = i / 10.0
			state = est.estimate([_tracked(1, v * t, 50.0)], t)[0]
		assert state is not None
		assert state.velocity.dx == pytest.approx(v, abs=0.3)
		assert state.heading.dx > 0.9  # heading east from smoothed velocity

	def test_separate_tracks_keep_independent_filters(self) -> None:
		est = _estimator()
		est.estimate([_tracked(1, 0.0, 0.0), _tracked(2, 100.0, 0.0)], 0.0)
		states = est.estimate([_tracked(1, 5.0, 0.0), _tracked(2, 95.0, 0.0)], 0.1)
		by_id = {s.vehicle_id: s for s in states}
		assert by_id[1].velocity.dx > 0.0  # moving east
		assert by_id[2].velocity.dx < 0.0  # moving west

	def test_stale_tracks_are_evicted(self) -> None:
		est = _estimator()
		est.estimate([_tracked(1, 0.0, 0.0)], 0.0)
		assert 1 in est._tracks
		# A later frame far beyond the horizon, without track 1, evicts it.
		est.estimate([_tracked(2, 0.0, 0.0)], _EVICTION_HORIZON_SECONDS + 1.0)
		assert 1 not in est._tracks
		assert 2 in est._tracks

	def test_rejects_non_positive_scale(self) -> None:
		with pytest.raises(ValueError, match="meters_per_pixel"):
			KalmanOrientationEstimator(meters_per_pixel=0.0, pos_noise=1.0, jerk=10.0)
