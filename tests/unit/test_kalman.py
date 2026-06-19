"""Tests for the constant-acceleration Kalman / RTS smoothing core."""

from __future__ import annotations

import numpy as np
import pytest

from tratrac.application.kalman import KinematicKalmanFilter, smooth_track


def _times(n: int, fps: float = 10.0) -> list[float]:
	return [i / fps for i in range(n)]


class TestSmoothTrack:
	def test_recovers_constant_acceleration_motion(self) -> None:
		# Noiseless CA trajectory: p = p0 + v0 t + 0.5 a t². The smoother must recover
		# position, velocity and (constant) acceleration across the whole track.
		t = np.array(_times(40))
		ax_true, vx0, px0 = 2.0, 1.0, 5.0
		xs = px0 + vx0 * t + 0.5 * ax_true * t**2
		ys = np.zeros_like(t)
		out = smooth_track(xs.tolist(), ys.tolist(), t.tolist(), pos_noise=0.5, jerk=1e-3)
		# Check a settled interior point (filter has learned the dynamics).
		mid = out[25]
		assert mid.px == pytest.approx(xs[25], abs=0.1)
		assert mid.vx == pytest.approx(vx0 + ax_true * t[25], abs=0.2)
		assert mid.ax == pytest.approx(ax_true, abs=0.2)
		assert mid.ay == pytest.approx(0.0, abs=0.2)

	def test_smoothed_position_beats_raw_on_noisy_track(self) -> None:
		rng = np.random.default_rng(0)
		t = np.array(_times(120))
		true_x = 3.0 * t  # constant velocity
		true_y = np.zeros_like(t)
		noise = rng.normal(0.0, 1.0, size=t.shape)
		meas_x = true_x + noise
		out = smooth_track(meas_x.tolist(), true_y.tolist(), t.tolist(), pos_noise=1.0, jerk=1e-2)
		smoothed_x = np.array([s.px for s in out])
		raw_rmse = float(np.sqrt(np.mean((meas_x - true_x) ** 2)))
		smooth_rmse = float(np.sqrt(np.mean((smoothed_x - true_x) ** 2)))
		assert smooth_rmse < raw_rmse  # de-jittering reduces position error

	def test_suppresses_phantom_acceleration_on_constant_velocity(self) -> None:
		# True acceleration is zero; finite-differencing noisy position would explode it.
		rng = np.random.default_rng(1)
		t = np.array(_times(120))
		meas_x = 3.0 * t + rng.normal(0.0, 1.0, size=t.shape)
		out = smooth_track(meas_x.tolist(), [0.0] * len(t), t.tolist(), pos_noise=1.0, jerk=1e-3)
		accel = np.array([s.ax for s in out[10:-10]])  # ignore edges
		# Finite-difference accel on this noise would be ~ O(100 m/s²); smoothed stays small.
		assert np.max(np.abs(accel)) < 5.0

	def test_zero_phase_no_lag_at_track_start(self) -> None:
		# A causal forward filter lags early samples; RTS uses future data, so even the
		# first interior points track a noiseless ramp closely.
		t = np.array(_times(30))
		xs = 4.0 * t
		out = smooth_track(xs.tolist(), [0.0] * len(t), t.tolist(), pos_noise=0.5, jerk=1e-3)
		early = out[3]
		assert early.px == pytest.approx(xs[3], abs=0.15)
		assert early.vx == pytest.approx(4.0, abs=0.3)

	def test_empty_track_returns_empty(self) -> None:
		assert smooth_track([], [], [], pos_noise=1.0, jerk=1e-3) == []

	def test_single_sample_track(self) -> None:
		out = smooth_track([7.0], [2.0], [0.0], pos_noise=1.0, jerk=1e-3)
		assert len(out) == 1
		assert out[0].px == pytest.approx(7.0, abs=1e-9)

	def test_rejects_mismatched_lengths(self) -> None:
		with pytest.raises(ValueError, match="equal length"):
			smooth_track([1.0, 2.0], [1.0], [0.0, 0.1], pos_noise=1.0, jerk=1e-3)

	def test_rejects_non_positive_params(self) -> None:
		with pytest.raises(ValueError, match="pos_noise"):
			smooth_track([1.0], [1.0], [0.0], pos_noise=0.0, jerk=1e-3)
		with pytest.raises(ValueError, match="jerk"):
			smooth_track([1.0], [1.0], [0.0], pos_noise=1.0, jerk=0.0)


class TestKinematicKalmanFilter:
	def test_first_observation_returns_measurement(self) -> None:
		kf = KinematicKalmanFilter(pos_noise=1.0, jerk=1e-2)
		s = kf.observe(10.0, 20.0, dt=0.0)
		assert s.px == pytest.approx(10.0)
		assert s.py == pytest.approx(20.0)
		assert (s.vx, s.vy, s.ax, s.ay) == (0.0, 0.0, 0.0, 0.0)

	def test_tracks_constant_velocity(self) -> None:
		kf = KinematicKalmanFilter(pos_noise=0.5, jerk=1e-3)
		v = 2.0
		last = None
		for i in range(40):
			last = kf.observe(v * (i / 10.0), 0.0, dt=0.0 if i == 0 else 0.1)
		assert last is not None
		assert last.vx == pytest.approx(v, abs=0.2)

	def test_requires_positive_dt_after_first(self) -> None:
		kf = KinematicKalmanFilter(pos_noise=1.0, jerk=1e-2)
		kf.observe(0.0, 0.0, dt=0.0)
		with pytest.raises(ValueError, match="dt must be positive"):
			kf.observe(1.0, 0.0, dt=0.0)
