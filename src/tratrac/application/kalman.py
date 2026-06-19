"""Constant-acceleration Kalman filter + RTS smoother for trajectory de-jittering.

Detector center jitter makes the raw bbox-centroid trajectory noisy; differentiating
it for velocity/acceleration amplifies that noise (the keystone of
``vault/research_high_precision_tracking.md`` §1). The fix is to *smooth position* with
a constant-acceleration motion model and read velocity/acceleration out of the filter
state instead of finite-differencing.

This module is the shared, pure-numpy core (no ``filterpy``): a per-axis
constant-acceleration state ``[position, velocity, acceleration]`` with a white-noise
*jerk* process model and a variable time step (so it survives ``input.process_fps``
decimation). Two consumers:

* ``smooth_track`` — forward Kalman pass + Rauch-Tung-Striebel backward pass over a
  whole track. Offline, **zero-phase** (no lag). The ``tratrac-smooth`` post-pass uses
  it (see vault/22_smoothing.md).
* ``KinematicKalmanFilter`` — the stateful forward-only filter for streaming use.

x and y are filtered independently (a constant-acceleration model has no cross-axis
coupling), so everything is built from a 1-D filter run on each axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[np.float64]

# Initial variance for the unobserved velocity/acceleration states: large, so the
# filter learns them from the data rather than from a confident wrong prior.
_INITIAL_RATE_VARIANCE = 1.0e6


def _transition(dt: float) -> _Array:
	"""Constant-acceleration state-transition matrix for a step of ``dt`` seconds."""
	return np.array(
		[
			[1.0, dt, 0.5 * dt * dt],
			[0.0, 1.0, dt],
			[0.0, 0.0, 1.0],
		]
	)


def _process_noise(dt: float, jerk: float) -> _Array:
	"""Discrete white-noise-jerk process covariance (spectral density ``jerk``)."""
	dt2 = dt * dt
	dt3 = dt2 * dt
	dt4 = dt3 * dt
	dt5 = dt4 * dt
	return jerk * np.array(
		[
			[dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
			[dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
			[dt3 / 6.0, dt2 / 2.0, dt],
		]
	)


_H = np.array([[1.0, 0.0, 0.0]])  # measurement picks out position
_I3 = np.eye(3)


def _predict(x: _Array, p: _Array, dt: float, jerk: float) -> tuple[_Array, _Array]:
	f = _transition(dt)
	return f @ x, f @ p @ f.T + _process_noise(dt, jerk)


def _update(x: _Array, p: _Array, z: float, r: float) -> tuple[_Array, _Array]:
	innovation = z - (_H @ x)[0]
	s = (_H @ p @ _H.T)[0, 0] + r
	gain = (p @ _H.T) / s  # (3, 1)
	x_new = x + gain[:, 0] * innovation
	p_new = (_I3 - gain @ _H) @ p
	return x_new, p_new


def _smooth_axis(measurements: _Array, dts: _Array, r: float, jerk: float) -> _Array:
	"""Forward CA Kalman + RTS over a 1-D sequence; return an ``(n, 3)`` smoothed state."""
	n = len(measurements)
	x_post = np.zeros((n, 3))
	p_post = np.zeros((n, 3, 3))
	x_prior = np.zeros((n, 3))
	p_prior = np.zeros((n, 3, 3))

	x = np.array([measurements[0], 0.0, 0.0])
	p = np.diag([r, _INITIAL_RATE_VARIANCE, _INITIAL_RATE_VARIANCE])
	for k in range(n):
		if k > 0:
			x, p = _predict(x, p, dts[k], jerk)
		x_prior[k], p_prior[k] = x, p
		x, p = _update(x, p, measurements[k], r)
		x_post[k], p_post[k] = x, p

	# Rauch-Tung-Striebel backward recursion: fold future measurements into each step.
	x_smooth = x_post.copy()
	for k in range(n - 2, -1, -1):
		f = _transition(dts[k + 1])
		c = p_post[k] @ f.T @ np.linalg.inv(p_prior[k + 1])
		x_smooth[k] = x_post[k] + c @ (x_smooth[k + 1] - x_prior[k + 1])
	return x_smooth


@dataclass(frozen=True, slots=True)
class SmoothedSample:
	"""Smoothed kinematics at one timestep: position, velocity, acceleration per axis."""

	px: float
	py: float
	vx: float
	vy: float
	ax: float
	ay: float


def smooth_track(
	xs: Sequence[float],
	ys: Sequence[float],
	timestamps: Sequence[float],
	*,
	pos_noise: float,
	jerk: float,
) -> list[SmoothedSample]:
	"""Forward+RTS smooth one track's position series into smoothed kinematics.

	``pos_noise`` is the measurement-noise std (same units as ``xs``/``ys`` — pixels);
	``jerk`` is the process spectral density (larger = more responsive, less smooth).
	Time steps are taken from ``timestamps`` (variable dt supported). Zero-phase: the
	estimate at every step uses both past and future measurements.
	"""
	n = len(xs)
	if not (n == len(ys) == len(timestamps)):
		raise ValueError("xs, ys and timestamps must have equal length.")
	if pos_noise <= 0.0:
		raise ValueError(f"pos_noise must be positive, got {pos_noise}.")
	if jerk <= 0.0:
		raise ValueError(f"jerk must be positive, got {jerk}.")
	if n == 0:
		return []
	t = np.asarray(timestamps, dtype=np.float64)
	dts = np.zeros(n)
	dts[1:] = np.diff(t)
	r = pos_noise * pos_noise
	sx = _smooth_axis(np.asarray(xs, dtype=np.float64), dts, r, jerk)
	sy = _smooth_axis(np.asarray(ys, dtype=np.float64), dts, r, jerk)
	return [
		SmoothedSample(sx[k, 0], sy[k, 0], sx[k, 1], sy[k, 1], sx[k, 2], sy[k, 2]) for k in range(n)
	]


class KinematicKalmanFilter:
	"""Stateful forward-only constant-acceleration filter (per-axis), for streaming use.

	Feed observations in stream order with the elapsed ``dt`` since the previous one;
	each call predicts then updates and returns the current smoothed estimate. Causal,
	so it has settling lag — the offline ``smooth_track`` (RTS) is zero-phase and
	preferred when the whole track is available.
	"""

	def __init__(self, *, pos_noise: float, jerk: float) -> None:
		if pos_noise <= 0.0:
			raise ValueError(f"pos_noise must be positive, got {pos_noise}.")
		if jerk <= 0.0:
			raise ValueError(f"jerk must be positive, got {jerk}.")
		self._r = pos_noise * pos_noise
		self._jerk = jerk
		self._x: list[_Array] | None = None  # [axis_x_state, axis_y_state]
		self._p: list[_Array] | None = None

	def observe(self, x_meas: float, y_meas: float, dt: float) -> SmoothedSample:
		"""Predict by ``dt`` then update with ``(x_meas, y_meas)``; return the estimate.

		The first call initialises from the measurement (``dt`` ignored). Later calls
		require ``dt > 0``."""
		if self._x is None or self._p is None:
			self._x = [
				np.array([x_meas, 0.0, 0.0]),
				np.array([y_meas, 0.0, 0.0]),
			]
			self._p = [
				np.diag([self._r, _INITIAL_RATE_VARIANCE, _INITIAL_RATE_VARIANCE]),
				np.diag([self._r, _INITIAL_RATE_VARIANCE, _INITIAL_RATE_VARIANCE]),
			]
			return self._sample()
		if dt <= 0.0:
			raise ValueError(f"dt must be positive after the first observation, got {dt}.")
		for axis, z in ((0, x_meas), (1, y_meas)):
			x, p = _predict(self._x[axis], self._p[axis], dt, self._jerk)
			self._x[axis], self._p[axis] = _update(x, p, z, self._r)
		return self._sample()

	def _sample(self) -> SmoothedSample:
		assert self._x is not None
		x, y = self._x
		return SmoothedSample(x[0], y[0], x[1], y[1], x[2], y[2])
