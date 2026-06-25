"""Tests for the pure ``WorldProjector`` impls + the local-scale helper (MVP2, vault/06).

numpy-only — no cv2 / model downloads. The homography matrices are built directly here
(the cv2 fit lives in ``test_world_calibration.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tratrac.application.world_projection import (
	IdentityWorldProjector,
	SingleHomographyProjector,
	local_scale_at,
)
from tratrac.domain.geometry import Point2D


def _scale_homography(s: float) -> np.ndarray:
	"""A pure-scaling (s metres per pixel) homography: world = s * image."""
	return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class TestIdentityWorldProjector:
	def test_passes_the_point_through_unchanged(self) -> None:
		projector = IdentityWorldProjector()
		assert projector.to_world(Point2D(3.0, 4.0), frame_index=11) == Point2D(3.0, 4.0)


class TestSingleHomographyProjector:
	def test_identity_matrix_is_a_no_op(self) -> None:
		projector = SingleHomographyProjector(np.eye(3, dtype=np.float64))
		assert projector.to_world(Point2D(7.0, 9.0), frame_index=0) == Point2D(7.0, 9.0)

	def test_pure_scale_multiplies_both_axes(self) -> None:
		projector = SingleHomographyProjector(_scale_homography(0.5))
		out = projector.to_world(Point2D(10.0, 20.0), frame_index=0)
		assert out == Point2D(5.0, 10.0)

	def test_perspective_divide_is_applied(self) -> None:
		# Last row (0, 0.1, 1): w = 0.1*y + 1 -> a real projective divide, not affine.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.1, 1.0]], dtype=np.float64)
		projector = SingleHomographyProjector(matrix)
		out = projector.to_world(Point2D(2.0, 10.0), frame_index=0)
		# w = 0.1*10 + 1 = 2 -> (2/2, 10/2)
		assert out.x == pytest.approx(1.0)
		assert out.y == pytest.approx(5.0)

	def test_frame_index_is_ignored(self) -> None:
		projector = SingleHomographyProjector(_scale_homography(2.0))
		assert projector.to_world(Point2D(1.0, 1.0), 0) == projector.to_world(
			Point2D(1.0, 1.0), 999
		)

	def test_point_at_infinity_raises(self) -> None:
		# Last row (0, 1, -5): w = y - 5 = 0 at y = 5.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -5.0]], dtype=np.float64)
		projector = SingleHomographyProjector(matrix)
		with pytest.raises(ValueError, match="infinity"):
			projector.to_world(Point2D(3.0, 5.0), frame_index=0)


class TestLocalScaleAt:
	def test_uniform_scale_recovers_the_scale_factor(self) -> None:
		projector = SingleHomographyProjector(_scale_homography(0.25))
		assert local_scale_at(projector, Point2D(100.0, 100.0)) == pytest.approx(0.25)

	def test_identity_projector_has_unit_scale(self) -> None:
		assert local_scale_at(IdentityWorldProjector(), Point2D(50.0, 50.0)) == pytest.approx(1.0)

	def test_scale_is_local_under_perspective(self) -> None:
		# Foreshortening grows with y; the metres-per-pixel must differ between two rows.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.01, 1.0]], dtype=np.float64)
		projector = SingleHomographyProjector(matrix)
		near = local_scale_at(projector, Point2D(0.0, 0.0))
		far = local_scale_at(projector, Point2D(0.0, 80.0))
		assert not math.isclose(near, far)
