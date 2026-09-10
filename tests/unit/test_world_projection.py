"""Tests for the pure ``WorldProjector`` impls + the local-scale helper (MVP2, src/tratrac/application/WORLD_PROJECTION.md).

numpy-only — no cv2 / model downloads. The homography matrices are built directly here
(the cv2 fit lives in ``test_world_calibration.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tratrac.application.world_projection import (
	IdentityWorldProjector,
	MultiHomographyWorldProjector,
	PerAnchorWorldProjector,
	local_scale_at,
)
from tratrac.domain.geometry import Point2D


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


def _scale_homography(s: float) -> np.ndarray:
	"""A pure-scaling (s metres per pixel) homography: world = s * image."""
	return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class TestIdentityWorldProjector:
	def test_passes_the_point_through_unchanged(self) -> None:
		projector = IdentityWorldProjector()
		assert projector.to_world(Point2D(3.0, 4.0), frame_index=11) == Point2D(3.0, 4.0)


class TestPerAnchorWorldProjectorSingleAnchor:
	"""A single-entry map is the single-homography (bounded/static scene) case — see
	``PerAnchorWorldProjector``'s docstring for why there's no separate class for it."""

	def test_identity_matrix_is_a_no_op(self) -> None:
		projector = PerAnchorWorldProjector({0: np.eye(3, dtype=np.float64)})
		assert projector.to_world(Point2D(7.0, 9.0), frame_index=0) == Point2D(7.0, 9.0)

	def test_pure_scale_multiplies_both_axes(self) -> None:
		projector = PerAnchorWorldProjector({0: _scale_homography(0.5)})
		out = projector.to_world(Point2D(10.0, 20.0), frame_index=0)
		assert out == Point2D(5.0, 10.0)

	def test_perspective_divide_is_applied(self) -> None:
		# Last row (0, 0.1, 1): w = 0.1*y + 1 -> a real projective divide, not affine.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.1, 1.0]], dtype=np.float64)
		projector = PerAnchorWorldProjector({0: matrix})
		out = projector.to_world(Point2D(2.0, 10.0), frame_index=0)
		# w = 0.1*10 + 1 = 2 -> (2/2, 10/2)
		assert out.x == pytest.approx(1.0)
		assert out.y == pytest.approx(5.0)

	def test_frame_index_is_ignored_with_only_one_anchor(self) -> None:
		projector = PerAnchorWorldProjector({0: _scale_homography(2.0)})
		assert projector.to_world(Point2D(1.0, 1.0), 0) == projector.to_world(
			Point2D(1.0, 1.0), 999
		)

	def test_point_at_infinity_raises(self) -> None:
		# Last row (0, 1, -5): w = y - 5 = 0 at y = 5.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -5.0]], dtype=np.float64)
		projector = PerAnchorWorldProjector({0: matrix})
		with pytest.raises(ValueError, match="infinity"):
			projector.to_world(Point2D(3.0, 5.0), frame_index=0)

	def test_inverse_round_trips_through_a_general_homography(self) -> None:
		# Not a pure scale/affine -- includes a real perspective term, so a correct inverse
		# has to undo the actual projective divide, not just a linear map.
		matrix = np.array([[2.0, 0.3, 5.0], [0.1, 1.5, -2.0], [0.001, 0.0005, 1.0]])
		projector = PerAnchorWorldProjector({0: matrix})
		original = Point2D(37.0, -12.0)
		world = projector.to_world(original, frame_index=0)
		recovered = projector.inverse().to_world(world, frame_index=0)
		assert recovered.x == pytest.approx(original.x, abs=1e-6)
		assert recovered.y == pytest.approx(original.y, abs=1e-6)

	def test_shifted_adds_a_translation_after_projection(self) -> None:
		projector = PerAnchorWorldProjector({0: _scale_homography(2.0)}).shifted(100.0, -50.0)
		out = projector.to_world(Point2D(10.0, 10.0), frame_index=0)
		# Plain scale would give (20, 20); shifted adds (100, -50).
		assert out == Point2D(120.0, -30.0)

	def test_shifted_then_inverse_undoes_both_the_shift_and_the_homography(self) -> None:
		projector = PerAnchorWorldProjector({0: _scale_homography(0.25)}).shifted(7.0, -3.0)
		original = Point2D(40.0, 80.0)
		world = projector.to_world(original, frame_index=0)
		recovered = projector.inverse().to_world(world, frame_index=0)
		assert recovered.x == pytest.approx(original.x, abs=1e-9)
		assert recovered.y == pytest.approx(original.y, abs=1e-9)


class TestPerAnchorWorldProjector:
	def test_rejects_empty_mapping(self) -> None:
		with pytest.raises(ValueError, match="at least one anchor"):
			PerAnchorWorldProjector({})

	def test_uses_the_exact_anchors_homography(self) -> None:
		projector = PerAnchorWorldProjector(
			{0: _scale_homography(1.0), 100: _scale_homography(10.0)}
		)
		assert projector.to_world(Point2D(2.0, 2.0), frame_index=0) == Point2D(2.0, 2.0)
		assert projector.to_world(Point2D(2.0, 2.0), frame_index=100) == Point2D(20.0, 20.0)

	def test_picks_the_nearer_anchor_for_an_in_between_frame(self) -> None:
		projector = PerAnchorWorldProjector(
			{0: _scale_homography(1.0), 100: _scale_homography(10.0)}
		)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=30) == Point2D(1.0, 1.0)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=70) == Point2D(10.0, 10.0)

	def test_ties_break_toward_the_earlier_anchor(self) -> None:
		projector = PerAnchorWorldProjector(
			{0: _scale_homography(1.0), 100: _scale_homography(10.0)}
		)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=50) == Point2D(1.0, 1.0)

	def test_clamps_outside_the_anchor_range(self) -> None:
		projector = PerAnchorWorldProjector({50: _scale_homography(2.0)})
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=-1000) == Point2D(2.0, 2.0)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=1000) == Point2D(2.0, 2.0)

	def test_inverse_uses_the_same_anchor_selection_as_the_forward_projector(self) -> None:
		projector = PerAnchorWorldProjector(
			{0: _scale_homography(1.0), 100: _scale_homography(10.0)}
		)
		original = Point2D(3.0, 4.0)
		# frame_index=70 -> nearer to anchor 100 on the way forward.
		world = projector.to_world(original, frame_index=70)
		recovered = projector.inverse().to_world(world, frame_index=70)
		assert recovered.x == pytest.approx(original.x)
		assert recovered.y == pytest.approx(original.y)

	def test_shifted_translates_every_anchors_output(self) -> None:
		projector = PerAnchorWorldProjector(
			{0: _scale_homography(1.0), 100: _scale_homography(10.0)}
		).shifted(5.0, 5.0)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=0) == Point2D(6.0, 6.0)
		assert projector.to_world(Point2D(1.0, 1.0), frame_index=100) == Point2D(15.0, 15.0)


class TestMultiHomographyWorldProjector:
	_PLANES = (
		(0, _square(0.0, 0.0, 100.0, 100.0)),  # ground
		(1, _square(200.0, 200.0, 300.0, 300.0)),  # bridge
	)

	def test_rejects_empty_mapping(self) -> None:
		with pytest.raises(ValueError, match="at least one plane"):
			MultiHomographyWorldProjector({}, self._PLANES)

	def test_uses_the_matching_planes_homography(self) -> None:
		projector = MultiHomographyWorldProjector(
			{0: _scale_homography(1.0), 1: _scale_homography(10.0)}, self._PLANES
		)
		assert projector.to_world(Point2D(50.0, 50.0), frame_index=0) == Point2D(50.0, 50.0)
		assert projector.to_world(Point2D(250.0, 250.0), frame_index=0) == Point2D(2500.0, 2500.0)

	def test_frame_index_is_ignored(self) -> None:
		projector = MultiHomographyWorldProjector({0: _scale_homography(2.0)}, self._PLANES)
		assert projector.to_world(Point2D(1.0, 1.0), 0) == projector.to_world(
			Point2D(1.0, 1.0), 999
		)

	def test_unfitted_plane_raises_clearly(self) -> None:
		# Plane 1 (the bridge) classifies fine but has no fitted homography.
		projector = MultiHomographyWorldProjector({0: _scale_homography(1.0)}, self._PLANES)
		with pytest.raises(ValueError, match="plane 1"):
			projector.to_world(Point2D(250.0, 250.0), frame_index=0)

	def test_unclassified_point_needs_a_plane_zero_homography(self) -> None:
		projector = MultiHomographyWorldProjector({0: _scale_homography(1.0)}, self._PLANES)
		# Outside every explicit zone -> plane_id_for_point defaults to 0.
		assert projector.to_world(Point2D(-50.0, -50.0), frame_index=0) == Point2D(-50.0, -50.0)


class TestLocalScaleAt:
	def test_uniform_scale_recovers_the_scale_factor(self) -> None:
		projector = PerAnchorWorldProjector({0: _scale_homography(0.25)})
		assert local_scale_at(projector, Point2D(100.0, 100.0)) == pytest.approx(0.25)

	def test_identity_projector_has_unit_scale(self) -> None:
		assert local_scale_at(IdentityWorldProjector(), Point2D(50.0, 50.0)) == pytest.approx(1.0)

	def test_scale_is_local_under_perspective(self) -> None:
		# Foreshortening grows with y; the metres-per-pixel must differ between two rows.
		matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.01, 1.0]], dtype=np.float64)
		projector = PerAnchorWorldProjector({0: matrix})
		near = local_scale_at(projector, Point2D(0.0, 0.0))
		far = local_scale_at(projector, Point2D(0.0, 80.0))
		assert not math.isclose(near, far)
