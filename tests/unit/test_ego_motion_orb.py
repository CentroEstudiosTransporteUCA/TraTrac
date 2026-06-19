"""Sanity tests for OrbEgoMotionEstimator on synthetic translated frames.

Uses cv2 only (no model downloads): a textured patch is shifted by a known
translation and the estimator must recover the inverse mapping (current frame
back into the reference frame). See vault/05_75_mvp1_9.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import BoundingBox, Point2D, Polygon, Transform2D
from tratrac.infrastructure.video.ego_motion_orb import OrbEgoMotionEstimator, _AnchorChain


def _vehicle(x: float, y: float, width: float, height: float) -> Detection:
	return Detection(
		bbox=BoundingBox(x=x, y=y, width=width, height=height),
		score=0.9,
		vehicle_class=VehicleClass.CAR,
	)


def _textured_image(seed: int = 0) -> NDArray[np.uint8]:
	# Deterministic high-texture image so ORB finds plenty of corners.
	rng = np.random.default_rng(seed)
	return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


def _shifted(image: NDArray[np.uint8], tx: float, ty: float) -> NDArray[np.uint8]:
	matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
	h, w = image.shape[:2]
	shifted: NDArray[np.uint8] = cv2.warpAffine(image, matrix, (w, h))
	return shifted


def _make_estimator(
	min_anchor_overlap: float = 0.5, exclusion_zones: ExclusionZones | None = None
) -> OrbEgoMotionEstimator:
	return OrbEgoMotionEstimator(
		n_features=2000,
		match_ratio=0.75,
		min_matches=10,
		ransac_threshold=3.0,
		min_anchor_overlap=min_anchor_overlap,
		exclusion_zones=exclusion_zones,
	)


def _full_frame_zone() -> ExclusionZones:
	return ExclusionZones(
		zones=(
			Polygon(
				vertices=(
					Point2D(0.0, 0.0),
					Point2D(200.0, 0.0),
					Point2D(200.0, 200.0),
					Point2D(0.0, 200.0),
				)
			),
		)
	)


class TestOrbEgoMotionEstimator:
	def test_first_frame_is_identity(self) -> None:
		estimator = _make_estimator()
		transform = estimator.estimate(Frame(index=0, pixels=_textured_image()))
		assert transform == Transform2D.identity()

	def test_recovers_inverse_of_a_known_shift(self) -> None:
		base = _textured_image()
		tx, ty = 7.0, -4.0
		estimator = _make_estimator()

		estimator.estimate(Frame(index=0, pixels=base))
		transform = estimator.estimate(Frame(index=1, pixels=_shifted(base, tx, ty)))

		# Content shifted by (tx, ty); mapping the current frame back to the
		# reference must undo it, i.e. translate by (-tx, -ty) with ~unit scale.
		assert transform.tx == pytest.approx(-tx, abs=1.0)
		assert transform.ty == pytest.approx(-ty, abs=1.0)
		assert transform.a == pytest.approx(1.0, abs=0.05)
		assert transform.d == pytest.approx(1.0, abs=0.05)

	def test_too_few_matches_carries_transform_forward(self) -> None:
		# A blank frame yields no usable features; the estimate must not crash and
		# should return the carried-forward (identity) transform.
		estimator = _make_estimator()
		blank = np.zeros((200, 200, 3), dtype=np.uint8)
		first = estimator.estimate(Frame(index=0, pixels=blank))
		second = estimator.estimate(Frame(index=1, pixels=blank))

		assert first == Transform2D.identity()
		assert second == Transform2D.identity()

	def test_full_frame_vehicle_mask_suppresses_all_features(self) -> None:
		# A detection covering the whole frame masks every pixel from ORB, so the
		# next frame yields no features to match -> no step fit -> carry forward.
		# Proves the observed detections actually reach detectAndCompute's mask.
		base = _textured_image()
		estimator = _make_estimator()
		estimator.estimate(Frame(index=0, pixels=base))
		estimator.observe([_vehicle(0.0, 0.0, 200.0, 200.0)])

		transform = estimator.estimate(Frame(index=1, pixels=_shifted(base, 7.0, -4.0)))

		assert transform == Transform2D.identity()

	def test_full_frame_exclusion_zone_suppresses_all_features(self) -> None:
		# A static exclusion zone covering the whole frame masks every pixel from ORB
		# even with NO detections observed, so the next frame yields no features to
		# match -> carry forward. Proves the static polygon mask reaches the extractor.
		base = _textured_image()
		estimator = _make_estimator(exclusion_zones=_full_frame_zone())
		estimator.estimate(Frame(index=0, pixels=base))

		transform = estimator.estimate(Frame(index=1, pixels=_shifted(base, 7.0, -4.0)))

		assert transform == Transform2D.identity()

	def test_mask_over_empty_region_still_recovers_shift(self) -> None:
		# A small masked corner leaves ample background features elsewhere, so the
		# known shift is still recovered — masking is targeted, not all-or-nothing.
		base = _textured_image()
		tx, ty = 7.0, -4.0
		estimator = _make_estimator()
		estimator.estimate(Frame(index=0, pixels=base))
		estimator.observe([_vehicle(0.0, 0.0, 20.0, 20.0)])

		transform = estimator.estimate(Frame(index=1, pixels=_shifted(base, tx, ty)))

		assert transform.tx == pytest.approx(-tx, abs=1.0)
		assert transform.ty == pytest.approx(-ty, abs=1.0)

	def test_current_transform_tracks_last_estimate(self) -> None:
		base = _textured_image()
		estimator = _make_estimator()
		assert estimator.current_transform == Transform2D.identity()
		estimator.estimate(Frame(index=0, pixels=base))
		returned = estimator.estimate(Frame(index=1, pixels=_shifted(base, 6.0, 3.0)))
		assert estimator.current_transform == returned

	def test_rejects_out_of_range_overlap(self) -> None:
		with pytest.raises(ValueError, match="min_anchor_overlap"):
			_make_estimator(min_anchor_overlap=0.0)
		with pytest.raises(ValueError, match="min_anchor_overlap"):
			_make_estimator(min_anchor_overlap=1.0)


class TestAnchorChain:
	"""Pure anchor/re-anchor/chaining logic, independent of ORB."""

	def _translation(self, tx: float, ty: float) -> Transform2D:
		return Transform2D(a=1.0, b=0.0, tx=tx, c=0.0, d=1.0, ty=ty)

	def test_high_overlap_does_not_reanchor_and_composes_onto_anchor(self) -> None:
		chain = _AnchorChain()
		# Tiny shift in a 100x100 frame: ~99% overlap, well above 0.5.
		g, reanchor = chain.advance(self._translation(1.0, 0.0), 100, 100, 0.5)
		assert reanchor is False
		assert (g.tx, g.ty) == (1.0, 0.0)
		# Anchor pose stays at identity (still anchored to frame 0).
		assert chain.global_pose == Transform2D.identity()

	def test_low_overlap_reanchors_and_adopts_global_pose(self) -> None:
		chain = _AnchorChain()
		# 70px shift in a 100px frame -> 30% overlap, below 0.5.
		g, reanchor = chain.advance(self._translation(70.0, 0.0), 100, 100, 0.5)
		assert reanchor is True
		assert g.tx == pytest.approx(70.0)
		# The new anchor's global pose is the current frame's global pose.
		assert chain.global_pose.tx == pytest.approx(70.0)

	def test_chain_is_continuous_across_a_reanchor(self) -> None:
		chain = _AnchorChain()
		# Drift past the threshold -> re-anchor; global pose now at 70.
		chain.advance(self._translation(70.0, 0.0), 100, 100, 0.5)
		# Next frame is measured against the NEW anchor (a further 10px), and its
		# global pose must continue from 70, not jump back to 10.
		g, reanchor = chain.advance(self._translation(10.0, 0.0), 100, 100, 0.5)
		assert reanchor is False
		assert g.tx == pytest.approx(80.0)

	def test_failed_fit_holds_pose_and_requests_reanchor(self) -> None:
		chain = _AnchorChain()
		chain.advance(self._translation(5.0, 0.0), 100, 100, 0.5)  # global -> identity still
		g, reanchor = chain.advance(None, 100, 100, 0.5)
		assert reanchor is True
		assert g == chain.global_pose  # held the anchor's pose
