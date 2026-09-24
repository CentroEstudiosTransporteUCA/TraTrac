"""Keyframe-anchored ORB EgoMotionEstimator adapter (MVP1.9, see src/tratrac/infrastructure/video/EGO_MOTION.md).

Estimates camera ego-motion with ORB features, a ratio-tested brute-force Hamming
matcher, and a RANSAC 4-DOF similarity fit (``cv2.estimateAffinePartial2D``).

Unlike a frame-to-frame chain, this matches each frame against a **keyframe anchor**
and composes anchor poses into a single continuous global frame. The anchor is
re-set whenever the current frame no longer shares enough area with it
(``min_anchor_overlap``). Two reasons:

* **Robust matching late in a clip.** Matching a recent, high-overlap anchor keeps
  ORB correspondences plentiful instead of trying to match a far-away first frame.
* **Bounded per-step error.** Within an anchor's life every frame is matched
  independently against it, so per-frame error does not compound; drift accrues
  only at the (sparse) re-anchor compositions.

The returned transform maps the current frame's pixels into the global frame. The
pipeline applies it to *detections* (not pixels), so detection/tracking run on the
raw, full-resolution frame and nothing is cropped — see src/tratrac/infrastructure/video/EGO_MOTION.md.

Feature-based (not intensity ECC) because aerial traffic is dominated by moving
foreground: explicit correspondences let RANSAC reject moving-vehicle matches. The
estimator masks vehicles out of feature extraction; where the mask comes from is
pluggable behind ``MaskSource`` (below) — today's only implementation is
operator-authored background zones (``BackgroundZoneMaskSource``, no detector
needed, used by `tratrac-preprocess` — see "Detector-free ego-motion" in
src/tratrac/infrastructure/video/EGO_MOTION.md); `tratrac` itself never
constructs this estimator live, only `tratrac-preprocess` does. SuperPoint+LightGlue
is the eventual estimator upgrade (Group B3, GitHub Issues).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from tratrac.domain.background import BackgroundZone, BackgroundZones
from tratrac.domain.detection import Detection
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D, clipped_overlap_fraction

# Notified when a frame becomes a keyframe anchor: (frame_index, global pose). The
# scout subscribes to enumerate the reference frames an operator draws zones on.
AnchorObserver = Callable[[int, Transform2D], None]


class MaskSource(Protocol):
	"""Supplies the per-frame ORB feature mask (255 = usable, 0 = ignore).

	``observe`` receives each frame's detections (a no-op for a source that doesn't
	need them, e.g. ``BackgroundZoneMaskSource``) so every source satisfies the same
	shape regardless of whether it's detection-fed or operator-authored.
	"""

	def observe(self, detections: list[Detection]) -> None: ...
	def mask_for(self, frame_index: int, height: int, width: int) -> NDArray[np.uint8] | None: ...


class BackgroundZoneMaskSource:
	"""Masks ORB feature extraction to operator-authored background zones — no
	detector, no live detections needed (see `tratrac-preprocess`).

	``zones`` is a `BackgroundZones` ordered by ``reference_frame``; each entry marks
	"use this polygon as the keep-region from this frame until a later entry
	supersedes it" — not tied to ORB's own re-anchor points. ``mask_for`` resolves
	the most recent entry at or before ``frame_index`` (falling back to the earliest
	zone for a frame before the first entry, rather than masking nothing).
	"""

	def __init__(self, zones: BackgroundZones) -> None:
		self._zones = sorted(zones.zones, key=lambda zone: zone.reference_frame)

	def observe(self, detections: list[Detection]) -> None:
		del detections  # the mask is entirely operator-authored; no detections needed

	def mask_for(self, frame_index: int, height: int, width: int) -> NDArray[np.uint8] | None:
		zone = self._zone_at(frame_index)
		mask: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
		points = np.array([[v.x, v.y] for v in zone.polygon.vertices], dtype=np.int32)
		cv2.fillPoly(mask, [points], 255)
		return mask

	def _zone_at(self, frame_index: int) -> BackgroundZone:
		candidate = self._zones[0]
		for zone in self._zones:
			if zone.reference_frame > frame_index:
				break
			candidate = zone
		return candidate


class _AnchorChain:
	"""Tracks the anchor's global pose and decides when to re-anchor. Pure.

	Holds ``global_pose`` = the transform mapping the *current anchor's* frame into
	the global frame. ``advance`` composes the just-measured local transform onto it
	to get the current frame's global pose, and reports whether the caller should
	promote the current frame to be the new anchor.
	"""

	def __init__(self) -> None:
		self._global = Transform2D.identity()

	@property
	def global_pose(self) -> Transform2D:
		return self._global

	def advance(
		self, local: Transform2D | None, width: int, height: int, min_overlap: float
	) -> tuple[Transform2D, bool]:
		"""Return ``(current_frame_global_pose, should_reanchor)``.

		``local`` maps the current frame into the anchor frame, or is ``None`` when
		the fit failed. On failure we hold the anchor's pose for this frame and ask
		to re-anchor (so a stale anchor after a hard cut is not matched forever). On
		success we compose ``global ∘ local``; if the current frame's overlap with
		the anchor has dropped below ``min_overlap`` we adopt that composed pose as
		the new anchor's global pose.
		"""
		if local is None:
			return self._global, True
		current_global = self._global.compose(local)
		if clipped_overlap_fraction(local, width, height) < min_overlap:
			self._global = current_global
			return current_global, True
		return current_global, False


class OrbEgoMotionEstimator:
	"""Implements ``EgoMotionEstimator`` (and ``DetectionObserver``) with keyframe ORB.

	Stateful: keeps the anchor's masked keypoints/descriptors, the anchor chain, and
	the latest detections fed back for masking. The first usable frame becomes the
	anchor and returns identity. A frame with too few features to match holds the
	last pose and keeps the anchor; a frame that has features but cannot be fit to
	the anchor triggers a re-anchor to itself.
	"""

	def __init__(
		self,
		*,
		n_features: int,
		match_ratio: float,
		min_matches: int,
		ransac_threshold: float,
		min_anchor_overlap: float,
		mask_source: MaskSource,
		anchor_observer: AnchorObserver | None = None,
	) -> None:
		if n_features <= 0:
			raise ValueError(f"n_features must be positive, got {n_features}.")
		if not 0.0 < match_ratio < 1.0:
			raise ValueError(f"match_ratio must be in (0, 1), got {match_ratio}.")
		if min_matches < 2:
			raise ValueError(f"min_matches must be >= 2, got {min_matches}.")
		if ransac_threshold <= 0.0:
			raise ValueError(f"ransac_threshold must be positive, got {ransac_threshold}.")
		if not 0.0 < min_anchor_overlap < 1.0:
			raise ValueError(f"min_anchor_overlap must be in (0, 1), got {min_anchor_overlap}.")
		self._match_ratio = match_ratio
		self._min_matches = min_matches
		self._ransac_threshold = ransac_threshold
		self._min_anchor_overlap = min_anchor_overlap
		self._mask_source = mask_source
		self._anchor_observer = anchor_observer
		# ORB_create is a factory alias absent from opencv's bundled type stubs.
		self._orb: Any = cv2.ORB_create(nfeatures=n_features)  # type: ignore[attr-defined]
		self._matcher: Any = cv2.BFMatcher(cv2.NORM_HAMMING)
		self._anchor_keypoints: Any = None
		self._anchor_descriptors: NDArray[np.uint8] | None = None
		self._chain = _AnchorChain()
		# The current frame's global pose; what the overlay reads to map back to raw.
		self._current = Transform2D.identity()

	@property
	def current_transform(self) -> Transform2D:
		"""The last transform returned by ``estimate`` (current frame → global)."""
		return self._current

	def observe(self, detections: list[Detection]) -> None:
		self._mask_source.observe(detections)

	def estimate(self, frame: Frame) -> Transform2D:
		gray = cv2.cvtColor(frame.pixels, cv2.COLOR_BGR2GRAY)
		height, width = gray.shape[:2]
		mask = self._mask_source.mask_for(frame.index, height, width)
		keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
		has_features = descriptors is not None and len(keypoints) >= self._min_matches

		if self._anchor_descriptors is None:
			# No anchor yet: establish it from the first usable frame, identity meanwhile.
			if has_features:
				self._set_anchor(keypoints, descriptors)
				self._announce_anchor(frame.index)
			return self._current

		if not has_features:
			# Nothing to match this frame; hold the pose and keep the anchor (re-anchoring
			# to a featureless frame would only make the next match fail too).
			return self._current

		local = self._fit_against_anchor(keypoints, descriptors)
		self._current, reanchor = self._chain.advance(
			local, width, height, self._min_anchor_overlap
		)
		if reanchor:
			self._set_anchor(keypoints, descriptors)
			self._announce_anchor(frame.index)
		return self._current

	def _announce_anchor(self, frame_index: int) -> None:
		"""Notify the observer that ``frame_index`` is now a keyframe anchor.

		``self._current`` is that anchor's global pose (identity for the first anchor;
		the just-composed current-frame pose on a re-anchor)."""
		if self._anchor_observer is not None:
			self._anchor_observer(frame_index, self._current)

	def _set_anchor(self, keypoints: Any, descriptors: NDArray[np.uint8]) -> None:
		self._anchor_keypoints = keypoints
		self._anchor_descriptors = descriptors

	def _fit_against_anchor(
		self, keypoints: Any, descriptors: NDArray[np.uint8]
	) -> Transform2D | None:
		"""Fit the current→anchor similarity transform, or ``None`` if unreliable."""
		if len(keypoints) < 2 or len(self._anchor_keypoints) < 2:
			return None
		matches = self._matcher.knnMatch(descriptors, self._anchor_descriptors, k=2)
		good = [
			pair[0]
			for pair in matches
			if len(pair) == 2 and pair[0].distance < self._match_ratio * pair[1].distance
		]
		if len(good) < self._min_matches:
			return None
		src = np.array([keypoints[m.queryIdx].pt for m in good], dtype=np.float64)
		dst = np.array([self._anchor_keypoints[m.trainIdx].pt for m in good], dtype=np.float64)
		matrix, _inliers = cv2.estimateAffinePartial2D(
			src, dst, method=cv2.RANSAC, ransacReprojThreshold=self._ransac_threshold
		)
		if matrix is None:
			return None
		return Transform2D(
			a=float(matrix[0, 0]),
			b=float(matrix[0, 1]),
			tx=float(matrix[0, 2]),
			c=float(matrix[1, 0]),
			d=float(matrix[1, 1]),
			ty=float(matrix[1, 2]),
		)
