"""Keyframe-anchored ORB EgoMotionEstimator adapter (MVP1.9, intermediate).

Estimates camera ego-motion with ORB features, a ratio-tested brute-force Hamming
matcher, and a RANSAC 4-DOF similarity fit (``cv2.estimateAffinePartial2D``). Always
attempted by ``tratrac-preprocess estimate``, a no-op when a clip has no camera motion
to correct — not a config toggle, see "Config surface" below. An intermediate
"keep if good enough" shortcut before MVP2's learned stabilizer + world projection;
the MVP number is a capability ID, not execution order.

What this is: a basic-but-good-enough ego-motion compensation step that removes drone
motion from exported trajectories, slotting between MVP1.75 (metric sizes/speeds from
GSD) and MVP2 (world projection). Explicitly an intermediate, measure-then-keep
deliverable: ship the cheap feature-based estimator behind the ``EgoMotionEstimator``
port, quantify how much it improves trajectories on real footage, and keep it only if
the gain justifies the cost. If not, the *port* survives as the seam MVP2 plugs the
learned stabilizer into (the B3 SuperPoint+LightGlue issue in GitHub Issues). This is
not world projection: coordinates remain image-space and therefore still non-physical
per the coordinate-semantics invariant (``src/tratrac/domain/ARCHITECTURE.md``). What
changes is that they are image-space in a single stabilized (keyframe-chained global)
frame instead of per-frame raw pixels, so frame-to-frame displacement — and therefore
the speed/acceleration the smoother derives — no longer contains the camera's own
motion.

Design decision: stabilize coordinates, not pixels. Ego-motion is removed by
transforming *detection coordinates*, not by warping the image — the detector and
tracker run on the raw, full-resolution frame, and the estimated transform is applied
to each detection's bounding box *before* tracking, mapping it into the stabilized
frame. The rejected alternative (a ``VideoSource`` decorator warping each frame into a
fixed reference before the pipeline sees it) crops everything that drifts off frame 0
to black, starving the detector late in a panning/zooming clip, and also blurs/rescales
the cars. An earlier slice tried exactly that (``StabilizedVideoSource``, since
removed): on a real moving+zooming clip the cumulative transform drifted ~72% of frame
width and the warped content slid out of the fixed canvas — by the end the frame was
mostly black and the detector found almost no cars. The flaw is intrinsic to warping
into a bounded image: a picture has edges, so anything that no longer fits is
discarded. A *point* has no edges — mapping a detection at (1800, 1000) to (2400, -150)
in the global frame is a fine value. So this transforms points, not pixels — a
poor-man's version of what MVP2 does (project coordinates to a stable frame), using the
similarity transform already estimated here. Doing it at the *detection* level, before
tracking, makes the tracker ego-motion-free too.

The transform model: a 4-DOF similarity (translation + rotation + uniform scale). Right
model for near-nadir aerial — full affine (6-DOF) absorbs residual vehicle motion into
a skew; full perspective homography (8-DOF) is deliberately MVP2's job.
``domain/geometry.py``'s ``Transform2D`` is the pure value object: the six affine
coefficients with ``identity()``, ``apply(Point2D)``, ``compose``, ``inverse``, and
``scale`` (``sqrt`` of the determinant — the uniform scale factor, used to resize a box
when mapping a detection between frames). No cv2 matrix leaks into the domain.

Unlike a frame-to-frame chain, this matches each frame against a **keyframe anchor**
and composes anchor poses into a single continuous global frame — matching every frame
against a *fixed* frame 0 has two problems: ORB correspondences thin out as the view
diverges from frame 0, and per-frame composition compounds error. Instead: match the
current frame against the anchor -> ``L`` (current -> anchor); the anchor carries a
global pose ``A`` (anchor -> global frame), so the current frame's returned transform
is ``G = A . L`` (current -> global); measure ``clipped_overlap_fraction(L)`` — the
share of the anchor rectangle still covered by the mapped current frame — and when it
drops below ``min_anchor_overlap``, promote the current frame to the new anchor with
global pose ``G``. This keeps matching against a recent, high-overlap frame (robust
ORB, no per-frame error compounding — drift accrues only at the sparse re-anchor
compositions) while the composed global pose stays continuous across re-anchors:
continuity matters because the tracker associates in the global frame, so a coordinate
jump at a re-anchor would break every track ID. The anchor is purely an estimation
reference, not a reset of the output frame. The anchor/overlap/chain policy lives in
the pure, unit-tested ``_AnchorChain``; the cv2 feature work stays in this class.
Keyframe anchoring bounds the *conditioning* of the estimate, not the *magnitude* of
coordinates — a car far from frame 0 still has large/negative global coordinates,
which is harmless for points and tracking (see "Known limitations" below).

Feature-based ORB, not intensity-based ECC: the dominant failure mode of aerial
traffic footage is moving foreground — much of the frame is the vehicles being
tracked, and a stabilizer must estimate background motion while ignoring them. ORB
(feature-based) gives explicit point correspondences, so RANSAC rejects
moving-vehicle matches as outliers; ECC (intensity-based) optimizes over all pixels
with no outlier rejection, cannot be told to ignore the cars, and on bare asphalt
often locks onto them. Both are zero-new-dependency (``cv2`` already present).
``docs/TECH_STACK.md``/``application/WORLD_PROJECTION.md`` name SuperPoint + LightGlue
as the eventual target; ORB is the intermediate, tracked as Group B3 in GitHub Issues.

Vehicle-masked feature extraction: RANSAC alone isn't enough on low-texture aerial
footage — the background is feature-poor (bare asphalt), so coherently-moving vehicles
can become the inlier majority and bias the fit. The fix masks vehicles out of
``detectAndCompute``. Where the mask comes from is pluggable behind the ``MaskSource``
Protocol below; the only implementation is ``BackgroundZoneMaskSource``, reading
operator-authored polygons (see "Detector-free ego-motion" below) — an earlier
``DetectionMaskSource``, masking from a live detector's own detections, was removed
once ``tratrac`` stopped estimating ego-motion itself, since nothing constructed it any
more. Masking is intrinsic to the estimator (``mask_source`` is a required constructor
argument, not a config key); it addresses the bias source but not every failure mode —
a genuinely static camera has no ego-motion to remove, so ``tratrac-preprocess
estimate`` should simply omit ``--background-zones`` there regardless (ORB would
otherwise inject phantom motion into the *coordinates* of parked cars).

Detector-free ego-motion is the only path. Masking from a live detector's own
detections would couple ego-motion estimation to detection — exactly why anchor
discovery couldn't be a separate pass without either losing mask quality or running
the detector twice (both explored and rejected; see "Rejected alternatives" below).
Swapping the mask *source* removes that coupling entirely: the mask comes from
operator-drawn polygons instead, so ORB needs no detector at all, ever, and
``tratrac`` itself never estimates ego-motion — it only ever reads an already-built
transform table. ``MaskSource`` is the seam this required: ``observe(detections)`` /
``mask_for(frame_index, height, width) -> NDArray[np.uint8] | None``. This class takes
one at construction instead of building the mask inline.
``BackgroundZoneMaskSource(zones: BackgroundZones)`` is the only implementation: each
zone is a polygon plus the frame it starts applying from ("use this mask from here
until a later entry supersedes it" — not tied to ORB's own re-anchor points).
``mask_for`` resolves the most recent zone at or before ``frame_index`` (falling back
to the earliest zone for a frame before the first entry) and rasterizes it as the keep
region; ``observe`` is a no-op, since the mask is entirely operator-authored.
``background_zones.json`` (``infrastructure/background/json.py``, see
``infrastructure/background/BACKGROUND_ZONES.md``) is the sidecar an external tool
(out of scope for this repo — same footing as ``calibration.json`` and
``exclusion_zones.json``) produces: an operator watches the video and draws the region
safe for ORB feature extraction, redrawing only when the view has changed enough to
warrant it.

``tratrac-preprocess estimate`` (``cli_preprocess.py``) is the tool that runs this:
walks a clip once with
``OrbEgoMotionEstimator(mask_source=BackgroundZoneMaskSource(zones))`` — no detector,
no tracker — writing an ego-motion row *and* a GSD-scale row per frame into one
transforms file (``--out``), plus the anchor PNGs (``--anchors-dir``, no separate
manifest: an anchor's pose is already that same file's row at that frame index).
``--background-zones`` is itself optional: omit it for a static camera and only the
scale rows get written (no ego-motion rows at all). ``tratrac``'s
``input.transforms_in`` is a required key naming this file — there is no
``[ego_motion]`` section, no ``enabled`` toggle, and no live-ORB fallback: ``cli.py``
loads the ego-motion-only rows via ``read_ego_motion`` into a ``TransformTable`` and
wraps it in ``PrecomputedEgoMotionEstimator`` (``infrastructure/transform/sink.py``) —
an ``EgoMotionEstimator`` that's a plain per-frame lookup, no ORB call, falling back to
the identity when the table has no ego-motion rows at all (the static case). The
detector then runs exactly once, ever, against already-known ego-motion. See
``infrastructure/transform/TRANSFORM_SINK.md`` for the full unified-row picture (scale
and world-projection homography rows live in the same file, written by
``estimate``/``tratrac-preprocess project`` respectively). Operator workflow::

    [external tool] operator watches VIDEO, draws background zones -> background_zones.json
    tratrac-preprocess estimate VIDEO --background-zones background_zones.json \\
        --out transforms.jsonl --anchors-dir anchors/ --meters-per-pixel 0.05
        # detector-free; also resolves + writes the GSD scale rows
    [external tool] operator draws exclusion zones / world-projection
        correspondences on anchors/*.png -> zones.json / calibration.json
    tratrac --config run.toml   # input.transforms_in = transforms.jsonl
        # detector runs exactly once here; ego-motion already resolved
    tratrac-preprocess project --transforms transforms.jsonl \\
        --calibration calibration.json   # fits + appends homography rows, still post-hoc
    tratrac-postprocess run.parquet --out run.trj --transforms transforms.jsonl

Rejected alternatives, before operator-authored zones was settled on: (1) a cheap
detector-free self-referential masking scheme (fit unmasked, treat RANSAC's own
outliers as the mask) — a real technique in the literature, but the same
sparse-background/dense-foreground failure mode that motivated detection-based masking
in the first place (RANSAC's *first*, unmasked pass can itself get captured by the
vehicle majority on bare asphalt) means it needs validation this project hasn't done,
and domain-specific literature for exactly this footage converges on detection-based
masking, not self-referential schemes; (2) reusing a live-detector's mask in an
earlier, separate pass would mean either running the detector twice (the "scout +
replay = ORB twice" pattern ``application/EXCLUSION_ZONES.md`` already documents
rejecting, generalized to the detector) or accepting an unmasked, lower-quality
estimate whose discovered anchor set could diverge from what the real masked run would
produce, invalidating any correspondences authored against it. Operator-authored zones
route around both: no detector dependency, and (since it *is* the real computation, not
an approximation of it) no anchor-set divergence risk. There is no
``StabilizedVideoSource`` — pixel warping was removed.

Known limitations (what measurement must watch): (1) unbounded coordinate magnitude —
the global frame is anchored to frame 0, so coordinates grow without bound under
sustained motion; harmless for points and tracking, only the SSAM export is affected
(next item) — keyframe anchoring fixes estimation robustness, not coordinate
magnitude. (2) SSAM bounds/y-flip — stabilized positions can fall outside
``[0,W]x[0,H]``, so the SSAM y-flip (``height - y``) can go negative and ``DIMENSIONS``
(kept at ``W x H``) no longer bounds the data; for MVP1.x image-space export this is
already "valid but not physically meaningful," proper world bounds are MVP2's job.
(3) axis-aligned box approximation — ``apply_transform`` transforms the box centre
exactly and scales its size, but does not re-fit a rotated box; exact for the centroid
trajectory (what velocity/heading use) and correctly zoom-normalises length/width
(fixing the GSD-varies-with-zoom problem), only the never-moved bbox-major-axis
*fallback* heading is approximate. (4) ORB robustness on static/low-texture clips — ORB
can still mis-estimate; on a static camera it manufactures phantom motion, now into
coordinates, so omit ``--background-zones`` from ``tratrac-preprocess estimate``
there. SuperPoint + LightGlue is the upgrade if measurement demands it.

Config surface (zero-defaults rule): every key in ``RunConfig`` is mandatory and "off
is explicit" (see ``application/config.py``'s module docstring). There is no
``[ego_motion]`` section, no ``enabled`` toggle, and no per-key "disabled" value:
``input.transforms_in`` is a plain required path, always — ``tratrac-preprocess
estimate`` is mandatory before every run, even a fully static camera (it's the only
place GSD scale gets resolved now too). Whether ego-motion is "on" is entirely a
property of that file's *content*: if it has no ego-motion rows, ``tratrac``'s
stabilization stage is the identity, with no config-level distinction from a
moving-drone run. The ORB tuning parameters (``n_features``, ``match_ratio``,
``min_matches``, ``ransac_threshold``, ``min_anchor_overlap``) are not part of
``tratrac``'s config at all; they live only on ``tratrac-preprocess estimate``'s CLI,
the one tool that actually runs ORB.

Relation to MVP2: MVP2 keeps the ``EgoMotionEstimator`` port and replaces the adapter —
a stabilizer upgrade (ORB -> SuperPoint + LightGlue, Group B3, GitHub Issues) if the
ORB measurement shows it's needed, plus world projection (a homography from the now
ego-motion-free image plane to metric world coordinates, making SSAM positions
physically meaningful and giving bounded, real-world coordinates). If MVP1.9's ORB
measures as "not worth keeping," the adapter is dropped but the port, ``Transform2D``,
and the coordinate-stabilization pipeline seam remain — MVP2 inherits a ready
structure.

Where the pieces live: ``domain/geometry.py`` (``Transform2D`` + ``scale``, and
``clipped_overlap_fraction`` — shapely-backed polygon intersection/area, pure overlap
geometry for re-anchoring); ``domain/ports.py`` (``EgoMotionEstimator``:
``estimate(frame) -> Transform2D``, stateful, returns the current frame -> global
transform); this module (``OrbEgoMotionEstimator`` — ORB -> ratio-tested Hamming match
against the anchor -> ``estimateAffinePartial2D`` RANSAC — plus the pure
``_AnchorChain``; exposes ``current_transform`` for the overlay);
``application/stabilization.py`` (``apply_transform(detection, transform)`` — maps a
detection's box centre exactly and scales its size by ``transform.scale``,
axis-aligned, pure); ``application/pipeline.py`` (owns the per-frame order: detect
(raw) -> observe -> ``estimate`` -> ``apply_transform`` to each detection -> track ->
record; the exporter receives the raw frame); ``infrastructure/tracking/boxmot_bot_sort.py``
(``compensate_camera_motion`` — ``cmc_method=None`` when stabilization is on, so
BoT-SORT does not *also* correct the already-stabilized boxes);
``infrastructure/export/overlay_video.py`` (maps stabilized coordinates back onto the
raw frame for drawing via the inverse of ``current_transform``, see
``infrastructure/export/overlay_video.py``'s own docstring); ``domain/background.py``
(``BackgroundZone``/``BackgroundZones``, the operator-authored polygon collection
``BackgroundZoneMaskSource`` reads); ``infrastructure/background/json.py``
(``load_background_zones``, the ``background_zones.json`` reader, sharing
``infrastructure/zones.py``'s parser with ``infrastructure/exclusion/json.py`` — see
``infrastructure/background/BACKGROUND_ZONES.md``); ``infrastructure/transform/sink.py``
(``PrecomputedEgoMotionEstimator``, the ``input.transforms_in`` side: an
``EgoMotionEstimator`` that's an exact ``TransformTable`` lookup over the
ego-motion-only rows via ``read_ego_motion``, no ORB call — the only way ``tratrac``
gets ego-motion); ``cli_preprocess.py`` (``tratrac-preprocess estimate`` — the
detector-free pre-pass: walks a clip once with ``BackgroundZoneMaskSource`` when
``--background-zones`` is given, writing the transforms file's ego-motion and scale
rows and the anchor PNGs, no manifest).
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
