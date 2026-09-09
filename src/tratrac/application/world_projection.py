"""``WorldProjector`` implementations: map image points onto the metric world plane.

The pure side of MVP2 world projection (``src/tratrac/application/WORLD_PROJECTION.md``). The homography *matrix* is
fitted in infrastructure (cv2); here we only *apply* it — a 3x3 projective multiply plus
the perspective divide — so this stays numpy-only and onion-clean.

Three impls today:

* ``IdentityWorldProjector`` — the Null Object: returns the point unchanged (image-space,
  the pre-MVP2 behavior; the projection step is simply not applied).
* ``SingleHomographyProjector`` — one homography for the whole (single-anchor / bounded)
  scene; ignores ``frame_index``.
* ``PerAnchorWorldProjector`` (Group C3, ``docs/IMPLEMENTATION_PLAN.md``) — one homography per
  keyframe anchor, selected by ``frame_index``, for a wide-swept scene where no single
  homography covers the whole clip. The fitter (grouping correspondences by anchor, one
  ``cv2`` fit per group) lives in ``cli_postprocess.py``, alongside ``SingleHomographyProjector``'s
  existing fitter — this module only applies an already-fitted matrix.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.geometry import Point2D
from tratrac.domain.ports import WorldProjector


class IdentityWorldProjector:
	"""``WorldProjector`` Null Object: pass the point through unchanged (image-space)."""

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index  # identity: no per-anchor selection
		return point


def _apply_homography(matrix: NDArray[np.float64], point: Point2D) -> Point2D:
	"""Project one point through a 3x3 homography, including the perspective divide."""
	projected = matrix @ np.array([point.x, point.y, 1.0])
	w = float(projected[2])
	if w == 0.0:
		raise ValueError("homography mapped a point to infinity (w = 0).")
	return Point2D(float(projected[0]) / w, float(projected[1]) / w)


class SingleHomographyProjector:
	"""``WorldProjector`` applying one 3x3 homography (image → world metres) to every point.

	``matrix`` maps the stabilized/global image frame onto the ground plane; it is fitted
	once from the calibration correspondences. ``frame_index`` is ignored (one homography
	for the whole scene).
	"""

	def __init__(self, matrix: NDArray[np.float64]) -> None:
		self._matrix = matrix

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index  # single homography: same map everywhere
		return _apply_homography(self._matrix, point)


class PerAnchorWorldProjector:
	"""``WorldProjector`` picking the nearest keyframe anchor's homography by ``frame_index``.

	``homographies_by_anchor`` maps each anchor's frame index to the homography fitted from
	correspondences authored on it. A point's anchor is the one whose frame index is closest
	to ``frame_index`` — the same "nearest keyframe" principle the ORB stabilizer itself uses
	to decide when a scene region is still well-represented by an anchor
	(``src/tratrac/infrastructure/video/EGO_MOTION.md``): a homography fit on a nearby anchor is more likely to
	still describe the local ground geometry than one fit far away in time/space.

	**Not interpolated** between neighboring anchors' homographies — a deliberate simplicity
	choice (``docs/IMPLEMENTATION_PLAN.md`` Group C3 open question): interpolating projective
	transforms is not a simple linear blend (it would need to be done in a shared parameter
	space, e.g. decomposed rotation/translation, to avoid producing an invalid, non-projective
	intermediate matrix), and a hard switch is enough as long as anchors are reasonably dense
	relative to scene changes. Revisit if ``scripts/validate_trj.py`` shows a discontinuity at
	anchor boundaries.
	"""

	def __init__(self, homographies_by_anchor: dict[int, NDArray[np.float64]]) -> None:
		if not homographies_by_anchor:
			raise ValueError("PerAnchorWorldProjector needs at least one anchor homography.")
		self._by_anchor = dict(homographies_by_anchor)
		self._anchor_frames = sorted(self._by_anchor)

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		anchor = self._nearest_anchor(frame_index)
		return _apply_homography(self._by_anchor[anchor], point)

	def _nearest_anchor(self, frame_index: int) -> int:
		frames = self._anchor_frames
		i = bisect.bisect_left(frames, frame_index)
		if i == 0:
			return frames[0]
		if i == len(frames):
			return frames[-1]
		before, after = frames[i - 1], frames[i]
		return before if frame_index - before <= after - frame_index else after


def local_scale_at(projector: WorldProjector, point: Point2D, frame_index: int = 0) -> float:
	"""Estimate metres-per-pixel at ``point`` (the homography's local scale there).

	Projects ``point`` and a neighbour one pixel away in x and measures the world gap.
	Used to convert pixel-tuned smoother noise into the world units the projection
	produces, so the Kalman behavior is preserved.
	"""
	here = projector.to_world(point, frame_index)
	over = projector.to_world(Point2D(point.x + 1.0, point.y), frame_index)
	return math.hypot(over.x - here.x, over.y - here.y)
