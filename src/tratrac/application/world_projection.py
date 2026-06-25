"""``WorldProjector`` implementations: map image points onto the metric world plane.

The pure side of MVP2 world projection (``vault/06_mvp2.md``). The homography *matrix* is
fitted in infrastructure (cv2); here we only *apply* it — a 3x3 projective multiply plus
the perspective divide — so this stays numpy-only and onion-clean.

Two impls today:

* ``IdentityWorldProjector`` — the Null Object: returns the point unchanged (image-space,
  the pre-MVP2 behavior; the projection step is simply not applied).
* ``SingleHomographyProjector`` — one homography for the whole (single-anchor / bounded)
  scene; ignores ``frame_index``.

A future ``PerAnchorWorldProjector`` (the moving-drone path, see ``vault/06_mvp2.md`` §
migration) would use ``frame_index`` to pick the anchor's homography — same port, so it
drops in without touching callers.
"""

from __future__ import annotations

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
		projected = self._matrix @ np.array([point.x, point.y, 1.0])
		w = float(projected[2])
		if w == 0.0:
			raise ValueError("homography mapped a point to infinity (w = 0).")
		return Point2D(float(projected[0]) / w, float(projected[1]) / w)


def local_scale_at(projector: WorldProjector, point: Point2D, frame_index: int = 0) -> float:
	"""Estimate metres-per-pixel at ``point`` (the homography's local scale there).

	Projects ``point`` and a neighbour one pixel away in x and measures the world gap.
	Used to convert pixel-tuned smoother noise into the world units the projection
	produces, so the Kalman behavior is preserved.
	"""
	here = projector.to_world(point, frame_index)
	over = projector.to_world(Point2D(point.x + 1.0, point.y), frame_index)
	return math.hypot(over.x - here.x, over.y - here.y)
