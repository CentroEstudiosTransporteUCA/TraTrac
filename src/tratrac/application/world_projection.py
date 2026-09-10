"""``WorldProjector`` implementations: map image points onto the metric world plane.

The pure side of MVP2 world projection (``src/tratrac/application/WORLD_PROJECTION.md``). The homography *matrix* is
fitted in infrastructure (cv2); here we only *apply* it — a 3x3 projective multiply plus
the perspective divide — so this stays numpy-only and onion-clean.

Three impls today:

* ``IdentityWorldProjector`` — the Null Object: returns the point unchanged (image-space,
  the pre-MVP2 behavior; the projection step is simply not applied).
* ``PerAnchorWorldProjector`` (Group C3, ``docs/IMPLEMENTATION_PLAN.md``) — one homography per
  keyframe anchor, selected by ``frame_index``, for a wide-swept scene where no single
  homography covers the whole clip. **The single-homography case (a bounded / static scene)
  is just this with one anchor** — a former separate ``SingleHomographyProjector`` class was
  removed once that redundancy was noticed: with exactly one entry, anchor selection always
  resolves to it regardless of ``frame_index``, so there was never a behavioral difference to
  justify two classes, only two code paths fitting the same shape.
* ``MultiHomographyWorldProjector`` (Group C5 / MVP3, ``docs/roadmap/mvp3.md``) — one homography
  per **elevation plane** (ground, bridge, overpass, ...), selected by classifying the point
  itself against the plane zones — orthogonal to ``PerAnchorWorldProjector``'s selection by
  *when* (``frame_index``): this one selects by *where* (spatial, independent of time). Not
  unifiable with ``PerAnchorWorldProjector`` the same way — its selection key is the point's
  own position, not something available independent of direction (see
  ``InvertibleWorldProjector`` in ``domain/ports.py`` for why that also blocks inversion).

Every fitter (grouping correspondences by anchor or by plane, one ``cv2`` fit per group) lives
in ``cli_postprocess.py`` — this module only applies an already-fitted matrix.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from numpy.typing import NDArray

from tratrac.application.road_graph import GlobalLabeledPolygon, plane_id_for_point
from tratrac.domain.geometry import Point2D
from tratrac.domain.ports import WorldProjector


class IdentityWorldProjector:
	"""``WorldProjector`` Null Object: pass the point through unchanged (image-space)."""

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index  # identity: no per-anchor selection
		return point

	def inverse(self) -> IdentityWorldProjector:
		"""The identity's own inverse is itself."""
		return self


def _apply_homography(matrix: NDArray[np.float64], point: Point2D) -> Point2D:
	"""Project one point through a 3x3 homography, including the perspective divide."""
	projected = matrix @ np.array([point.x, point.y, 1.0])
	w = float(projected[2])
	if w == 0.0:
		raise ValueError("homography mapped a point to infinity (w = 0).")
	return Point2D(float(projected[0]) / w, float(projected[1]) / w)


def _translation_matrix(dx: float, dy: float) -> NDArray[np.float64]:
	"""A 3x3 homogeneous matrix translating by ``(dx, dy)``, for composing onto a homography."""
	return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


class PerAnchorWorldProjector:
	"""``WorldProjector`` picking the nearest keyframe anchor's homography by ``frame_index``.

	``homographies_by_anchor`` maps each anchor's frame index to the homography fitted from
	correspondences authored on it. A point's anchor is the one whose frame index is closest
	to ``frame_index`` — the same "nearest keyframe" principle the ORB stabilizer itself uses
	to decide when a scene region is still well-represented by an anchor
	(``src/tratrac/infrastructure/video/EGO_MOTION.md``): a homography fit on a nearby anchor is more likely to
	still describe the local ground geometry than one fit far away in time/space.

	**A single-entry map is the single-homography (bounded/static scene) case** — with exactly
	one anchor, ``_nearest_anchor`` always resolves to it regardless of ``frame_index``, which
	is exactly "one homography for the whole scene." This is why there's no separate
	single-homography class: it would be structurally identical code fitting the same shape.

	**Not interpolated** between neighboring anchors' homographies (when there's more than
	one) — a deliberate simplicity choice (``docs/IMPLEMENTATION_PLAN.md`` Group C3 open
	question): interpolating projective transforms is not a simple linear blend (it would need
	to be done in a shared parameter space, e.g. decomposed rotation/translation, to avoid
	producing an invalid, non-projective intermediate matrix), and a hard switch is enough as
	long as anchors are reasonably dense relative to scene changes. Revisit if
	``scripts/validate_trj.py`` shows a discontinuity at anchor boundaries.
	"""

	def __init__(self, homographies_by_anchor: dict[int, NDArray[np.float64]]) -> None:
		if not homographies_by_anchor:
			raise ValueError("PerAnchorWorldProjector needs at least one anchor homography.")
		self._by_anchor = dict(homographies_by_anchor)
		self._anchor_frames = sorted(self._by_anchor)

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		anchor = self._nearest_anchor(frame_index)
		return _apply_homography(self._by_anchor[anchor], point)

	def inverse(self) -> PerAnchorWorldProjector:
		"""The world→image projector: anchor selection is by ``frame_index``, unchanged by
		direction, so inverting every anchor's own matrix is enough."""
		return PerAnchorWorldProjector(
			{
				anchor: np.linalg.inv(matrix).astype(np.float64)
				for anchor, matrix in self._by_anchor.items()
			}
		)

	def shifted(self, dx: float, dy: float) -> PerAnchorWorldProjector:
		"""An equivalent projector whose output is additionally translated by ``(dx, dy)``,
		applied uniformly to every anchor's homography (one, for the single-homography case).

		Used to fold ``_normalize_world_recording``'s post-hoc 0-origin shift into the
		projector itself, so ``.inverse()`` on the result undoes the *whole* forward
		transform (homography + shift), not just the homography — see
		``cli_postprocess.py``'s ``_project_to_world``.
		"""
		t = _translation_matrix(dx, dy)
		return PerAnchorWorldProjector(
			{anchor: t @ matrix for anchor, matrix in self._by_anchor.items()}
		)

	def _nearest_anchor(self, frame_index: int) -> int:
		frames = self._anchor_frames
		i = bisect.bisect_left(frames, frame_index)
		if i == 0:
			return frames[0]
		if i == len(frames):
			return frames[-1]
		before, after = frames[i - 1], frames[i]
		return before if frame_index - before <= after - frame_index else after


class MultiHomographyWorldProjector:
	"""``WorldProjector`` picking a homography by classifying the point against plane zones.

	``homographies_by_plane`` maps each plane id to the homography fitted from correspondences
	classified into it; ``global_plane_polygons`` are the same plane zones (already mapped into
	the global frame, see ``application/road_graph.to_global_plane_polygons``) used to fit them,
	so a point and the correspondences that determined its plane's homography are classified by
	the exact same rule. ``frame_index`` is ignored — plane membership is spatial, not temporal
	(contrast ``PerAnchorWorldProjector``, which selects by *when*; this selects by *where*).

	Raises ``ValueError`` from ``to_world`` if a point classifies into a plane with no fitted
	homography — e.g. the plane zones don't fully cover the scene and a point falls into the
	``0`` (unassigned) default with no ``plane_id: 0`` calibration group. Failing loudly here
	is deliberate: silently guessing a "closest" plane would produce a plausible-looking but
	physically wrong projection, and there's no principled way to decide "closest" for an
	elevation surface which vehicles can't otherwise be evaluated against.

	**Deliberately has no ``.inverse()``** (contrast ``PerAnchorWorldProjector``, which does,
	single-homography included as its one-anchor case): plane selection here classifies the *input* image
	point's position, which is exactly what's unknown when starting from a world point — see
	``domain/ports.py``'s ``InvertibleWorldProjector`` and ``application/SMOOTHING.md``'s
	"Dual-space export" section. A future version could carry each observation's plane id
	forward through smoothing so the reverse pass knows which homography to invert, but that
	isn't built yet.
	"""

	def __init__(
		self,
		homographies_by_plane: dict[int, NDArray[np.float64]],
		global_plane_polygons: tuple[GlobalLabeledPolygon, ...],
	) -> None:
		if not homographies_by_plane:
			raise ValueError("MultiHomographyWorldProjector needs at least one plane homography.")
		self._by_plane = dict(homographies_by_plane)
		self._global_plane_polygons = global_plane_polygons

	def to_world(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index  # plane membership is spatial, not temporal
		plane_id = plane_id_for_point(point, self._global_plane_polygons)
		matrix = self._by_plane.get(plane_id)
		if matrix is None:
			raise ValueError(
				f"point {point} classified into plane {plane_id}, which has no fitted "
				"homography; check the plane zones cover the whole scene and the "
				"calibration has correspondences for every plane."
			)
		return _apply_homography(matrix, point)


def local_scale_at(projector: WorldProjector, point: Point2D, frame_index: int = 0) -> float:
	"""Estimate metres-per-pixel at ``point`` (the homography's local scale there).

	Projects ``point`` and a neighbour one pixel away in x and measures the world gap.
	Used to convert pixel-tuned smoother noise into the world units the projection
	produces, so the Kalman behavior is preserved.
	"""
	here = projector.to_world(point, frame_index)
	over = projector.to_world(Point2D(point.x + 1.0, point.y), frame_index)
	return math.hypot(over.x - here.x, over.y - here.y)
