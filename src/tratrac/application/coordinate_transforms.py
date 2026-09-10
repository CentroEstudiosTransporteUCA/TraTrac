"""``CoordinateTransform`` implementations: every "map applied to pixel coordinates" in
the system, behind one shared interface.

Unifies what used to be three unrelated mechanisms — the GSD metric scale (ad hoc
``* scale`` / ``/ scale`` arithmetic), the ego-motion/anchor-pose table (a bare
``dict[int, Transform2D]``), and world-projection homographies (``WorldProjector``) —
as implementations of one ``CoordinateTransform``/``InvertibleCoordinateTransform``
Protocol (``domain/ports.py``): ``apply(point, frame_index)`` /
``reverse(point, frame_index)``. Renamed from ``world_projection.py`` since its scope
broadened from "world projection only" to every coordinate transform in the system.

Six impls today:

* ``IdentityTransform`` — the Null Object: returns the point unchanged.
* ``ScaleTransform`` — a constant isotropic scale (GSD calibration); ``reverse``
  divides back out.
* ``PerFrameTransform`` — exact ``frame_index`` lookup into a
  ``dict[int, Transform2D]``. Serves both the dense ego-motion table (every frame
  present) and a sparse anchor-pose lookup (only anchor frames present, queried
  only at a zone's/correspondence's own ``reference_frame`` — always an exact hit
  by construction).
* ``PerAnchorTransform`` (Group C3, ``docs/IMPLEMENTATION_PLAN.md``) — one
  homography per keyframe anchor, selected by ``frame_index``, for a wide-swept
  scene where no single homography covers the whole clip. **The single-homography
  case (a bounded / static scene) is just this with one anchor.**
* ``MultiPlaneTransform`` (Group C5 / MVP3, ``docs/roadmap/mvp3.md``) — one
  homography per **elevation plane** (ground, bridge, overpass, ...), selected by
  classifying the point itself against the plane zones — orthogonal to
  ``PerAnchorTransform``'s selection by *when* (``frame_index``): this one selects
  by *where* (spatial, independent of time). Apply-only; see its docstring for why
  it isn't invertible.
* ``ComposedTransform`` (via ``compose``/``compose_invertible``) — folds several
  stages into one. Not load-bearing for any current call site (every consumer
  today only ever needs one stage at a time — the ego-motion table, world
  projection, and scale never chain on the same call), provided for a uniform
  vocabulary and a future case that does chain (e.g. ego-motion-lift-then-project).

Every fitter (grouping correspondences by anchor or by plane, one ``cv2`` fit per
group) lives in ``cli_postprocess.py`` — this module only applies an
already-fitted matrix, plus the pure scale/per-frame/identity math.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tratrac.application.road_graph import GlobalLabeledPolygon, plane_id_for_point
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.ports import CoordinateTransform, InvertibleCoordinateTransform


class IdentityTransform:
	"""``CoordinateTransform`` Null Object: pass the point through unchanged."""

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return point

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return point


class ScaleTransform:
	"""A constant isotropic scale (GSD calibration) — ignores ``frame_index``.

	Replaces the ad hoc ``* scale`` / ``/ scale`` arithmetic ``track_smoothing.py``
	used to do inline: the GSD metric scale already behaved like a coordinate
	transform (``build_state`` multiplies the centroid by it, not just derived
	Length/Width/Speed/Acceleration) without being represented as one.
	"""

	def __init__(self, scale: float) -> None:
		if scale <= 0.0:
			raise ValueError(f"ScaleTransform scale must be positive, got {scale}.")
		self._scale = scale

	@property
	def factor(self) -> float:
		"""The raw metres-per-pixel scalar, for scaling a magnitude that isn't a
		position (a velocity, an acceleration, a length/width) — the same "resize a
		non-point quantity" case ``Transform2D.scale`` exists for."""
		return self._scale

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return Point2D(point.x * self._scale, point.y * self._scale)

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return Point2D(point.x / self._scale, point.y / self._scale)


class PerFrameTransform:
	"""Exact ``frame_index`` lookup into a ``{frame_index: Transform2D}`` table.

	Raises ``KeyError`` (via ``__getitem__``, wrapped for a clear message) on a
	frame not present in the table — same contract the CSV-era ``read_transforms``
	had. Inverts don't require a runtime check across calls: ``Transform2D.inverse()``
	is cheap (a 2x2 matrix invert), so ``reverse`` just inverts the looked-up
	transform each time rather than pre-computing an inverse table.
	"""

	def __init__(self, transforms: dict[int, Transform2D]) -> None:
		self._transforms = dict(transforms)

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		return self._lookup(frame_index).apply(point)

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		return self._lookup(frame_index).inverse().apply(point)

	def at(self, frame_index: int) -> Transform2D | None:
		"""The raw ``Transform2D`` recorded for ``frame_index``, or ``None`` if absent.

		For a consumer that needs the transform object itself (e.g. ``tratrac-render``,
		which composes it into an infrastructure-level drawing seam) rather than just
		mapping one point through it, and that tolerates a frame the table doesn't
		cover — unlike ``apply``/``reverse``, which raise on a missing frame.
		"""
		return self._transforms.get(frame_index)

	def _lookup(self, frame_index: int) -> Transform2D:
		transform = self._transforms.get(frame_index)
		if transform is None:
			raise KeyError(f"no transform recorded for frame {frame_index}.")
		return transform


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


class PerAnchorTransform:
	"""Picks the nearest keyframe anchor's homography by ``frame_index``.

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

	Each anchor's inverse is computed once at construction (mirroring the eager
	``.inverse()`` the old ``WorldProjector``-era class returned as a whole new
	object), not per ``reverse`` call — anchor selection is by ``frame_index``,
	unchanged by direction, so a fixed inverse table is always valid.
	"""

	def __init__(self, homographies_by_anchor: dict[int, NDArray[np.float64]]) -> None:
		if not homographies_by_anchor:
			raise ValueError("PerAnchorTransform needs at least one anchor homography.")
		self._by_anchor = dict(homographies_by_anchor)
		self._inverse_by_anchor = {
			anchor: np.linalg.inv(matrix).astype(np.float64)
			for anchor, matrix in self._by_anchor.items()
		}
		self._anchor_frames = sorted(self._by_anchor)

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		anchor = self._nearest_anchor(frame_index)
		return _apply_homography(self._by_anchor[anchor], point)

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		anchor = self._nearest_anchor(frame_index)
		return _apply_homography(self._inverse_by_anchor[anchor], point)

	def shifted(self, dx: float, dy: float) -> PerAnchorTransform:
		"""An equivalent transform whose output is additionally translated by ``(dx, dy)``,
		applied uniformly to every anchor's homography (one, for the single-homography case).

		Used to fold ``_normalize_world_recording``'s post-hoc 0-origin shift into the
		transform itself, so ``.reverse()`` on the result undoes the *whole* forward
		transform (homography + shift), not just the homography — see
		``cli_postprocess.py``'s ``_project_to_world``.
		"""
		t = _translation_matrix(dx, dy)
		return PerAnchorTransform(
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


class MultiPlaneTransform:
	"""Picks a homography by classifying the point against elevation-plane zones.

	``homographies_by_plane`` maps each plane id to the homography fitted from correspondences
	classified into it; ``global_plane_polygons`` are the same plane zones (already mapped into
	the global frame, see ``application/road_graph.to_global_plane_polygons``) used to fit them,
	so a point and the correspondences that determined its plane's homography are classified by
	the exact same rule. ``frame_index`` is ignored — plane membership is spatial, not temporal
	(contrast ``PerAnchorTransform``, which selects by *when*; this selects by *where*).

	Raises ``ValueError`` from ``apply`` if a point classifies into a plane with no fitted
	homography — e.g. the plane zones don't fully cover the scene and a point falls into the
	``0`` (unassigned) default with no ``plane_id: 0`` calibration group. Failing loudly here
	is deliberate: silently guessing a "closest" plane would produce a plausible-looking but
	physically wrong projection, and there's no principled way to decide "closest" for an
	elevation surface which vehicles can't otherwise be evaluated against.

	**Deliberately has no ``reverse``** (contrast ``PerAnchorTransform``, which has one,
	single-homography included as its one-anchor case): plane selection here classifies the
	*input* image point's position, which is exactly what's unknown when starting from a world
	point — see ``domain/ports.py``'s ``InvertibleCoordinateTransform`` and
	``application/SMOOTHING.md``'s "Dual-space export" section. A future version could carry
	each observation's plane id forward through smoothing so the reverse pass knows which
	homography to invert, but that isn't built yet.
	"""

	def __init__(
		self,
		homographies_by_plane: dict[int, NDArray[np.float64]],
		global_plane_polygons: tuple[GlobalLabeledPolygon, ...],
	) -> None:
		if not homographies_by_plane:
			raise ValueError("MultiPlaneTransform needs at least one plane homography.")
		self._by_plane = dict(homographies_by_plane)
		self._global_plane_polygons = global_plane_polygons

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
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


@dataclass(frozen=True, slots=True)
class ComposedTransform:
	"""Folds several ``CoordinateTransform`` stages into one, applied in listed order.

	Construct via ``compose`` (or ``compose_invertible`` for an all-invertible
	chain), not directly.
	"""

	stages: tuple[CoordinateTransform, ...]

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in self.stages:
			point = stage.apply(point, frame_index)
		return point


def compose(*stages: CoordinateTransform) -> CoordinateTransform:
	"""Fold ``stages`` into one ``CoordinateTransform``, applied in order."""
	return ComposedTransform(stages)


@dataclass(frozen=True, slots=True)
class _ComposedInvertibleTransform:
	stages: tuple[InvertibleCoordinateTransform, ...]

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in self.stages:
			point = stage.apply(point, frame_index)
		return point

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in reversed(self.stages):
			point = stage.reverse(point, frame_index)
		return point


def compose_invertible(*stages: InvertibleCoordinateTransform) -> InvertibleCoordinateTransform:
	"""Fold ``stages`` into one ``InvertibleCoordinateTransform``, applied in order."""
	return _ComposedInvertibleTransform(stages)


def local_scale_at(transform: CoordinateTransform, point: Point2D, frame_index: int = 0) -> float:
	"""Estimate metres-per-pixel at ``point`` (the transform's local scale there).

	Projects ``point`` and a neighbour one pixel away in x and measures the world gap.
	Used to convert pixel-tuned smoother noise into the world units the projection
	produces, so the Kalman behavior is preserved.
	"""
	here = transform.apply(point, frame_index)
	over = transform.apply(Point2D(point.x + 1.0, point.y), frame_index)
	return math.hypot(over.x - here.x, over.y - here.y)
