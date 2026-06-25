"""World-projection calibration: image↔world ground-plane correspondences.

The metric input MVP2 needs that pure feature-matching cannot supply (see
``vault/06_mvp2.md``): a set of points whose pixel position *and* real-world ground
coordinates are both known. ≥4 coplanar correspondences determine the homography that
maps the (stabilized) image frame onto the metric ground plane.

Each correspondence is authored on a *reference frame*, exactly like an exclusion zone
(``vault/21_exclusion_zones.md``): ``reference_frame = 0`` for a static camera / single
anchor; for a moving drone it is one of the keyframe anchors the run exported, mapped
into the global frame by that anchor's pose. Carrying ``reference_frame`` from day one is
what lets the single-homography path generalize to a per-anchor one without a schema
change.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Point2D


@dataclass(frozen=True, slots=True)
class Correspondence:
	"""One image↔world point pair on the ground plane.

	``image`` is a pixel coordinate in ``reference_frame``; ``world`` is its real ground
	position in metres (in one world frame shared across all correspondences).
	"""

	reference_frame: int
	image: Point2D
	world: Point2D


@dataclass(frozen=True, slots=True)
class Calibration:
	"""A set of correspondences from which a world homography is fitted. Pure data."""

	correspondences: tuple[Correspondence, ...]
