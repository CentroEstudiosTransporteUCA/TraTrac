"""Image-space background zones: regions safe for ORB feature extraction without a detector.

An operator-authored (external tool, out of scope for this repo) set of pixel
polygons marking "use this region's features for ego-motion", so
`tratrac-preprocess` never needs to run a detector at all. See
infrastructure/video/ego_motion_orb.py's module docstring's "Detector-free ego-motion" section.

Structurally identical to `domain/exclusion.py`'s `ExclusionZone`/`ExclusionZones`
(a `reference_frame` + `Polygon`) but kept as its own type since the *intent*
differs (mask-for-estimation vs. drop-from-analytics) even though the shape is the
same — matches this repo's style of small, purpose-named domain types.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Polygon


@dataclass(frozen=True, slots=True)
class BackgroundZone:
	"""One background polygon and the frame at which it starts applying.

	``reference_frame`` marks "use this polygon as the keep-region for ORB feature
	extraction from this frame until a later entry supersedes it" — it need not
	align with ORB's own re-anchor points; the external tool that authors these
	zones decides how often to redraw one.
	"""

	reference_frame: int
	polygon: Polygon


@dataclass(frozen=True, slots=True)
class BackgroundZones:
	"""A collection of background zones. Must be non-empty — unlike exclusion zones
	(where "no zones" legally means "exclude nothing"), a mask source with zero
	zones has no keep-region to fall back to."""

	zones: tuple[BackgroundZone, ...]

	def __post_init__(self) -> None:
		if not self.zones:
			raise ValueError("BackgroundZones needs at least one zone.")
