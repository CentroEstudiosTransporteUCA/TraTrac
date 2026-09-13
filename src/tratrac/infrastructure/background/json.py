"""Sidecar JSON reader for operator-authored background zones.

An external tool (out of scope for this repo, same as `calibration.json` and
`exclusion_zones.json` today) lets an operator watch a video and draw the regions
safe for ORB feature extraction, producing this file. `tratrac-preprocess` reads it
to mask ego-motion estimation without ever running a detector. See
src/tratrac/infrastructure/video/EGO_MOTION.md's "Detector-free ego-motion" section.

Schema (identical shape to `exclusion_zones.json`, different array key)::

    { "background_zones": [
        { "reference_frame": 0,   "vertices": [[x1, y1], [x2, y2], [x3, y3]] },
        { "reference_frame": 850, "vertices": [[...]] }
    ] }

``reference_frame`` marks the frame this polygon starts applying from (optional,
defaults to ``0``) — "use this mask from here until a later entry supersedes it,"
not tied to ORB's own re-anchor points. ``vertices`` are pixel coordinates, at
least three per polygon.
"""

from __future__ import annotations

from pathlib import Path

from tratrac.domain.background import BackgroundZone, BackgroundZones
from tratrac.infrastructure.zones import load_reference_frame_polygons


def load_background_zones(path: Path) -> BackgroundZones:
	"""Parse a sidecar JSON file into ``BackgroundZones``.

	Raises ``FileNotFoundError`` if ``path`` is absent and ``ValueError`` on any
	malformed content (bad JSON, wrong shape, fewer than three vertices, or zero
	zones), each re-wrapped with the file path so the CLI can report it cleanly.
	"""
	zones = load_reference_frame_polygons(path, array_key="background_zones", build=BackgroundZone)
	if not zones:
		raise ValueError(f'{path}: "background_zones" must contain at least one zone.')
	return BackgroundZones(zones=zones)
