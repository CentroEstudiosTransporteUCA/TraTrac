"""Sidecar JSON reader for image-space exclusion zones.

Thin I/O adapter (like ``infrastructure/config/toml.py``): reads a per-scene
JSON file listing the polygons whose detections must not be analyzed into the
pure ``ExclusionZones`` value object. See src/tratrac/application/EXCLUSION_ZONES.md.

Schema::

    { "exclusion_zones": [
        { "label": "parking_lot",
          "reference_frame": 0,
          "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
    ] }

``label`` is optional (operator documentation only). ``reference_frame`` is the
frame index the vertices are drawn on (optional, defaults to ``0`` for a static
camera; for a moving drone it is one of the scout's anchor frame indices).
``vertices`` are pixel coordinates; each polygon needs at least three.
"""

from __future__ import annotations

from pathlib import Path

from tratrac.domain.exclusion import ExclusionZone, ExclusionZones
from tratrac.infrastructure.zones import load_reference_frame_polygons


def load_exclusion_zones(path: Path) -> ExclusionZones:
	"""Parse a sidecar JSON file into ``ExclusionZones``.

	Raises ``FileNotFoundError`` if ``path`` is absent and ``ValueError`` on any
	malformed content (bad JSON, wrong shape, fewer than three vertices), each
	re-wrapped with the file path so the CLI can report it cleanly.
	"""
	zones = load_reference_frame_polygons(path, array_key="exclusion_zones", build=ExclusionZone)
	return ExclusionZones(zones=zones)
