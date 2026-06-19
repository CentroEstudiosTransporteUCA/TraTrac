"""Sidecar JSON reader for image-space exclusion zones.

Thin I/O adapter (like ``infrastructure/config/toml.py``): reads a per-scene
JSON file listing the polygons whose detections must not be analyzed into the
pure ``ExclusionZones`` value object. See vault/21_exclusion_zones.md.

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

import json
from pathlib import Path
from typing import Any

from tratrac.domain.exclusion import ExclusionZone, ExclusionZones
from tratrac.domain.geometry import Point2D, Polygon


def load_exclusion_zones(path: Path) -> ExclusionZones:
	"""Parse a sidecar JSON file into ``ExclusionZones``.

	Raises ``FileNotFoundError`` if ``path`` is absent and ``ValueError`` on any
	malformed content (bad JSON, wrong shape, fewer than three vertices), each
	re-wrapped with the file path so the CLI can report it cleanly.
	"""
	try:
		with path.open("rb") as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc

	if not isinstance(document, dict) or "exclusion_zones" not in document:
		raise ValueError(f'{path} must be a JSON object with an "exclusion_zones" array.')
	raw_zones = document["exclusion_zones"]
	if not isinstance(raw_zones, list):
		raise ValueError(f'{path}: "exclusion_zones" must be an array.')

	zones = [_parse_zone(raw, index, path) for index, raw in enumerate(raw_zones)]
	return ExclusionZones(zones=tuple(zones))


def _parse_zone(raw: Any, index: int, path: Path) -> ExclusionZone:
	if not isinstance(raw, dict) or "vertices" not in raw:
		raise ValueError(f'{path}: zone {index} must be an object with a "vertices" array.')
	reference_frame = _parse_reference_frame(raw.get("reference_frame", 0), index, path)
	raw_vertices = raw["vertices"]
	if not isinstance(raw_vertices, list):
		raise ValueError(f"{path}: zone {index} vertices must be an array.")
	vertices: list[Point2D] = []
	for vertex in raw_vertices:
		if (
			not isinstance(vertex, list | tuple)
			or len(vertex) != 2
			or not all(isinstance(c, int | float) and not isinstance(c, bool) for c in vertex)
		):
			raise ValueError(f"{path}: zone {index} vertices must be [x, y] number pairs.")
		vertices.append(Point2D(float(vertex[0]), float(vertex[1])))
	try:
		polygon = Polygon(vertices=tuple(vertices))
	except ValueError as exc:
		raise ValueError(f"{path}: zone {index}: {exc}") from exc
	return ExclusionZone(reference_frame=reference_frame, polygon=polygon)


def _parse_reference_frame(raw: Any, index: int, path: Path) -> int:
	if isinstance(raw, bool) or not isinstance(raw, int):
		raise ValueError(f"{path}: zone {index} reference_frame must be an integer.")
	if raw < 0:
		raise ValueError(f"{path}: zone {index} reference_frame must be >= 0.")
	return raw
