"""Sidecar JSON reader for road-link zones.

Thin I/O adapter mirroring ``infrastructure/exclusion/json.py``: reads a
per-scene JSON file listing labeled link polygons into the pure ``LinkZones``
value object.

Schema::

    { "link_zones": [
        { "link_id": 1,
          "reference_frame": 0,
          "vertices": [[x1, y1], [x2, y2], [x3, y3]] }
    ] }

``link_id`` is required and must be a positive integer (``0`` is the SSAM
"unknown" sentinel, reserved for points outside every zone).
``reference_frame`` is the frame index the vertices are drawn on (optional,
defaults to ``0`` for a static camera; for a moving drone it is one of the
scout's anchor frame indices). ``vertices`` are pixel coordinates; each
polygon needs at least three.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tratrac.domain.geometry import Point2D, Polygon
from tratrac.domain.road_graph import LinkZone, LinkZones


def load_link_zones(path: Path) -> LinkZones:
	"""Parse a sidecar JSON file into ``LinkZones``.

	Raises ``FileNotFoundError`` if ``path`` is absent and ``ValueError`` on any
	malformed content (bad JSON, wrong shape, missing/non-positive link_id,
	fewer than three vertices), each re-wrapped with the file path so the CLI
	can report it cleanly.
	"""
	try:
		with path.open("rb") as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc

	if not isinstance(document, dict) or "link_zones" not in document:
		raise ValueError(f'{path} must be a JSON object with a "link_zones" array.')
	raw_zones = document["link_zones"]
	if not isinstance(raw_zones, list):
		raise ValueError(f'{path}: "link_zones" must be an array.')

	zones = [_parse_zone(raw, index, path) for index, raw in enumerate(raw_zones)]
	return LinkZones(zones=tuple(zones))


def _parse_zone(raw: Any, index: int, path: Path) -> LinkZone:
	if not isinstance(raw, dict) or "vertices" not in raw:
		raise ValueError(f'{path}: zone {index} must be an object with a "vertices" array.')
	link_id = _parse_link_id(raw.get("link_id"), index, path)
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
	try:
		return LinkZone(link_id=link_id, reference_frame=reference_frame, polygon=polygon)
	except ValueError as exc:
		raise ValueError(f"{path}: zone {index}: {exc}") from exc


def _parse_link_id(raw: Any, index: int, path: Path) -> int:
	if raw is None:
		raise ValueError(f"{path}: zone {index} is missing required field link_id.")
	if isinstance(raw, bool) or not isinstance(raw, int):
		raise ValueError(f"{path}: zone {index} link_id must be an integer.")
	return raw


def _parse_reference_frame(raw: Any, index: int, path: Path) -> int:
	if isinstance(raw, bool) or not isinstance(raw, int):
		raise ValueError(f"{path}: zone {index} reference_frame must be an integer.")
	if raw < 0:
		raise ValueError(f"{path}: zone {index} reference_frame must be >= 0.")
	return raw
