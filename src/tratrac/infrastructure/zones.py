"""Shared reader for a JSON array of ``{reference_frame, vertices}`` polygons.

The wire shape `infrastructure/exclusion/json.py` (``exclusion_zones``) and
`infrastructure/background/json.py` (``background_zones``) both parse: a polygon
authored on a reference frame. Same shape, different meaning downstream (drop from
analytics vs. mask ORB feature extraction) — this module owns only the shared
parsing, not the domain types each caller builds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tratrac.domain.geometry import Point2D, Polygon


def load_reference_frame_polygons[T](
	path: Path, *, array_key: str, build: Callable[[int, Polygon], T]
) -> tuple[T, ...]:
	"""Parse ``path``'s ``array_key`` array of ``{reference_frame, vertices}`` objects.

	``build(reference_frame, polygon)`` constructs the caller's own zone type from
	each parsed entry. Raises ``FileNotFoundError`` if ``path`` is absent and
	``ValueError`` on any malformed content (bad JSON, wrong shape, fewer than three
	vertices), each re-wrapped with the file path.
	"""
	try:
		with path.open("rb") as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc

	if not isinstance(document, dict) or array_key not in document:
		raise ValueError(f'{path} must be a JSON object with a "{array_key}" array.')
	raw_zones = document[array_key]
	if not isinstance(raw_zones, list):
		raise ValueError(f'{path}: "{array_key}" must be an array.')

	return tuple(_parse_zone(raw, index, path, build) for index, raw in enumerate(raw_zones))


def _parse_zone[T](raw: Any, index: int, path: Path, build: Callable[[int, Polygon], T]) -> T:
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
	return build(reference_frame, polygon)


def _parse_reference_frame(raw: Any, index: int, path: Path) -> int:
	if isinstance(raw, bool) or not isinstance(raw, int):
		raise ValueError(f"{path}: zone {index} reference_frame must be an integer.")
	if raw < 0:
		raise ValueError(f"{path}: zone {index} reference_frame must be >= 0.")
	return raw
