"""Sidecar JSON reader + homography fit for world projection (MVP2, vault/06_mvp2.md).

``load_calibration`` parses a per-scene ``calibration.json`` of image↔world ground
correspondences into the pure ``Calibration`` value object (mirrors
``infrastructure/exclusion/json.py``). ``compute_homography`` fits the 3x3 image→world
homography from a set of point pairs via cv2 (exact for 4, RANSAC least-squares for more).

Schema::

    { "correspondences": [
        { "reference_frame": 0, "image": [u, v], "world": [x_m, y_m] }
    ] }

``reference_frame`` is the frame the pixel coordinates are drawn on (optional, defaults to
``0`` for a static camera; for a moving drone, an exported anchor frame index). ``image``
is a pixel pair, ``world`` a real ground pair in metres. At least 4 correspondences are
required to determine a homography.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tratrac.domain.geometry import Point2D
from tratrac.domain.world import Calibration, Correspondence

# A homography has 8 DOF; 4 point pairs (8 equations) determine it exactly.
_MIN_CORRESPONDENCES = 4


def load_calibration(path: Path) -> Calibration:
	"""Parse a ``calibration.json`` into a ``Calibration``.

	Raises ``FileNotFoundError`` if absent and ``ValueError`` on any malformed content
	(bad JSON, wrong shape, fewer than four correspondences), re-wrapped with the path so
	the CLI can report it cleanly.
	"""
	try:
		with path.open("rb") as handle:
			document: Any = json.load(handle)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{path} is not valid JSON: {exc}") from exc

	if not isinstance(document, dict) or "correspondences" not in document:
		raise ValueError(f'{path} must be a JSON object with a "correspondences" array.')
	raw = document["correspondences"]
	if not isinstance(raw, list):
		raise ValueError(f'{path}: "correspondences" must be an array.')
	if len(raw) < _MIN_CORRESPONDENCES:
		raise ValueError(
			f"{path}: need at least {_MIN_CORRESPONDENCES} correspondences to fit a homography, "
			f"got {len(raw)}."
		)
	correspondences = tuple(
		_parse_correspondence(item, index, path) for index, item in enumerate(raw)
	)
	return Calibration(correspondences=correspondences)


def _parse_correspondence(raw: Any, index: int, path: Path) -> Correspondence:
	if not isinstance(raw, dict) or "image" not in raw or "world" not in raw:
		raise ValueError(
			f'{path}: correspondence {index} must be an object with "image" and "world" pairs.'
		)
	reference_frame = _parse_reference_frame(raw.get("reference_frame", 0), index, path)
	image = _parse_pair(raw["image"], index, "image", path)
	world = _parse_pair(raw["world"], index, "world", path)
	return Correspondence(reference_frame=reference_frame, image=image, world=world)


def _parse_pair(raw: Any, index: int, field: str, path: Path) -> Point2D:
	if (
		not isinstance(raw, list | tuple)
		or len(raw) != 2
		or not all(isinstance(c, int | float) and not isinstance(c, bool) for c in raw)
	):
		raise ValueError(f"{path}: correspondence {index} {field} must be an [x, y] number pair.")
	return Point2D(float(raw[0]), float(raw[1]))


def _parse_reference_frame(raw: Any, index: int, path: Path) -> int:
	if isinstance(raw, bool) or not isinstance(raw, int):
		raise ValueError(f"{path}: correspondence {index} reference_frame must be an integer.")
	if raw < 0:
		raise ValueError(f"{path}: correspondence {index} reference_frame must be >= 0.")
	return raw


def compute_homography(
	image_points: list[Point2D], world_points: list[Point2D]
) -> NDArray[np.float64]:
	"""Fit the 3x3 homography mapping ``image_points`` to ``world_points`` (ground plane).

	Exact (``getPerspectiveTransform``) for 4 pairs; RANSAC least-squares
	(``findHomography``) for more, which averages out point-picking error. The RANSAC path
	raises ``ValueError`` when no consensus homography can be fit (degenerate / collinear
	correspondences -> cv2 returns ``None``). The exact 4-point path does **not** detect
	degeneracy: cv2 returns a finite-but-meaningless matrix for collinear points, so four
	well-spread, non-collinear correspondences are the operator's responsibility.
	"""
	if len(image_points) != len(world_points):
		raise ValueError("image_points and world_points must have equal length.")
	if len(image_points) < _MIN_CORRESPONDENCES:
		raise ValueError(f"need at least {_MIN_CORRESPONDENCES} point pairs to fit a homography.")
	src = np.array([[p.x, p.y] for p in image_points], dtype=np.float64)
	dst = np.array([[p.x, p.y] for p in world_points], dtype=np.float64)
	if len(image_points) == _MIN_CORRESPONDENCES:
		matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
	else:
		matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
	if matrix is None:
		raise ValueError("could not fit a homography (degenerate or collinear correspondences).")
	return np.asarray(matrix, dtype=np.float64)
