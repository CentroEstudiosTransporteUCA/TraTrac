"""Tests for image-space exclusion zones: Polygon overlap, ExclusionZones, the
MaskingDetector decorator, and the sidecar-JSON loader. All pure — no cv2, no
model downloads. See vault/21_exclusion_zones.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import BoundingBox, Point2D, Polygon
from tratrac.infrastructure.detection.masking import MaskingDetector
from tratrac.infrastructure.exclusion.json import load_exclusion_zones


def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
	return Polygon(vertices=(Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1)))


_PIXELS = np.zeros((1, 1, 3), dtype=np.uint8)  # MaskingDetector never reads pixels


def _detection(x: float, y: float, width: float, height: float) -> Detection:
	return Detection(
		bbox=BoundingBox(x=x, y=y, width=width, height=height),
		score=0.9,
		vehicle_class=VehicleClass.CAR,
	)


class TestPolygonOverlap:
	def test_rejects_fewer_than_three_vertices(self) -> None:
		with pytest.raises(ValueError, match="at least 3"):
			Polygon(vertices=(Point2D(0, 0), Point2D(1, 1)))

	def test_box_fully_inside_polygon_is_full_overlap(self) -> None:
		zone = _square(0, 0, 100, 100)
		assert zone.overlap_fraction(BoundingBox(x=10, y=10, width=20, height=20)) == pytest.approx(
			1.0
		)

	def test_box_fully_outside_polygon_is_zero(self) -> None:
		zone = _square(0, 0, 10, 10)
		assert zone.overlap_fraction(BoundingBox(x=50, y=50, width=10, height=10)) == 0.0

	def test_box_half_covered_is_one_half(self) -> None:
		# Zone covers the left half (x in [0, 50]) of a box spanning x in [0, 100].
		zone = _square(0, 0, 50, 100)
		frac = zone.overlap_fraction(BoundingBox(x=0, y=0, width=100, height=100))
		assert frac == pytest.approx(0.5)

	def test_partial_corner_overlap(self) -> None:
		# Zone covers the top-left quarter of the box.
		zone = _square(0, 0, 50, 50)
		frac = zone.overlap_fraction(BoundingBox(x=0, y=0, width=100, height=100))
		assert frac == pytest.approx(0.25)


class TestExclusionZones:
	def test_empty_excludes_nothing(self) -> None:
		assert (
			ExclusionZones(zones=()).excludes(BoundingBox(x=0, y=0, width=10, height=10)) is False
		)

	def test_majority_inside_is_excluded(self) -> None:
		zones = ExclusionZones(zones=(_square(0, 0, 80, 100),))
		# 80% of the box is inside -> majority -> excluded.
		assert zones.excludes(BoundingBox(x=0, y=0, width=100, height=100)) is True

	def test_exactly_half_is_not_excluded(self) -> None:
		# Strictly greater than half is the rule; exactly 50% survives.
		zones = ExclusionZones(zones=(_square(0, 0, 50, 100),))
		assert zones.excludes(BoundingBox(x=0, y=0, width=100, height=100)) is False

	def test_max_not_union_across_zones(self) -> None:
		# Two adjacent zones each cover 30% of the box (union 60%), but per-zone max
		# is 30% < 50%, so the box survives — the documented union-vs-max caveat.
		zones = ExclusionZones(zones=(_square(0, 0, 30, 100), _square(70, 0, 100, 100)))
		assert zones.excludes(BoundingBox(x=0, y=0, width=100, height=100)) is False


class _FakeDetector:
	"""A Detector stub returning a fixed list, no model download."""

	def __init__(self, detections: list[Detection]) -> None:
		self._detections = detections

	def detect(self, frame: Frame) -> list[Detection]:
		return self._detections


class TestMaskingDetector:
	def test_drops_majority_overlap_keeps_rest(self) -> None:
		inside = _detection(0, 0, 100, 100)  # 80% inside the zone below
		outside = _detection(500, 500, 20, 20)
		inner = _FakeDetector([inside, outside])
		zones = ExclusionZones(zones=(_square(0, 0, 80, 100),))

		kept = MaskingDetector(inner, zones).detect(Frame(index=0, pixels=_PIXELS))

		assert kept == [outside]

	def test_empty_zones_pass_everything_through(self) -> None:
		dets = [_detection(0, 0, 10, 10), _detection(20, 20, 10, 10)]
		inner = _FakeDetector(dets)
		kept = MaskingDetector(inner, ExclusionZones(zones=())).detect(
			Frame(index=0, pixels=_PIXELS)
		)
		assert kept == dets


class TestLoadExclusionZones:
	def test_reads_polygons(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps(
				{
					"exclusion_zones": [
						{"label": "lot", "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]]}
					]
				}
			)
		)
		zones = load_exclusion_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].vertices[0] == Point2D(0.0, 0.0)

	def test_empty_list_is_legal(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"exclusion_zones": []}))
		assert load_exclusion_zones(path).zones == ()

	def test_malformed_json_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text("{not json")
		with pytest.raises(ValueError, match="not valid JSON"):
			load_exclusion_zones(path)

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"polygons": []}))
		with pytest.raises(ValueError, match="exclusion_zones"):
			load_exclusion_zones(path)

	def test_too_few_vertices_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"exclusion_zones": [{"vertices": [[0, 0], [1, 1]]}]}))
		with pytest.raises(ValueError, match="at least 3"):
			load_exclusion_zones(path)

	def test_non_pair_vertex_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps({"exclusion_zones": [{"vertices": [[0, 0, 0], [1, 1], [2, 2]]}]})
		)
		with pytest.raises(ValueError, match="number pairs"):
			load_exclusion_zones(path)
