"""Tests for image-space exclusion zones: the rasterized RasterExclusionMask, the
zone->global conversion, and the sidecar-JSON loader. cv2/numpy only — no model
downloads. See vault/21_exclusion_zones.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tratrac.application.exclusion import to_global_polygons
from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.exclusion import ExclusionZone, ExclusionZones
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import BoundingBox, Point2D, Polygon, Transform2D
from tratrac.infrastructure.exclusion.json import load_exclusion_zones
from tratrac.infrastructure.exclusion.raster import RasterExclusionMask


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


def _detection(x: float, y: float, width: float, height: float) -> Detection:
	return Detection(
		bbox=BoundingBox(x=x, y=y, width=width, height=height),
		score=0.9,
		vehicle_class=VehicleClass.CAR,
	)


def _frame(width: int = 200, height: int = 200) -> Frame:
	return Frame(index=0, pixels=np.zeros((height, width, 3), dtype=np.uint8))


def _identity_mask(*polygons: tuple[Point2D, ...]) -> RasterExclusionMask:
	return RasterExclusionMask(tuple(polygons))


class TestRasterExclusionMask:
	def test_no_zones_passes_everything(self) -> None:
		mask = RasterExclusionMask(())
		dets = [_detection(0, 0, 10, 10)]
		assert mask.filter(dets, Transform2D.identity(), _frame()) == dets

	def test_majority_inside_is_dropped(self) -> None:
		# Zone covers x in [0, 80]; the box spans [0, 100] -> 80% covered -> dropped.
		mask = _identity_mask(_square(0, 0, 80, 100))
		kept = mask.filter([_detection(0, 0, 100, 100)], Transform2D.identity(), _frame())
		assert kept == []

	def test_minority_inside_is_kept(self) -> None:
		# Zone covers x in [0, 30] -> 30% of the box -> survives.
		mask = _identity_mask(_square(0, 0, 30, 100))
		box = _detection(0, 0, 100, 100)
		assert mask.filter([box], Transform2D.identity(), _frame()) == [box]

	def test_concave_zone_is_handled(self) -> None:
		# An L / arrow-shaped concave polygon almost entirely covering the box: an
		# analytic convex clip would mis-area it; the raster fills it correctly.
		concave = (
			Point2D(0, 0),
			Point2D(100, 0),
			Point2D(100, 100),
			Point2D(0, 100),
			Point2D(0, 60),
			Point2D(60, 60),
			Point2D(60, 40),
			Point2D(0, 40),
		)
		mask = _identity_mask(concave)
		# Box sitting in the filled lower-left arm -> mostly covered -> dropped.
		kept = mask.filter([_detection(0, 70, 50, 25)], Transform2D.identity(), _frame())
		assert kept == []

	def test_union_of_two_zones_drops_a_straddling_box(self) -> None:
		# The limitation-2 fix: two disjoint zones covering [0,30] and [60,100] of the
		# box (union 70%). The old per-zone max (40%) kept it; the rasterized union drops it.
		mask = _identity_mask(_square(0, 0, 30, 100), _square(60, 0, 100, 100))
		kept = mask.filter([_detection(0, 0, 100, 100)], Transform2D.identity(), _frame())
		assert kept == []

	def test_translation_pose_moves_the_effective_zone(self) -> None:
		# Zone authored in global coords as [0,40]x[0,200]. A pose translating the raw
		# frame by +100 in x (raw -> global) means global x in [0,40] maps to raw
		# x in [-100,-60] -> off-frame, so a box at raw x~0 is NOT covered.
		zone = _square(0, 0, 40, 200)
		mask = _identity_mask(zone)
		pose = Transform2D(a=1.0, b=0.0, tx=100.0, c=0.0, d=1.0, ty=0.0)
		box = _detection(0, 0, 30, 200)
		assert mask.filter([box], pose, _frame()) == [box]
		# Under identity, the same zone covers the box -> dropped.
		assert mask.filter([box], Transform2D.identity(), _frame()) == []


class TestToGlobalPolygons:
	def test_identity_pose_keeps_raw_coordinates(self) -> None:
		zones = ExclusionZones(
			zones=(ExclusionZone(reference_frame=0, polygon=Polygon(_square(1, 2, 3, 4))),)
		)
		out = to_global_polygons(zones, lambda _f: Transform2D.identity())
		assert out == (_square(1, 2, 3, 4),)

	def test_pose_lookup_is_keyed_by_reference_frame(self) -> None:
		zones = ExclusionZones(
			zones=(ExclusionZone(reference_frame=7, polygon=Polygon(_square(0, 0, 10, 10))),)
		)
		shift = Transform2D(a=1.0, b=0.0, tx=5.0, c=0.0, d=1.0, ty=0.0)
		out = to_global_polygons(zones, lambda f: shift if f == 7 else Transform2D.identity())
		assert out[0][0] == Point2D(5.0, 0.0)  # (0,0) shifted by +5 in x


class TestLoadExclusionZones:
	def test_reads_polygons_with_reference_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps(
				{
					"exclusion_zones": [
						{
							"label": "lot",
							"reference_frame": 12,
							"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
						}
					]
				}
			)
		)
		zones = load_exclusion_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].reference_frame == 12
		assert zones.zones[0].polygon.vertices[0] == Point2D(0.0, 0.0)

	def test_reference_frame_defaults_to_zero(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"exclusion_zones": [{"vertices": [[0, 0], [1, 0], [0, 1]]}]}))
		assert load_exclusion_zones(path).zones[0].reference_frame == 0

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

	def test_negative_reference_frame_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps(
				{"exclusion_zones": [{"reference_frame": -1, "vertices": [[0, 0], [1, 0], [0, 1]]}]}
			)
		)
		with pytest.raises(ValueError, match="reference_frame"):
			load_exclusion_zones(path)
