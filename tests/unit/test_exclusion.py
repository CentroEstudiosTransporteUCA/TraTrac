"""Tests for exclusion zones: the zone->global conversion, the track-aware filter, and the
sidecar-JSON loader. Pure stdlib — no cv2/model downloads. See vault/21_exclusion_zones.md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tratrac.application.exclusion import excluded_track_ids, to_global_polygons
from tratrac.domain.exclusion import ExclusionZone, ExclusionZones
from tratrac.domain.geometry import Point2D, Polygon, Transform2D
from tratrac.infrastructure.exclusion.json import load_exclusion_zones


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


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


class TestExcludedTrackIds:
	_ZONE = (_square(0.0, 0.0, 100.0, 100.0),)

	def test_track_mostly_inside_is_excluded(self) -> None:
		centroids = [
			(1, Point2D(50.0, 50.0)),  # track 1: 2/3 inside -> dropped
			(1, Point2D(60.0, 60.0)),
			(1, Point2D(200.0, 200.0)),
			(2, Point2D(300.0, 300.0)),  # track 2: 0/2 inside -> kept
			(2, Point2D(310.0, 310.0)),
		]
		assert excluded_track_ids(centroids, self._ZONE, min_fraction=0.5) == {1}

	def test_below_threshold_is_kept(self) -> None:
		centroids = [
			(1, Point2D(50.0, 50.0)),  # 1/3 inside < 0.5
			(1, Point2D(200.0, 200.0)),
			(1, Point2D(210.0, 210.0)),
		]
		assert excluded_track_ids(centroids, self._ZONE, min_fraction=0.5) == set()

	def test_threshold_is_inclusive(self) -> None:
		centroids = [(1, Point2D(50.0, 50.0)), (1, Point2D(200.0, 200.0))]  # exactly 1/2
		assert excluded_track_ids(centroids, self._ZONE, min_fraction=0.5) == {1}

	def test_no_polygons_excludes_nothing(self) -> None:
		assert excluded_track_ids([(1, Point2D(50.0, 50.0))], (), min_fraction=0.5) == set()


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
