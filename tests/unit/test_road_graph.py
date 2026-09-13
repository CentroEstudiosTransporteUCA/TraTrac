"""Tests for road-topology zones: the zone->global conversion, per-point classification, and
the sidecar-JSON loaders. Pure stdlib — no cv2/model downloads. See docs/IMPLEMENTATION_PLAN.md
Groups C1/C2/C5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tratrac.application.coordinate_transforms import TransformTable
from tratrac.application.road_graph import (
	lane_id_for_point,
	link_id_for_point,
	plane_id_for_point,
	to_global_lane_polygons,
	to_global_link_polygons,
	to_global_plane_polygons,
)
from tratrac.domain.geometry import Point2D, Polygon, Transform2D
from tratrac.domain.road_graph import (
	LaneZone,
	LaneZones,
	LinkZone,
	LinkZones,
	PlaneZone,
	PlaneZones,
)
from tratrac.infrastructure.road_graph.json import (
	load_lane_zones,
	load_link_zones,
	load_plane_zones,
)
from tratrac.infrastructure.transform.records import SimilarityFunction, TransformRow


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


# Strictly contains every test point below (point_in_polygon excludes the boundary,
# so this can't start flush at 0,0 the way a real whole-canvas zone would).
_WIDE_ZONE = _square(-1000.0, -1000.0, 1000.0, 1000.0)


class TestLinkZone:
	def test_rejects_non_positive_link_id(self) -> None:
		with pytest.raises(ValueError, match="link_id"):
			LinkZone(link_id=0, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))


class TestToGlobalLinkPolygons:
	def test_identity_pose_keeps_raw_coordinates(self) -> None:
		zones = LinkZones(
			zones=(LinkZone(link_id=1, reference_frame=0, polygon=Polygon(_square(1, 2, 3, 4))),)
		)
		out = to_global_link_polygons(zones, TransformTable([]))
		assert out == ((1, _square(1, 2, 3, 4)),)

	def test_pose_lookup_is_keyed_by_reference_frame(self) -> None:
		zones = LinkZones(
			zones=(LinkZone(link_id=5, reference_frame=7, polygon=Polygon(_square(0, 0, 10, 10))),)
		)
		shift = Transform2D(a=1.0, b=0.0, tx=5.0, c=0.0, d=1.0, ty=0.0)
		table = TransformTable([TransformRow(7, _WIDE_ZONE, SimilarityFunction(shift))])
		out = to_global_link_polygons(zones, table)
		assert out[0][0] == 5
		assert out[0][1][0] == Point2D(5.0, 0.0)


class TestLinkIdForPoint:
	_ZONES = ((1, _square(0.0, 0.0, 100.0, 100.0)), (2, _square(200.0, 200.0, 300.0, 300.0)))

	def test_point_inside_a_zone_gets_its_link_id(self) -> None:
		assert link_id_for_point(Point2D(50.0, 50.0), self._ZONES) == 1
		assert link_id_for_point(Point2D(250.0, 250.0), self._ZONES) == 2

	def test_point_outside_every_zone_is_unassigned(self) -> None:
		assert link_id_for_point(Point2D(500.0, 500.0), self._ZONES) == 0

	def test_no_zones_is_always_unassigned(self) -> None:
		assert link_id_for_point(Point2D(1.0, 1.0), ()) == 0

	def test_first_match_wins_on_overlap(self) -> None:
		overlapping = (
			(1, _square(0.0, 0.0, 100.0, 100.0)),
			(2, _square(50.0, 50.0, 150.0, 150.0)),
		)
		assert link_id_for_point(Point2D(75.0, 75.0), overlapping) == 1


class TestLoadLinkZones:
	def test_reads_polygons_with_link_id_and_reference_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(
			json.dumps(
				{
					"link_zones": [
						{
							"link_id": 3,
							"reference_frame": 12,
							"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
						}
					]
				}
			)
		)
		zones = load_link_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].link_id == 3
		assert zones.zones[0].reference_frame == 12
		assert zones.zones[0].polygon.vertices[0] == Point2D(0.0, 0.0)

	def test_reference_frame_defaults_to_zero(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(
			json.dumps({"link_zones": [{"link_id": 1, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)
		assert load_link_zones(path).zones[0].reference_frame == 0

	def test_empty_list_is_legal(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(json.dumps({"link_zones": []}))
		assert load_link_zones(path).zones == ()

	def test_malformed_json_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text("{not json")
		with pytest.raises(ValueError, match="not valid JSON"):
			load_link_zones(path)

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(json.dumps({"polygons": []}))
		with pytest.raises(ValueError, match="link_zones"):
			load_link_zones(path)

	def test_missing_link_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(json.dumps({"link_zones": [{"vertices": [[0, 0], [1, 0], [0, 1]]}]}))
		with pytest.raises(ValueError, match="link_id"):
			load_link_zones(path)

	def test_non_positive_link_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(
			json.dumps({"link_zones": [{"link_id": 0, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)
		with pytest.raises(ValueError, match="link_id"):
			load_link_zones(path)

	def test_too_few_vertices_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(json.dumps({"link_zones": [{"link_id": 1, "vertices": [[0, 0], [1, 1]]}]}))
		with pytest.raises(ValueError, match="at least 3"):
			load_link_zones(path)

	def test_non_pair_vertex_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(
			json.dumps({"link_zones": [{"link_id": 1, "vertices": [[0, 0, 0], [1, 1], [2, 2]]}]})
		)
		with pytest.raises(ValueError, match="number pairs"):
			load_link_zones(path)

	def test_negative_reference_frame_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "links.json"
		path.write_text(
			json.dumps(
				{
					"link_zones": [
						{"link_id": 1, "reference_frame": -1, "vertices": [[0, 0], [1, 0], [0, 1]]}
					]
				}
			)
		)
		with pytest.raises(ValueError, match="reference_frame"):
			load_link_zones(path)


class TestLaneZone:
	def test_rejects_non_positive_lane_id(self) -> None:
		with pytest.raises(ValueError, match="lane_id"):
			LaneZone(link_id=1, lane_id=0, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))

	def test_rejects_lane_id_above_byte_range(self) -> None:
		with pytest.raises(ValueError, match="lane_id"):
			LaneZone(
				link_id=1, lane_id=256, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1))
			)

	def test_rejects_non_positive_link_id(self) -> None:
		with pytest.raises(ValueError, match="link_id"):
			LaneZone(link_id=0, lane_id=1, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))


class TestToGlobalLanePolygons:
	def test_identity_pose_keeps_raw_coordinates(self) -> None:
		zones = LaneZones(
			zones=(
				LaneZone(
					link_id=1, lane_id=2, reference_frame=0, polygon=Polygon(_square(1, 2, 3, 4))
				),
			)
		)
		out = to_global_lane_polygons(zones, TransformTable([]))
		assert out == ((2, _square(1, 2, 3, 4)),)


class TestLaneIdForPoint:
	_ZONES = ((1, _square(0.0, 0.0, 100.0, 100.0)), (2, _square(200.0, 200.0, 300.0, 300.0)))

	def test_point_inside_a_zone_gets_its_lane_id(self) -> None:
		assert lane_id_for_point(Point2D(50.0, 50.0), self._ZONES) == 1
		assert lane_id_for_point(Point2D(250.0, 250.0), self._ZONES) == 2

	def test_point_outside_every_zone_is_unassigned(self) -> None:
		assert lane_id_for_point(Point2D(500.0, 500.0), self._ZONES) == 0


class TestLoadLaneZones:
	def test_reads_polygons_with_link_lane_and_reference_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "lanes.json"
		path.write_text(
			json.dumps(
				{
					"lane_zones": [
						{
							"link_id": 1,
							"lane_id": 2,
							"reference_frame": 12,
							"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
						}
					]
				}
			)
		)
		zones = load_lane_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].link_id == 1
		assert zones.zones[0].lane_id == 2
		assert zones.zones[0].reference_frame == 12

	def test_missing_lane_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "lanes.json"
		path.write_text(
			json.dumps({"lane_zones": [{"link_id": 1, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)
		with pytest.raises(ValueError, match="lane_id"):
			load_lane_zones(path)

	def test_non_positive_lane_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "lanes.json"
		path.write_text(
			json.dumps(
				{"lane_zones": [{"link_id": 1, "lane_id": 0, "vertices": [[0, 0], [1, 0], [0, 1]]}]}
			)
		)
		with pytest.raises(ValueError, match="lane_id"):
			load_lane_zones(path)

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "lanes.json"
		path.write_text(json.dumps({"polygons": []}))
		with pytest.raises(ValueError, match="lane_zones"):
			load_lane_zones(path)

	def test_empty_list_is_legal(self, tmp_path: Path) -> None:
		path = tmp_path / "lanes.json"
		path.write_text(json.dumps({"lane_zones": []}))
		assert load_lane_zones(path).zones == ()


class TestPlaneZone:
	def test_zero_plane_id_is_legal(self) -> None:
		# Unlike Link/Lane, 0 is not a reserved sentinel here (e.g. the ground plane).
		zone = PlaneZone(plane_id=0, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))
		assert zone.plane_id == 0

	def test_rejects_negative_plane_id(self) -> None:
		with pytest.raises(ValueError, match="plane_id"):
			PlaneZone(plane_id=-1, reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))


class TestToGlobalPlanePolygons:
	def test_identity_pose_keeps_raw_coordinates(self) -> None:
		zones = PlaneZones(
			zones=(PlaneZone(plane_id=1, reference_frame=0, polygon=Polygon(_square(1, 2, 3, 4))),)
		)
		out = to_global_plane_polygons(zones, TransformTable([]))
		assert out == ((1, _square(1, 2, 3, 4)),)


class TestPlaneIdForPoint:
	_ZONES = ((1, _square(0.0, 0.0, 100.0, 100.0)), (2, _square(200.0, 200.0, 300.0, 300.0)))

	def test_point_inside_a_zone_gets_its_plane_id(self) -> None:
		assert plane_id_for_point(Point2D(50.0, 50.0), self._ZONES) == 1
		assert plane_id_for_point(Point2D(250.0, 250.0), self._ZONES) == 2

	def test_point_outside_every_zone_defaults_to_zero(self) -> None:
		assert plane_id_for_point(Point2D(500.0, 500.0), self._ZONES) == 0


class TestLoadPlaneZones:
	def test_reads_polygons_with_plane_id_and_reference_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(
			json.dumps(
				{
					"plane_zones": [
						{
							"plane_id": 1,
							"reference_frame": 12,
							"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
						}
					]
				}
			)
		)
		zones = load_plane_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].plane_id == 1
		assert zones.zones[0].reference_frame == 12

	def test_zero_plane_id_round_trips(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(
			json.dumps({"plane_zones": [{"plane_id": 0, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)
		assert load_plane_zones(path).zones[0].plane_id == 0

	def test_missing_plane_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(json.dumps({"plane_zones": [{"vertices": [[0, 0], [1, 0], [0, 1]]}]}))
		with pytest.raises(ValueError, match="plane_id"):
			load_plane_zones(path)

	def test_negative_plane_id_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(
			json.dumps({"plane_zones": [{"plane_id": -1, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)
		with pytest.raises(ValueError, match="plane_id"):
			load_plane_zones(path)

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(json.dumps({"polygons": []}))
		with pytest.raises(ValueError, match="plane_zones"):
			load_plane_zones(path)

	def test_empty_list_is_legal(self, tmp_path: Path) -> None:
		path = tmp_path / "planes.json"
		path.write_text(json.dumps({"plane_zones": []}))
		assert load_plane_zones(path).zones == ()
