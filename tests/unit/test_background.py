"""Tests for background zones: the domain type and the sidecar-JSON loader.

Pure stdlib — no cv2/model downloads. See infrastructure/video/ego_motion_orb.py's
module docstring, "Detector-free ego-motion" section.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tratrac.domain.background import BackgroundZone, BackgroundZones
from tratrac.domain.geometry import Point2D, Polygon
from tratrac.infrastructure.background.json import load_background_zones


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


class TestBackgroundZones:
	def test_rejects_an_empty_collection(self) -> None:
		with pytest.raises(ValueError, match="at least one zone"):
			BackgroundZones(zones=())

	def test_accepts_at_least_one_zone(self) -> None:
		zone = BackgroundZone(reference_frame=0, polygon=Polygon(_square(0, 0, 1, 1)))
		assert BackgroundZones(zones=(zone,)).zones == (zone,)


class TestLoadBackgroundZones:
	def test_reads_polygons_with_reference_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps(
				{
					"background_zones": [
						{"reference_frame": 850, "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]]}
					]
				}
			)
		)
		zones = load_background_zones(path)
		assert len(zones.zones) == 1
		assert zones.zones[0].reference_frame == 850
		assert zones.zones[0].polygon.vertices[0] == Point2D(0.0, 0.0)

	def test_reference_frame_defaults_to_zero(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"background_zones": [{"vertices": [[0, 0], [1, 0], [0, 1]]}]}))
		assert load_background_zones(path).zones[0].reference_frame == 0

	def test_multiple_zones_ordered_by_authoring_not_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(
			json.dumps(
				{
					"background_zones": [
						{"reference_frame": 850, "vertices": [[0, 0], [1, 0], [0, 1]]},
						{"reference_frame": 0, "vertices": [[2, 2], [3, 2], [2, 3]]},
					]
				}
			)
		)
		zones = load_background_zones(path)
		assert [z.reference_frame for z in zones.zones] == [850, 0]

	def test_empty_list_is_rejected(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"background_zones": []}))
		with pytest.raises(ValueError, match="at least one zone"):
			load_background_zones(path)

	def test_malformed_json_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text("{not json")
		with pytest.raises(ValueError, match="not valid JSON"):
			load_background_zones(path)

	def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"polygons": []}))
		with pytest.raises(ValueError, match="background_zones"):
			load_background_zones(path)

	def test_too_few_vertices_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "zones.json"
		path.write_text(json.dumps({"background_zones": [{"vertices": [[0, 0], [1, 1]]}]}))
		with pytest.raises(ValueError, match="at least 3"):
			load_background_zones(path)
