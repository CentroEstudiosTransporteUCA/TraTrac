"""Tests for the calibration package: drone specs, GSD, SRT parsing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tratrac.calibration.drone_specs import DroneSpec, known_models, lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.calibration.srt_parser import extract_altitudes, mean_altitude


class TestDroneSpec:
	def test_rejects_non_positive_sensor(self) -> None:
		with pytest.raises(ValueError, match="sensor_width_mm"):
			DroneSpec(sensor_width_mm=0.0, focal_length_mm=12.0)

	def test_rejects_non_positive_focal(self) -> None:
		with pytest.raises(ValueError, match="focal_length_mm"):
			DroneSpec(sensor_width_mm=17.0, focal_length_mm=-1.0)


class TestRegistry:
	def test_known_mavic_3(self) -> None:
		spec = lookup("mavic_3")
		assert math.isclose(spec.sensor_width_mm, 17.3)

	def test_is_case_insensitive(self) -> None:
		assert lookup("MAVIC_3") == lookup("mavic_3")

	def test_unknown_model_lists_alternatives(self) -> None:
		with pytest.raises(KeyError, match="mavic_3"):
			lookup("not_a_drone")

	def test_known_models_sorted(self) -> None:
		models = known_models()
		assert models == sorted(models)
		assert "mavic_3" in models


class TestGsd:
	def test_known_example_mavic_3_at_50m(self) -> None:
		# Mavic 3, 50 m AGL, 1920 px wide image.
		gsd = ground_sample_distance(
			sensor_width_mm=17.3,
			focal_length_mm=12.29,
			altitude_m=50.0,
			image_width_pixels=1920,
		)
		# (17.3 * 50) / (12.29 * 1920) = 865 / 23596.8 ~= 0.03666
		assert math.isclose(gsd, 0.03666, abs_tol=1e-4)

	def test_doubling_altitude_doubles_gsd(self) -> None:
		base = ground_sample_distance(
			sensor_width_mm=17.3,
			focal_length_mm=12.29,
			altitude_m=50.0,
			image_width_pixels=1920,
		)
		doubled = ground_sample_distance(
			sensor_width_mm=17.3,
			focal_length_mm=12.29,
			altitude_m=100.0,
			image_width_pixels=1920,
		)
		assert math.isclose(doubled, 2.0 * base)

	def test_rejects_non_positive_inputs(self) -> None:
		base_kwargs = {
			"sensor_width_mm": 17.3,
			"focal_length_mm": 12.29,
			"altitude_m": 50.0,
			"image_width_pixels": 1920,
		}
		for key in base_kwargs:
			kwargs = dict(base_kwargs)
			kwargs[key] = 0
			with pytest.raises(ValueError, match=key):
				ground_sample_distance(**kwargs)  # type: ignore[arg-type]


_SRT_SAMPLE = """\
1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2024-05-21 14:30:00,123,456
[iso: 100] [shutter: 1/2000.0] [fnum: 2.8] [ev: 0] [focal_len: 12.29] [latitude: 36.123456] [longitude: -123.456789] [rel_alt: 50.000 abs_alt: 100.500]
</font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
2024-05-21 14:30:00,156,789
[iso: 100] [shutter: 1/2000.0] [fnum: 2.8] [ev: 0] [focal_len: 12.29] [latitude: 36.123456] [longitude: -123.456789] [rel_alt: 51.000 abs_alt: 101.500]
</font>

3
00:00:00,066 --> 00:00:00,099
<font size="28">FrameCnt: 3, DiffTime: 33ms
2024-05-21 14:30:00,189,012
[iso: 100] [shutter: 1/2000.0] [fnum: 2.8] [ev: 0] [focal_len: 12.29] [latitude: 36.123456] [longitude: -123.456789] [rel_alt: 49.000 abs_alt: 99.500]
</font>
"""


class TestSrtParser:
	def _write_srt(self, tmp_path: Path, content: str) -> Path:
		path = tmp_path / "sample.SRT"
		path.write_text(content, encoding="utf-8")
		return path

	def test_extract_three_altitudes(self, tmp_path: Path) -> None:
		path = self._write_srt(tmp_path, _SRT_SAMPLE)
		assert extract_altitudes(path) == [50.0, 51.0, 49.0]

	def test_mean_altitude(self, tmp_path: Path) -> None:
		path = self._write_srt(tmp_path, _SRT_SAMPLE)
		assert math.isclose(mean_altitude(path), 50.0)

	def test_falls_back_to_abs_alt_when_rel_alt_missing(self, tmp_path: Path) -> None:
		srt = "[abs_alt: 88.5]\n[abs_alt: 89.5]\n"
		path = self._write_srt(tmp_path, srt)
		assert extract_altitudes(path) == [88.5, 89.5]

	def test_empty_srt_raises(self, tmp_path: Path) -> None:
		path = self._write_srt(tmp_path, "")
		with pytest.raises(ValueError, match="No rel_alt"):
			mean_altitude(path)

	def test_missing_file_raises(self, tmp_path: Path) -> None:
		with pytest.raises(FileNotFoundError):
			extract_altitudes(tmp_path / "missing.SRT")
