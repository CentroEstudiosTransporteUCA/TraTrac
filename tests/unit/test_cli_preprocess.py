"""Tests for ``tratrac-preprocess``'s pure helpers: the GSD scale one-of (``_resolve_scale``,
moved here from the old ``[calibration]`` config section) and canvas inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tratrac.calibration.drone_specs import lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.cli_preprocess import _infer_canvas, _resolve_scale
from tratrac.infrastructure.transform.sink import whole_canvas

_IMAGE_WIDTH = 1920


class TestResolveScale:
	def test_meters_per_pixel_method(self) -> None:
		scale = _resolve_scale(
			meters_per_pixel=0.05,
			drone_model=None,
			altitude_m=None,
			srt=None,
			image_width_pixels=_IMAGE_WIDTH,
		)
		assert scale == 0.05

	def test_drone_model_with_altitude_computes_gsd(self) -> None:
		scale = _resolve_scale(
			meters_per_pixel=None,
			drone_model="mavic_3",
			altitude_m=80.0,
			srt=None,
			image_width_pixels=_IMAGE_WIDTH,
		)
		spec = lookup("mavic_3")
		expected = ground_sample_distance(
			sensor_width_mm=spec.sensor_width_mm,
			focal_length_mm=spec.focal_length_mm,
			altitude_m=80.0,
			image_width_pixels=_IMAGE_WIDTH,
		)
		assert scale == pytest.approx(expected)

	def test_both_methods_is_an_error(self) -> None:
		with pytest.raises(ValueError, match="exactly one"):
			_resolve_scale(
				meters_per_pixel=0.05,
				drone_model="mavic_3",
				altitude_m=None,
				srt=None,
				image_width_pixels=_IMAGE_WIDTH,
			)

	def test_drone_model_without_altitude_source_is_an_error(self) -> None:
		with pytest.raises(ValueError, match="altitude"):
			_resolve_scale(
				meters_per_pixel=None,
				drone_model="mavic_3",
				altitude_m=None,
				srt=None,
				image_width_pixels=_IMAGE_WIDTH,
			)

	def test_unknown_drone_model_is_an_error(self) -> None:
		with pytest.raises(ValueError, match="unknown"):
			_resolve_scale(
				meters_per_pixel=None,
				drone_model="not_a_drone",
				altitude_m=80.0,
				srt=None,
				image_width_pixels=_IMAGE_WIDTH,
			)

	def test_non_positive_meters_per_pixel_is_an_error(self) -> None:
		with pytest.raises(ValueError, match="positive"):
			_resolve_scale(
				meters_per_pixel=0.0,
				drone_model=None,
				altitude_m=None,
				srt=None,
				image_width_pixels=_IMAGE_WIDTH,
			)

	def test_srt_path_as_altitude_source_resolves(self, tmp_path: Path) -> None:
		srt = tmp_path / "clip.SRT"
		srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n[rel_alt: 80.000 abs_alt: 100.0]\n\n")
		scale = _resolve_scale(
			meters_per_pixel=None,
			drone_model="mavic_3",
			altitude_m=None,
			srt=srt,
			image_width_pixels=_IMAGE_WIDTH,
		)
		assert scale > 0.0

	def test_neither_method_is_an_error(self) -> None:
		with pytest.raises(ValueError, match="set --meters-per-pixel"):
			_resolve_scale(
				meters_per_pixel=None,
				drone_model=None,
				altitude_m=None,
				srt=None,
				image_width_pixels=_IMAGE_WIDTH,
			)


class TestInferCanvas:
	def test_recovers_width_and_height_from_a_whole_canvas_zone(self) -> None:
		zone = whole_canvas(1920, 1080)
		assert _infer_canvas(zone) == (1920, 1080)
