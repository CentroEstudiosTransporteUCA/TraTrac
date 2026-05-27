"""Tests for run-config resolution and the TOML loader.

``RunConfig.resolve`` is pure and takes plain mappings, so most tests build the
config as a dict — no temp files needed. The TOML loader is tested separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tratrac.application.config import (
	ConfigError,
	DetectorChoice,
	RunConfig,
)
from tratrac.calibration.drone_specs import lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.domain.frame import VideoMetadata
from tratrac.infrastructure.config.toml import load_toml

_METADATA = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=100)


def _complete(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
	"""A complete, valid file-values mapping; ``overrides`` patch whole sections."""
	file_values: dict[str, Any] = {
		"input": {"video": str(tmp_path / "v.mp4")},
		"detector": {
			"name": "yolov8_visdrone",
			"checkpoint": "repo/model",
			"conf": 0.25,
			"filename": "model.pt",
		},
		"runtime": {"device": "cpu"},
		"calibration": {"meters_per_pixel": 0.05},
		"tracker": {"det_thresh": 0.1},
		"orientation": {"smoothing_window": 5},
		"export": {"out": str(tmp_path / "out.trj"), "timestep_precision": 0.0},
		"window": {"start": "", "end": ""},
		"run": {"force": False, "timing_csv": ""},
	}
	file_values.update(overrides)
	return file_values


class TestResolveComplete:
	def test_complete_file_resolves_with_no_overrides(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path), {})
		assert run.input.video == tmp_path / "v.mp4"
		assert run.detector.name is DetectorChoice.YOLOV8_VISDRONE
		assert run.detector.conf == 0.25
		assert run.runtime.device == "cpu"
		assert run.calibration.meters_per_pixel == 0.05
		assert run.tracker.det_thresh == 0.1
		assert run.orientation.smoothing_window == 5
		assert run.export.out == tmp_path / "out.trj"
		assert run.export.timestep_precision == 0.0
		assert run.window.start_seconds is None
		assert run.window.end_seconds is None
		assert run.options.force is False
		assert run.options.timing_csv is None  # "" disables

	def test_off_values_are_explicit_and_legal(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path), {})
		# Empty strings / 0 / false are valid "disabled" values, not missing keys.
		assert run.options.timing_csv is None
		assert run.window.start_seconds is None
		assert run.export.timestep_precision == 0.0


class TestPrecedence:
	def test_cli_override_beats_file(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(
			_complete(tmp_path),
			{"detector.conf": 0.4, "detector.name": "rt_detr"},
		)
		assert run.detector.conf == 0.4
		assert run.detector.name is DetectorChoice.RT_DETR

	def test_positional_video_and_out_override_file(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(
			_complete(tmp_path),
			{"input.video": Path("/other/clip.mp4"), "export.out": Path("/other/out.trj")},
		)
		assert run.input.video == Path("/other/clip.mp4")
		assert run.export.out == Path("/other/out.trj")

	def test_none_override_falls_through_to_file(self, tmp_path: Path) -> None:
		# A flag not passed arrives as None and must not shadow the file value.
		run = RunConfig.resolve(_complete(tmp_path), {"detector.conf": None})
		assert run.detector.conf == 0.25


class TestMissingKeys:
	def test_empty_config_lists_every_missing_key(self) -> None:
		with pytest.raises(ConfigError) as excinfo:
			RunConfig.resolve({}, {})
		problems = excinfo.value.problems
		assert "input.video is missing." in problems
		assert "detector.name is missing." in problems
		assert "runtime.device is missing." in problems
		assert any("calibration" in p for p in problems)
		assert len(problems) >= 8  # aggregated, not one-at-a-time

	def test_partial_config_reports_only_what_is_absent(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["runtime"]
		with pytest.raises(ConfigError) as excinfo:
			RunConfig.resolve(file_values, {})
		assert excinfo.value.problems == ["runtime.device is missing."]


class TestCalibration:
	def test_meters_per_pixel_method(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path), {})
		assert run.calibration.resolve_scale(_METADATA) == 0.05

	def test_drone_model_with_altitude_computes_gsd(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(
			_complete(tmp_path, calibration={"drone_model": "mavic_3", "altitude_m": 80.0}),
			{},
		)
		spec = lookup("mavic_3")
		expected = ground_sample_distance(
			sensor_width_mm=spec.sensor_width_mm,
			focal_length_mm=spec.focal_length_mm,
			altitude_m=80.0,
			image_width_pixels=_METADATA.width,
		)
		assert run.calibration.resolve_scale(_METADATA) == pytest.approx(expected)

	def test_both_methods_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="exactly one"):
			RunConfig.resolve(
				_complete(
					tmp_path, calibration={"meters_per_pixel": 0.05, "drone_model": "mavic_3"}
				),
				{},
			)

	def test_drone_model_without_altitude_source_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="altitude_m"):
			RunConfig.resolve(_complete(tmp_path, calibration={"drone_model": "mavic_3"}), {})

	def test_unknown_drone_model_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="unknown"):
			RunConfig.resolve(
				_complete(tmp_path, calibration={"drone_model": "not_a_drone", "altitude_m": 80.0}),
				{},
			)

	def test_non_positive_meters_per_pixel_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="positive"):
			RunConfig.resolve(_complete(tmp_path, calibration={"meters_per_pixel": 0.0}), {})

	def test_srt_path_as_altitude_source_resolves(self, tmp_path: Path) -> None:
		srt = tmp_path / "clip.SRT"
		srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n[rel_alt: 80.000 abs_alt: 100.0]\n\n")
		run = RunConfig.resolve(
			_complete(tmp_path, calibration={"drone_model": "mavic_3", "srt": str(srt)}),
			{},
		)
		assert run.calibration.resolve_scale(_METADATA) > 0.0


class TestWindow:
	def test_timecodes_parse_to_seconds(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path, window={"start": "0:10", "end": "1:30"}), {})
		assert run.window.start_seconds == 10.0
		assert run.window.end_seconds == 90.0

	def test_end_before_start_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="after"):
			RunConfig.resolve(_complete(tmp_path, window={"start": "1:00", "end": "0:30"}), {})

	def test_malformed_timecode_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="timecode"):
			RunConfig.resolve(_complete(tmp_path, window={"start": "abc", "end": ""}), {})


class TestValueValidation:
	def test_wrong_type_is_reported(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["detector"]["conf"] = "high"
		with pytest.raises(ConfigError, match="must be a number"):
			RunConfig.resolve(file_values, {})

	def test_smoothing_window_below_two_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match=">= 2"):
			RunConfig.resolve(_complete(tmp_path, orientation={"smoothing_window": 1}), {})

	def test_unknown_detector_name_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["detector"]["name"] = "yolov11"
		with pytest.raises(ConfigError, match="unknown"):
			RunConfig.resolve(file_values, {})

	def test_invalid_device_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError, match="device"):
			RunConfig.resolve(_complete(tmp_path, runtime={"device": "gpu"}), {})

	def test_negative_timestep_precision_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["export"]["timestep_precision"] = -0.5
		with pytest.raises(ConfigError, match="timestep_precision"):
			RunConfig.resolve(file_values, {})

	def test_timing_csv_path_resolves_to_path(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["run"]["timing_csv"] = str(tmp_path / "timing.csv")
		run = RunConfig.resolve(file_values, {})
		assert run.options.timing_csv == tmp_path / "timing.csv"


class TestLoadToml:
	def test_reads_a_nested_table(self, tmp_path: Path) -> None:
		path = tmp_path / "c.toml"
		path.write_text('[detector]\nname = "rt_detr"\nconf = 0.3\n')
		assert load_toml(path) == {"detector": {"name": "rt_detr", "conf": 0.3}}

	def test_malformed_toml_raises_value_error(self, tmp_path: Path) -> None:
		path = tmp_path / "c.toml"
		path.write_text("this is = = not toml")
		with pytest.raises(ValueError, match="not valid TOML"):
			load_toml(path)

	def test_missing_file_raises(self, tmp_path: Path) -> None:
		with pytest.raises(FileNotFoundError):
			load_toml(tmp_path / "nope.toml")
