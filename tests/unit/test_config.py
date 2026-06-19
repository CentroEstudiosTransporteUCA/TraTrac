"""Tests for run-config resolution and the TOML loader.

``RunConfig.resolve`` is pure and takes plain mappings, so most tests build the
config as a dict — no temp files needed. The TOML loader is tested separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

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
		"input": {"video": str(tmp_path / "v.mp4"), "process_fps": 0.0},
		"detector": {
			"name": "yolov8_visdrone",
			"checkpoint": "repo/model",
			"conf": 0.25,
			"filename": "model.pt",
		},
		"runtime": {"device": "cpu"},
		"calibration": {"meters_per_pixel": 0.05},
		"ego_motion": {"enabled": False},
		"tracker": {"det_thresh": 0.1},
		"orientation": {"smoothing_window": 5},
		"export": {
			"out": str(tmp_path / "out.trj"),
			"timestep_precision": 0.0,
			"video_out": "",
			"video_trail": 0,
			"transform_csv": "",
		},
		"window": {"start": "", "end": ""},
		"analysis": {"exclusion_zones": ""},
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
		assert run.export.video_out is None  # "" disables
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


class TestProcessFps:
	def test_zero_means_every_frame(self, tmp_path: Path) -> None:
		assert RunConfig.resolve(_complete(tmp_path), {}).input.process_fps == 0.0

	def test_positive_resolves(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["input"]["process_fps"] = 10.0
		assert RunConfig.resolve(file_values, {}).input.process_fps == 10.0

	def test_negative_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["input"]["process_fps"] = -1.0
		with pytest.raises(ConfigError, match="process_fps"):
			RunConfig.resolve(file_values, {})

	def test_missing_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["input"]["process_fps"]
		with pytest.raises(ConfigError, match=r"input\.process_fps is missing"):
			RunConfig.resolve(file_values, {})


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


class TestEgoMotion:
	_ENABLED: ClassVar[dict[str, Any]] = {
		"enabled": True,
		"n_features": 2000,
		"match_ratio": 0.75,
		"min_matches": 10,
		"ransac_threshold": 3.0,
		"min_anchor_overlap": 0.6,
		"transforms": "",
	}

	def test_disabled_needs_no_orb_params(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path, ego_motion={"enabled": False}), {})
		assert run.ego_motion.enabled is False

	def test_enabled_resolves_orb_params(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path, ego_motion=self._ENABLED), {})
		assert run.ego_motion.enabled is True
		assert run.ego_motion.n_features == 2000
		assert run.ego_motion.match_ratio == 0.75
		assert run.ego_motion.min_matches == 10
		assert run.ego_motion.ransac_threshold == 3.0
		assert run.ego_motion.min_anchor_overlap == 0.6

	def test_missing_enabled_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["ego_motion"]
		with pytest.raises(ConfigError, match=r"ego_motion\.enabled is missing"):
			RunConfig.resolve(file_values, {})

	def test_enabled_without_orb_params_is_an_error(self, tmp_path: Path) -> None:
		with pytest.raises(ConfigError) as excinfo:
			RunConfig.resolve(_complete(tmp_path, ego_motion={"enabled": True}), {})
		problems = excinfo.value.problems
		assert "ego_motion.n_features is missing." in problems
		assert "ego_motion.match_ratio is missing." in problems
		assert "ego_motion.min_matches is missing." in problems
		assert "ego_motion.ransac_threshold is missing." in problems
		assert "ego_motion.min_anchor_overlap is missing." in problems

	def test_replay_transforms_resolves_without_orb_params(self, tmp_path: Path) -> None:
		# A replay source makes the ORB params unnecessary (poses are read back).
		run = RunConfig.resolve(
			_complete(
				tmp_path,
				ego_motion={"enabled": True, "transforms": str(tmp_path / "transforms.csv")},
			),
			{},
		)
		assert run.ego_motion.enabled is True
		assert run.ego_motion.transforms == tmp_path / "transforms.csv"

	def test_transforms_requires_enabled(self, tmp_path: Path) -> None:
		# A replay source with stabilization off is a contradictory spec: transforms is
		# only read when enabled, so it is silently dropped — assert it is None.
		run = RunConfig.resolve(
			_complete(
				tmp_path,
				ego_motion={"enabled": False, "transforms": str(tmp_path / "transforms.csv")},
			),
			{},
		)
		assert run.ego_motion.enabled is False
		assert run.ego_motion.transforms is None

	def test_out_of_range_anchor_overlap_is_an_error(self, tmp_path: Path) -> None:
		bad = {**self._ENABLED, "min_anchor_overlap": 1.5}
		with pytest.raises(ConfigError, match="min_anchor_overlap"):
			RunConfig.resolve(_complete(tmp_path, ego_motion=bad), {})

	def test_out_of_range_orb_params_are_errors(self, tmp_path: Path) -> None:
		bad = {**self._ENABLED, "match_ratio": 1.5, "min_matches": 1}
		with pytest.raises(ConfigError) as excinfo:
			RunConfig.resolve(_complete(tmp_path, ego_motion=bad), {})
		problems = excinfo.value.problems
		assert any("match_ratio" in p for p in problems)
		assert any("min_matches" in p for p in problems)


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


class TestVideoExport:
	def test_video_out_path_enables_overlay_and_resolves_trail(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["export"]["video_out"] = str(tmp_path / "overlay.mp4")
		file_values["export"]["video_trail"] = 60
		run = RunConfig.resolve(file_values, {})
		assert run.export.video_out == tmp_path / "overlay.mp4"
		assert run.export.video_trail == 60

	def test_missing_video_out_key_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["export"]["video_out"]
		with pytest.raises(ConfigError, match="video_out is missing"):
			RunConfig.resolve(file_values, {})

	def test_video_trail_only_required_when_video_enabled(self, tmp_path: Path) -> None:
		# With the overlay off ("" ), a missing video_trail must NOT be an error.
		file_values = _complete(tmp_path)
		del file_values["export"]["video_trail"]
		run = RunConfig.resolve(file_values, {})
		assert run.export.video_out is None

	def test_video_trail_required_when_video_enabled(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["export"]["video_out"] = str(tmp_path / "overlay.mp4")
		del file_values["export"]["video_trail"]
		with pytest.raises(ConfigError, match="video_trail is missing"):
			RunConfig.resolve(file_values, {})


class TestTransformCsv:
	_ENABLED: ClassVar[dict[str, Any]] = {
		"enabled": True,
		"n_features": 2000,
		"match_ratio": 0.75,
		"min_matches": 10,
		"ransac_threshold": 3.0,
		"min_anchor_overlap": 0.6,
		"transforms": "",
	}

	def test_off_resolves_to_none(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path), {})
		assert run.export.transform_csv is None  # "" disables

	def test_missing_key_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["export"]["transform_csv"]
		with pytest.raises(ConfigError, match="transform_csv is missing"):
			RunConfig.resolve(file_values, {})

	def test_path_resolves_when_ego_motion_enabled(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path, ego_motion=self._ENABLED)
		file_values["export"]["transform_csv"] = str(tmp_path / "transforms.csv")
		run = RunConfig.resolve(file_values, {})
		assert run.export.transform_csv == tmp_path / "transforms.csv"

	def test_set_without_stabilization_is_an_error(self, tmp_path: Path) -> None:
		# ego_motion disabled in the default fixture: a transform CSV would only ever
		# hold identities, so requesting one is a contradictory run spec.
		file_values = _complete(tmp_path)
		file_values["export"]["transform_csv"] = str(tmp_path / "transforms.csv")
		with pytest.raises(ConfigError, match=r"transform_csv requires ego_motion\.enabled"):
			RunConfig.resolve(file_values, {})

	def test_negative_video_trail_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["export"]["video_out"] = str(tmp_path / "overlay.mp4")
		file_values["export"]["video_trail"] = -1
		with pytest.raises(ConfigError, match="video_trail must be"):
			RunConfig.resolve(file_values, {})

	def test_cli_video_out_override_enables_overlay(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(
			_complete(tmp_path),
			{"export.video_out": tmp_path / "overlay.mp4", "export.video_trail": 30},
		)
		assert run.export.video_out == tmp_path / "overlay.mp4"
		assert run.export.video_trail == 30


class TestAnalysis:
	def test_off_resolves_to_none(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(_complete(tmp_path), {})
		assert run.analysis.exclusion_zones is None  # "" disables

	def test_path_resolves(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		file_values["analysis"]["exclusion_zones"] = str(tmp_path / "zones.json")
		run = RunConfig.resolve(file_values, {})
		assert run.analysis.exclusion_zones == tmp_path / "zones.json"

	def test_missing_key_is_an_error(self, tmp_path: Path) -> None:
		file_values = _complete(tmp_path)
		del file_values["analysis"]["exclusion_zones"]
		with pytest.raises(ConfigError, match="exclusion_zones is missing"):
			RunConfig.resolve(file_values, {})

	def test_cli_override_enables(self, tmp_path: Path) -> None:
		run = RunConfig.resolve(
			_complete(tmp_path), {"analysis.exclusion_zones": tmp_path / "zones.json"}
		)
		assert run.analysis.exclusion_zones == tmp_path / "zones.json"


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
