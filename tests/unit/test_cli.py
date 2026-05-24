"""Tests for CLI output-path preparation."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from tratrac import cli
from tratrac.cli import (
	_open_video,
	_parse_timecode,
	_prepare_output_path,
	_resolve_scale,
	_validate_device,
	_validate_drone_model,
	_validate_timestep_precision,
	app,
)
from tratrac.domain.frame import VideoMetadata


class TestPrepareOutputPath:
	def test_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
		target = tmp_path / "nested" / "deep" / "out.trj"
		_prepare_output_path(target)
		assert target.parent.is_dir()

	def test_does_not_prompt_when_file_is_absent(
		self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		def _boom(*args: object, **kwargs: object) -> bool:
			raise AssertionError("must not prompt when the file does not exist")

		monkeypatch.setattr(typer, "confirm", _boom)
		_prepare_output_path(tmp_path / "fresh.trj")  # must not raise

	def test_proceeds_when_user_confirms_overwrite(
		self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		target = tmp_path / "exists.trj"
		target.write_text("old")
		monkeypatch.setattr(cli, "_is_interactive", lambda: True)
		monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
		_prepare_output_path(target)  # confirmed -> no abort

	def test_aborts_when_user_declines_overwrite(
		self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		target = tmp_path / "exists.trj"
		target.write_text("old")

		def _decline(*args: object, **kwargs: object) -> bool:
			raise typer.Abort

		monkeypatch.setattr(cli, "_is_interactive", lambda: True)
		monkeypatch.setattr(typer, "confirm", _decline)
		with pytest.raises(typer.Abort):
			_prepare_output_path(target)

	def test_force_overwrites_without_prompting(
		self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		target = tmp_path / "exists.trj"
		target.write_text("old")

		def _boom(*args: object, **kwargs: object) -> bool:
			raise AssertionError("--force must not prompt")

		# Even with a TTY, --force skips the prompt entirely.
		monkeypatch.setattr(cli, "_is_interactive", lambda: True)
		monkeypatch.setattr(typer, "confirm", _boom)
		_prepare_output_path(target, force=True)  # must not raise

	def test_errors_non_interactively_when_file_exists(
		self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		target = tmp_path / "exists.trj"
		target.write_text("old")

		def _boom(*args: object, **kwargs: object) -> bool:
			raise AssertionError("must not prompt when stdin is not a TTY")

		monkeypatch.setattr(cli, "_is_interactive", lambda: False)
		monkeypatch.setattr(typer, "confirm", _boom)
		with pytest.raises(typer.BadParameter, match="--force"):
			_prepare_output_path(target)


_METADATA = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=100)


class TestResolveScale:
	def test_returns_explicit_meters_per_pixel(self, tmp_path: Path) -> None:
		scale = _resolve_scale(
			video_path=tmp_path / "v.mp4",
			metadata=_METADATA,
			meters_per_pixel=0.05,
			drone_model="",
			altitude_m=0.0,
			srt_path=None,
		)
		assert scale == 0.05

	def test_errors_when_no_calibration_supplied(self, tmp_path: Path) -> None:
		with pytest.raises(typer.Exit) as excinfo:
			_resolve_scale(
				video_path=tmp_path / "v.mp4",
				metadata=_METADATA,
				meters_per_pixel=0.0,
				drone_model="",
				altitude_m=0.0,
				srt_path=None,
			)
		assert excinfo.value.exit_code == 2


class TestProcessCalibrationGuard:
	def test_aborts_before_touching_outputs_when_uncalibrated(self, tmp_path: Path) -> None:
		# A pre-existing output would make the old flow prompt to overwrite first;
		# the calibration guard must fire before that and leave the file untouched.
		video = tmp_path / "v.mp4"
		video.write_bytes(b"\x00")  # exists, to clear typer's Argument(exists=True)
		out = tmp_path / "out.trj"
		out.write_text("old")
		result = CliRunner().invoke(app, ["process", str(video), "--out", str(out)], input="")
		assert result.exit_code == 2
		assert out.read_text() == "old"


class TestParseTimecode:
	@pytest.mark.parametrize(
		("value", "expected"),
		[
			("12.5", 12.5),
			("0", 0.0),
			("1:30", 90.0),
			("0:01:05.25", 65.25),
			("1:00:00", 3600.0),
		],
	)
	def test_parses_supported_formats(self, value: str, expected: float) -> None:
		assert _parse_timecode(value) == pytest.approx(expected)

	@pytest.mark.parametrize("value", ["1:2:3:4", "abc", "1:xx", "-5", "1:-3"])
	def test_rejects_malformed_input(self, value: str) -> None:
		with pytest.raises(typer.BadParameter):
			_parse_timecode(value)


class TestValidateDevice:
	@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:1"])
	def test_accepts_supported_devices(self, device: str) -> None:
		_validate_device(device)  # must not raise

	@pytest.mark.parametrize("device", ["gpu", "cuda:", "CPU", "", "cuda:x"])
	def test_rejects_unsupported_devices(self, device: str) -> None:
		with pytest.raises(typer.BadParameter):
			_validate_device(device)


class TestValidateDroneModel:
	def test_empty_is_a_noop(self) -> None:
		_validate_drone_model("")  # must not raise

	@pytest.mark.parametrize("model", ["mavic_3", "MAVIC_3"])
	def test_accepts_known_model_case_insensitively(self, model: str) -> None:
		_validate_drone_model(model)  # must not raise

	def test_rejects_unknown_model(self) -> None:
		with pytest.raises(typer.BadParameter, match="Unknown drone model"):
			_validate_drone_model("definitely_not_a_drone")


class TestOpenVideo:
	def test_translates_open_value_error_to_bad_parameter(
		self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
	) -> None:
		class _Boom:
			def __init__(self, *args: object, **kwargs: object) -> None:
				pass

			def __enter__(self) -> _Boom:
				raise ValueError("start is at or beyond the video duration (~1.000s).")

			def __exit__(self, *args: object) -> None:
				pass

		monkeypatch.setattr(cli, "OpenCvVideoSource", _Boom)
		with (
			pytest.raises(typer.BadParameter, match="video duration"),
			_open_video(tmp_path / "v.mp4", start_seconds=99.0, end_seconds=None),
		):
			pass

	def test_yields_source_and_closes_on_exit(
		self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
	) -> None:
		events: list[str] = []

		class _Fake:
			def __init__(self, *args: object, **kwargs: object) -> None:
				pass

			def __enter__(self) -> _Fake:
				events.append("enter")
				return self

			def __exit__(self, *args: object) -> None:
				events.append("exit")

		monkeypatch.setattr(cli, "OpenCvVideoSource", _Fake)
		with _open_video(tmp_path / "v.mp4", start_seconds=None, end_seconds=None) as source:
			events.append(type(source).__name__)
		assert events == ["enter", "_Fake", "exit"]


class TestProcessOutputPathSanitization:
	def _video(self, tmp_path: Path) -> Path:
		video = tmp_path / "v.mp4"
		video.write_bytes(b"\x00")
		return video

	def test_rejects_identical_out_and_timing_csv(self, tmp_path: Path) -> None:
		video = self._video(tmp_path)
		shared = tmp_path / "same.trj"
		result = CliRunner().invoke(
			app,
			[
				"process",
				str(video),
				"--out",
				str(shared),
				"--timing-csv",
				str(shared),
				"--meters-per-pixel",
				"0.1",
			],
		)
		assert result.exit_code == 2

	def test_rejects_directory_as_output(self, tmp_path: Path) -> None:
		video = self._video(tmp_path)
		a_dir = tmp_path / "outdir"
		a_dir.mkdir()
		result = CliRunner().invoke(
			app,
			["process", str(video), "--out", str(a_dir), "--meters-per-pixel", "0.1"],
		)
		assert result.exit_code == 2


class TestValidateTimestepPrecision:
	def test_none_is_accepted(self) -> None:
		_validate_timestep_precision(None)  # no decimation requested; must not raise

	def test_positive_sub_second_is_accepted_silently(
		self, capsys: pytest.CaptureFixture[str]
	) -> None:
		_validate_timestep_precision(0.1)
		assert capsys.readouterr().err == ""

	def test_zero_is_rejected(self) -> None:
		with pytest.raises(typer.BadParameter):
			_validate_timestep_precision(0.0)

	def test_negative_is_rejected(self) -> None:
		with pytest.raises(typer.BadParameter):
			_validate_timestep_precision(-0.5)

	def test_coarse_value_warns_but_is_accepted(self, capsys: pytest.CaptureFixture[str]) -> None:
		_validate_timestep_precision(1.0)  # valid, just too coarse for SSAM
		assert "coarse" in capsys.readouterr().err
