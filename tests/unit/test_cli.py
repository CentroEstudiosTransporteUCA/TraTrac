"""Tests for CLI wiring: output-path preparation, video opening, and the
config-resolution guards. The resolution logic itself is covered by
``test_config.py``; here we test only the CLI's use of it."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from tratrac import cli
from tratrac.cli import _open_video, _prepare_output_path, app


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


def _video(tmp_path: Path) -> Path:
	video = tmp_path / "v.mp4"
	video.write_bytes(b"\x00")
	return video


def _full_config(tmp_path: Path, *, video: Path) -> Path:
	"""Write a complete, valid persisted run config to ``tmp_path``."""
	config = tmp_path / "run.toml"
	config.write_text(
		"[input]\n"
		f'video = "{video}"\n'
		"[detector]\n"
		'name = "yolov8_visdrone"\n'
		'checkpoint = "repo/model"\n'
		"conf = 0.25\n"
		'filename = "model.pt"\n'
		"[runtime]\n"
		'device = "cpu"\n'
		"[calibration]\n"
		"meters_per_pixel = 0.1\n"
		"[tracker]\n"
		"det_thresh = 0.1\n"
		"[orientation]\n"
		"smoothing_window = 5\n"
		"[export]\n"
		f'out = "{tmp_path / "out.trj"}"\n'
		"timestep_precision = 0.0\n"
		"[window]\n"
		'start = ""\n'
		'end = ""\n'
		"[run]\n"
		"force = false\n"
		'timing_csv = ""\n'
	)
	return config


class TestProcessConfigGuard:
	def test_aborts_before_touching_outputs_when_underspecified(self, tmp_path: Path) -> None:
		# Resolution fails (no config, no calibration/detector flags) before the
		# overwrite step, so a pre-existing output is left untouched.
		video = _video(tmp_path)
		out = tmp_path / "out.trj"
		out.write_text("old")
		result = CliRunner().invoke(app, [str(video), "--out", str(out)], input="")
		assert result.exit_code == 2
		assert out.read_text() == "old"

	def test_reports_every_missing_key_at_once(self) -> None:
		result = CliRunner().invoke(app, [], input="")
		assert result.exit_code == 2
		assert "input.video is missing" in result.output
		assert "runtime.device is missing" in result.output


class TestProcessOutputPathSanitization:
	def test_rejects_identical_out_and_timing_csv(self, tmp_path: Path) -> None:
		video = _video(tmp_path)
		config = _full_config(tmp_path, video=video)
		shared = tmp_path / "same.trj"
		result = CliRunner().invoke(
			app,
			["--config", str(config), "--out", str(shared), "--timing-csv", str(shared)],
		)
		assert result.exit_code == 2

	def test_rejects_directory_as_output(self, tmp_path: Path) -> None:
		# Typer's dir_okay=False on --out rejects a directory at parse time.
		video = _video(tmp_path)
		a_dir = tmp_path / "outdir"
		a_dir.mkdir()
		result = CliRunner().invoke(app, [str(video), "--out", str(a_dir)])
		assert result.exit_code == 2


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
