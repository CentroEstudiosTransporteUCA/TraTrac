"""Typer CLI entry point for TraTrac.

A run is fully described by a persisted ``RunConfig`` (see
``tratrac.application.config`` and ``vault/19_config_file.md``). There are no
built-in defaults: every value comes from the ``--config`` TOML or a flag, and a
missing value fails the run listing exactly what is absent. Flags override the
config file key-for-key; the positional ``video`` overrides ``input.video`` and
``--out`` overrides ``export.out``. A complete config replays with no arguments.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer

from tratrac.application.config import (
	ConfigError,
	DetectorChoice,
	DetectorConfig,
	RunConfig,
)
from tratrac.application.orientation import EmaOrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.domain.ports import (
	Detector,
	OrientationEstimator,
	TimingSink,
	Tracker,
	TrajectoryExporter,
)
from tratrac.infrastructure.config.toml import load_toml
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.progress.console import ConsoleProgressReporter
from tratrac.infrastructure.timing.csv import CsvTimingSink
from tratrac.infrastructure.timing.decorators import (
	TimedDetector,
	TimedExporter,
	TimedOrientation,
	TimedTracker,
)
from tratrac.infrastructure.tracking.boxmot_bot_sort import BoxmotBotSortTracker
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac",
	help="Vehicle tracking and trajectory export for aerial video.",
	no_args_is_help=True,
)


@app.command()
def process(
	video: Annotated[
		Path | None,
		typer.Argument(help="Input video. Overrides input.video from the config."),
	] = None,
	config: Annotated[
		Path | None,
		typer.Option(
			"--config",
			exists=True,
			dir_okay=False,
			readable=True,
			help="Persisted run config (TOML). Supplies every value not given as a flag.",
		),
	] = None,
	out: Annotated[
		Path | None,
		typer.Option("--out", "-o", dir_okay=False, help="Output .trj path (export.out)."),
	] = None,
	detector: Annotated[
		DetectorChoice | None,
		typer.Option("--detector", help="Detector adapter (detector.name)."),
	] = None,
	conf: Annotated[
		float | None, typer.Option("--conf", help="Detection confidence threshold (detector.conf).")
	] = None,
	checkpoint: Annotated[
		str | None,
		typer.Option("--checkpoint", help="Checkpoint / HF repo id (detector.checkpoint)."),
	] = None,
	checkpoint_file: Annotated[
		str | None,
		typer.Option(
			"--checkpoint-file",
			help="Weights filename within the repo (detector.filename; yolov8_visdrone only).",
		),
	] = None,
	device: Annotated[
		str | None,
		typer.Option("--device", help="torch device (runtime.device): cpu, mps, cuda[:N]."),
	] = None,
	meters_per_pixel: Annotated[
		float | None,
		typer.Option("--meters-per-pixel", help="Direct GSD (calibration.meters_per_pixel)."),
	] = None,
	drone_model: Annotated[
		str | None,
		typer.Option(
			"--drone-model", help="Drone model key for GSD lookup (calibration.drone_model)."
		),
	] = None,
	altitude_m: Annotated[
		float | None,
		typer.Option("--altitude", help="Flight altitude in metres AGL (calibration.altitude_m)."),
	] = None,
	srt: Annotated[
		Path | None,
		typer.Option("--srt", help="DJI .SRT sidecar with per-frame altitude (calibration.srt)."),
	] = None,
	det_thresh: Annotated[
		float | None,
		typer.Option("--det-thresh", help="Tracker detection threshold (tracker.det_thresh)."),
	] = None,
	smoothing_window: Annotated[
		int | None,
		typer.Option(
			"--smoothing-window",
			help="Orientation EMA window, >= 2 (orientation.smoothing_window).",
		),
	] = None,
	timestep_precision: Annotated[
		float | None,
		typer.Option(
			"--timestep-precision",
			help="Min seconds between exported TIMESTEPs; 0 = every frame (export.timestep_precision).",
		),
	] = None,
	start: Annotated[
		str | None,
		typer.Option(
			"--start", help='Analysis-window start timecode; "" = clip start (window.start).'
		),
	] = None,
	end: Annotated[
		str | None,
		typer.Option("--end", help='Analysis-window end timecode; "" = clip end (window.end).'),
	] = None,
	force: Annotated[
		bool | None,
		typer.Option(
			"--force/--no-force", help="Overwrite existing outputs without prompting (run.force)."
		),
	] = None,
	timing_csv: Annotated[
		Path | None,
		typer.Option(
			"--timing-csv",
			dir_okay=False,
			help='Per-frame step-timing CSV; "" disables (run.timing_csv).',
		),
	] = None,
) -> None:
	"""Process a video into an SSAM .trj trajectory file."""
	overrides: dict[str, Any] = {
		"input.video": video,
		"export.out": out,
		"detector.name": detector.value if detector is not None else None,
		"detector.conf": conf,
		"detector.checkpoint": checkpoint,
		"detector.filename": checkpoint_file,
		"runtime.device": device,
		"calibration.meters_per_pixel": meters_per_pixel,
		"calibration.drone_model": drone_model,
		"calibration.altitude_m": altitude_m,
		"calibration.srt": srt,
		"tracker.det_thresh": det_thresh,
		"orientation.smoothing_window": smoothing_window,
		"export.timestep_precision": timestep_precision,
		"window.start": start,
		"window.end": end,
		"run.force": force,
		"run.timing_csv": timing_csv,
	}

	file_values: dict[str, Any] = {}
	if config is not None:
		try:
			file_values = load_toml(config)
		except ValueError as exc:
			raise typer.BadParameter(str(exc)) from exc

	try:
		run = RunConfig.resolve(file_values, overrides)
	except ConfigError as exc:
		typer.echo(f"ERROR: {exc}", err=True)
		raise typer.Exit(code=2) from exc

	# --- Fail-fast checks that need the filesystem but not the (costly) video open. ---
	if not run.input.video.is_file():
		raise typer.BadParameter(f"input video {run.input.video} does not exist or is not a file.")
	if (
		run.options.timing_csv is not None
		and run.export.out.resolve() == run.options.timing_csv.resolve()
	):
		raise typer.BadParameter("run.timing_csv must differ from export.out.")
	if run.export.timestep_precision > _COARSE_TIMESTEP_WARNING_SECONDS:
		typer.echo(
			f"WARNING: timestep_precision {run.export.timestep_precision}s is coarse; SSAM "
			"conflict analysis wants sub-second timesteps (~0.1s). The .trj stays valid but may "
			"be too sparse for surrogate-safety metrics.",
			err=True,
		)
	_prepare_output_path(run.export.out, force=run.options.force)
	if run.options.timing_csv is not None:
		_prepare_output_path(run.options.timing_csv, force=run.options.force)

	with _open_video(
		run.input.video,
		start_seconds=run.window.start_seconds,
		end_seconds=run.window.end_seconds,
	) as source:
		try:
			scale = run.calibration.resolve_scale(source.metadata)
		except (ValueError, ConfigError) as exc:
			# A non-positive altitude (e.g. an SRT with no usable values) surfaces from
			# the calibration chain; report it cleanly rather than as a traceback.
			raise typer.BadParameter(str(exc)) from exc
		det: Detector = _build_detector(run.detector, device=run.runtime.device)
		tracker: Tracker = BoxmotBotSortTracker(source.metadata, det_thresh=run.tracker.det_thresh)
		exporter: TrajectoryExporter = SsamTrjExporter(run.export.out, source.metadata, scale=scale)
		if run.export.timestep_precision > 0.0:
			# Decimate the TIMESTEP stream. Sits inside TimedExporter (below) so the
			# EXPORT step still records once per processed frame (see vault/15).
			exporter = DecimatingTrajectoryExporter(
				exporter,
				min_interval_seconds=run.export.timestep_precision,
				fps=source.metadata.fps,
			)
		orientation: OrientationEstimator = EmaOrientationEstimator(
			smoothing_window=run.orientation.smoothing_window,
			meters_per_pixel=scale,
		)
		with _timing_sink(run.options.timing_csv) as sink:
			if sink is not None:
				det = TimedDetector(det, sink)
				tracker = TimedTracker(tracker, sink)
				orientation = TimedOrientation(orientation, sink)
				exporter = TimedExporter(exporter, sink)
			pipeline = TrajectoryPipeline(
				video=source,
				detector=det,
				tracker=tracker,
				exporter=exporter,
				orientation=orientation,
				reporter=ConsoleProgressReporter(),
			)
			n_frames = pipeline.run()

	typer.echo(f"Processed {n_frames} frames -> {run.export.out} (scale={scale} m/px)")


# Above this, exported timesteps get too sparse for SSAM conflict analysis
# (vault/04: sub-second, ~0.1s, is the practical minimum). Still valid, so warn.
_COARSE_TIMESTEP_WARNING_SECONDS = 0.5


@contextmanager
def _open_video(
	video: Path, *, start_seconds: float | None, end_seconds: float | None
) -> Iterator[OpenCvVideoSource]:
	"""Open the (optionally windowed) video source.

	Translates range ``ValueError``s raised while opening — e.g. a start past the
	video's end — into a clean ``typer.BadParameter``. Exceptions from the
	processing body pass through untouched.
	"""
	with ExitStack() as stack:
		try:
			source = OpenCvVideoSource(video, start_seconds=start_seconds, end_seconds=end_seconds)
			stack.enter_context(source)
		except ValueError as exc:
			raise typer.BadParameter(str(exc)) from exc
		yield source


def _build_detector(detector: DetectorConfig, *, device: str) -> Detector:
	if detector.name is DetectorChoice.RT_DETR:
		return RtDetrDetector(
			checkpoint=detector.checkpoint,
			device=device,
			score_threshold=detector.conf,
		)
	if detector.name is DetectorChoice.YOLOV8_VISDRONE:
		return YoloV8VisDroneDetector(
			repo_id=detector.checkpoint,
			filename=detector.filename,
			device=device,
			score_threshold=detector.conf,
		)
	raise ValueError(f"Unknown detector choice: {detector.name}")


def _is_interactive() -> bool:
	"""Whether stdin can answer a prompt. False under pipes/redirects/CI.

	Wrapped (not inlined) so it is a single monkeypatch point in tests and the one
	place the CLI's interactivity assumption is named.
	"""
	return sys.stdin.isatty()


def _prepare_output_path(path: Path, *, force: bool = False) -> None:
	"""Make ``path`` writable: confirm overwrite if it exists, then create parents.

	The writers open with ``"w"``/``"wb"`` and would raise if the parent directory
	is missing. Overwrite confirmation is an interactive (CLI) concern, so it lives
	here rather than in the writers.

	``force`` skips the prompt outright. Otherwise, when the file exists and stdin
	is not a TTY (a non-interactive run), there is no way to answer the prompt, so we
	fail with an actionable error instead of letting ``click`` abort on EOF.
	"""
	if path.exists() and not force:
		if not _is_interactive():
			raise typer.BadParameter(
				f"{path} already exists and stdin is not a TTY to confirm overwrite. "
				"Re-run with --force to overwrite."
			)
		typer.confirm(f"{path} already exists. Overwrite?", abort=True)
	path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _timing_sink(path: Path | None) -> Iterator[TimingSink | None]:
	"""Yield a CSV timing sink when a path is given, else ``None`` (timing off)."""
	if path is None:
		yield None
		return
	with CsvTimingSink(path) as sink:
		yield sink


if __name__ == "__main__":
	app()
