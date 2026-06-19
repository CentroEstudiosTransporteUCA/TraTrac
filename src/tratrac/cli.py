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
from collections.abc import Callable, Iterator
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
from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import (
	DetectionObserver,
	Detector,
	EgoMotionEstimator,
	OrientationEstimator,
	TimingSink,
	Tracker,
	TrajectoryExporter,
	TransformSink,
)
from tratrac.infrastructure.config.toml import load_toml
from tratrac.infrastructure.detection.masking import MaskingDetector
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector
from tratrac.infrastructure.exclusion.json import load_exclusion_zones
from tratrac.infrastructure.export.composite import CompositeTrajectoryExporter
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter
from tratrac.infrastructure.export.overlay_video import OverlayVideoExporter
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
from tratrac.infrastructure.transform.csv import CsvTransformSink
from tratrac.infrastructure.transform.recording import RecordingEgoMotionEstimator
from tratrac.infrastructure.video.ego_motion_orb import OrbEgoMotionEstimator
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
	stabilize: Annotated[
		bool | None,
		typer.Option(
			"--stabilize/--no-stabilize",
			help="ORB ego-motion: stabilize detection coordinates (ego_motion.enabled).",
		),
	] = None,
	orb_features: Annotated[
		int | None,
		typer.Option("--orb-features", help="ORB keypoints per frame (ego_motion.n_features)."),
	] = None,
	orb_match_ratio: Annotated[
		float | None,
		typer.Option("--orb-match-ratio", help="Lowe ratio test, (0, 1) (ego_motion.match_ratio)."),
	] = None,
	orb_min_matches: Annotated[
		int | None,
		typer.Option(
			"--orb-min-matches",
			help="Min good matches to fit a transform (ego_motion.min_matches).",
		),
	] = None,
	orb_ransac_threshold: Annotated[
		float | None,
		typer.Option(
			"--orb-ransac-threshold",
			help="RANSAC reprojection threshold in px (ego_motion.ransac_threshold).",
		),
	] = None,
	min_anchor_overlap: Annotated[
		float | None,
		typer.Option(
			"--min-anchor-overlap",
			help="Re-anchor when the keyframe's shared area drops below this, (0, 1) "
			"(ego_motion.min_anchor_overlap).",
		),
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
	video_out: Annotated[
		Path | None,
		typer.Option(
			"--video-out",
			dir_okay=False,
			help="Overlay video output (raw frame + trajectories); omit to skip (export.video_out).",
		),
	] = None,
	video_trail: Annotated[
		int | None,
		typer.Option(
			"--video-trail",
			help="Trail length in frames for the overlay video; 0 = whole path (export.video_trail).",
		),
	] = None,
	transform_csv: Annotated[
		Path | None,
		typer.Option(
			"--transform-csv",
			dir_okay=False,
			help='Per-frame ego-motion transform CSV; "" disables. Requires --stabilize '
			"(export.transform_csv).",
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
	exclusion_zones: Annotated[
		Path | None,
		typer.Option(
			"--exclusion-zones",
			dir_okay=False,
			help='Sidecar JSON of image-space polygons to exclude from analysis; "" disables '
			"(analysis.exclusion_zones).",
		),
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
		"ego_motion.enabled": stabilize,
		"ego_motion.n_features": orb_features,
		"ego_motion.match_ratio": orb_match_ratio,
		"ego_motion.min_matches": orb_min_matches,
		"ego_motion.ransac_threshold": orb_ransac_threshold,
		"ego_motion.min_anchor_overlap": min_anchor_overlap,
		"tracker.det_thresh": det_thresh,
		"orientation.smoothing_window": smoothing_window,
		"export.timestep_precision": timestep_precision,
		"export.video_out": video_out,
		"export.video_trail": video_trail,
		"export.transform_csv": transform_csv,
		"window.start": start,
		"window.end": end,
		"analysis.exclusion_zones": exclusion_zones,
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
	# Load the exclusion-zone polygons up front (pure, cheap file read) so a bad path
	# or malformed JSON fails before the costly video open. None when the feature is off.
	zones: ExclusionZones | None = None
	if run.analysis.exclusion_zones is not None:
		if not run.analysis.exclusion_zones.is_file():
			raise typer.BadParameter(
				f"analysis.exclusion_zones {run.analysis.exclusion_zones} does not exist "
				"or is not a file."
			)
		try:
			zones = load_exclusion_zones(run.analysis.exclusion_zones)
		except (ValueError, OSError) as exc:
			raise typer.BadParameter(str(exc)) from exc
	if (
		run.options.timing_csv is not None
		and run.export.out.resolve() == run.options.timing_csv.resolve()
	):
		raise typer.BadParameter("run.timing_csv must differ from export.out.")
	if run.export.video_out is not None and run.export.video_out.resolve() in {
		run.export.out.resolve(),
		run.options.timing_csv.resolve() if run.options.timing_csv is not None else None,
	}:
		raise typer.BadParameter("export.video_out must differ from export.out and run.timing_csv.")
	if run.export.transform_csv is not None and run.export.transform_csv.resolve() in {
		run.export.out.resolve(),
		run.options.timing_csv.resolve() if run.options.timing_csv is not None else None,
		run.export.video_out.resolve() if run.export.video_out is not None else None,
	}:
		raise typer.BadParameter(
			"export.transform_csv must differ from export.out, run.timing_csv, and export.video_out."
		)
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
	if run.export.video_out is not None:
		_prepare_output_path(run.export.video_out, force=run.options.force)
	if run.export.transform_csv is not None:
		_prepare_output_path(run.export.transform_csv, force=run.options.force)

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
		# Coordinate stabilization (MVP1.9, vault/05_75_mvp1_9.md): the detector and
		# tracker run on the raw frame; the ego-motion transform is applied to the
		# detections (not the pixels) inside the pipeline. None when stabilization is
		# off. The estimator is also the DetectionObserver — it masks the raw-frame
		# detections out of its own ORB feature extraction.
		ego_motion: OrbEgoMotionEstimator | None = None
		detection_observer: DetectionObserver | None = None
		if run.ego_motion.enabled:
			ego_motion = OrbEgoMotionEstimator(
				n_features=run.ego_motion.n_features,
				match_ratio=run.ego_motion.match_ratio,
				min_matches=run.ego_motion.min_matches,
				ransac_threshold=run.ego_motion.ransac_threshold,
				min_anchor_overlap=run.ego_motion.min_anchor_overlap,
				# Also mask the static exclusion zones out of ORB feature extraction.
				exclusion_zones=zones,
			)
			detection_observer = ego_motion
		det: Detector = _build_detector(run.detector, device=run.runtime.device)
		if zones is not None:
			# Drop detections inside the exclusion zones at the detector seam — the one
			# point where detections are still in raw pixel space (vault/21). Wrapped
			# below TimedDetector so the EXPORT/timing step counts detect + filter.
			det = MaskingDetector(det, zones)
		# When we stabilize coordinates ourselves, disable BoT-SORT's own camera-motion
		# compensation so it does not double-correct the already-stabilized boxes.
		tracker: Tracker = BoxmotBotSortTracker(
			source.metadata,
			det_thresh=run.tracker.det_thresh,
			compensate_camera_motion=not run.ego_motion.enabled,
		)
		exporter: TrajectoryExporter = SsamTrjExporter(run.export.out, source.metadata, scale=scale)
		if run.export.timestep_precision > 0.0:
			# Decimate the TIMESTEP stream. Sits inside TimedExporter (below) so the
			# EXPORT step still records once per processed frame (see vault/15).
			exporter = DecimatingTrajectoryExporter(
				exporter,
				min_interval_seconds=run.export.timestep_precision,
				fps=source.metadata.fps,
			)
		if run.export.video_out is not None:
			# Fan out to the .trj (decimated above if requested) AND a full-framerate
			# overlay video. Only the .trj leg is decimated; the video keeps every
			# processed frame. See vault/20_video_export.md.
			exporter = CompositeTrajectoryExporter(
				[
					exporter,
					OverlayVideoExporter(
						run.export.video_out,
						source.metadata,
						scale=scale,
						trail_length=run.export.video_trail,
						# Map stabilized coordinates back onto the raw frame; identity
						# (default) when stabilization is off.
						transform_source=_overlay_transform_source(ego_motion),
					),
				]
			)
		orientation: OrientationEstimator = EmaOrientationEstimator(
			smoothing_window=run.orientation.smoothing_window,
			meters_per_pixel=scale,
		)
		with (
			_timing_sink(run.options.timing_csv) as sink,
			_transform_sink(run.export.transform_csv) as transform_sink,
		):
			if sink is not None:
				det = TimedDetector(det, sink)
				tracker = TimedTracker(tracker, sink)
				orientation = TimedOrientation(orientation, sink)
				exporter = TimedExporter(exporter, sink)
			# Persist the per-frame transform by decorating the estimator (the
			# pipeline stays untouched). The concrete estimator above keeps serving the
			# overlay's transform_source and the DetectionObserver; the pipeline drives
			# the recorder. Guaranteed present when transform_csv is set (config guard).
			pipeline_ego_motion: EgoMotionEstimator | None = ego_motion
			if transform_sink is not None and ego_motion is not None:
				pipeline_ego_motion = RecordingEgoMotionEstimator(ego_motion, transform_sink)
			pipeline = TrajectoryPipeline(
				video=source,
				detector=det,
				tracker=tracker,
				exporter=exporter,
				orientation=orientation,
				reporter=ConsoleProgressReporter(),
				detection_observer=detection_observer,
				ego_motion=pipeline_ego_motion,
			)
			n_frames = pipeline.run()

	typer.echo(f"Processed {n_frames} frames -> {run.export.out} (scale={scale} m/px)")


# Above this, exported timesteps get too sparse for SSAM conflict analysis
# (vault/04: sub-second, ~0.1s, is the practical minimum). Still valid, so warn.
_COARSE_TIMESTEP_WARNING_SECONDS = 0.5


def _overlay_transform_source(
	ego_motion: OrbEgoMotionEstimator | None,
) -> Callable[[], Transform2D] | None:
	"""A callable yielding the current stabilization transform for the overlay video,
	or ``None`` when stabilization is off (the overlay then uses identity)."""
	if ego_motion is None:
		return None
	return lambda: ego_motion.current_transform


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


@contextmanager
def _transform_sink(path: Path | None) -> Iterator[TransformSink | None]:
	"""Yield a CSV transform sink when a path is given, else ``None`` (off)."""
	if path is None:
		yield None
		return
	with CsvTransformSink(path) as sink:
		yield sink


if __name__ == "__main__":
	app()
