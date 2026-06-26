"""Typer CLI entry point for TraTrac.

A run is fully described by a persisted ``RunConfig`` (see
``tratrac.application.config`` and ``vault/19_config_file.md``). There are no
built-in defaults and **no per-key override flags**: every value comes from the
``--config`` TOML, and a missing value fails the run listing exactly what is
absent. The sole flag is ``--force`` (overwrite existing outputs) — overwrite
policy is *not* a config key, since it never affects the trajectories. A complete
config replays with just ``--config``.

The run is **perception only**: it writes the track record (the raw tracked
measurements, the run's canonical output). It does not produce an SSAM ``.trj`` —
run ``tratrac-postprocess`` on the record to filter/smooth it into a ``.trj`` (vault/22).
With ``--anchors-dir`` it also exports the ORB keyframe anchors (PNGs + manifest) an
operator draws exclusion zones on (vault/21).
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
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.application.stabilization import EgoMotionStabilizer
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import (
	AnchorSink,
	DetectionObserver,
	DetectionStabilizer,
	Detector,
	EgoMotionEstimator,
	TimingSink,
	Tracker,
	TrackSink,
	TransformSink,
)
from tratrac.infrastructure.anchors.recording import AnchorRecordingEgoMotionEstimator
from tratrac.infrastructure.anchors.sink import AnchorManifestSink
from tratrac.infrastructure.config.toml import load_toml
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector
from tratrac.infrastructure.progress.console import ConsoleProgressReporter
from tratrac.infrastructure.timing.csv import CsvTimingSink
from tratrac.infrastructure.timing.decorators import (
	TimedDetectionObserver,
	TimedDetector,
	TimedEgoMotion,
	TimedStabilizer,
	TimedTracker,
	TimedTrackSink,
)
from tratrac.infrastructure.tracking.boxmot_bot_sort import BoxmotBotSortTracker
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink
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
	config: Annotated[
		Path | None,
		typer.Option(
			"--config",
			exists=True,
			dir_okay=False,
			readable=True,
			help="Persisted run config (TOML). Supplies every value for the run.",
		),
	] = None,
	force: Annotated[
		bool,
		typer.Option("--force/--no-force", help="Overwrite existing outputs without prompting."),
	] = False,
) -> None:
	"""Track a video into a record file (run tratrac-postprocess on it to get a .trj).

	The run is driven entirely by ``--config``; the only operational flag is
	``--force`` (overwrite existing outputs without editing the config). Overwrite
	policy is not part of the config — it never affects the trajectories (vault/19).
	"""
	file_values: dict[str, Any] = {}
	if config is not None:
		try:
			file_values = load_toml(config)
		except ValueError as exc:
			raise typer.BadParameter(str(exc)) from exc

	try:
		run = RunConfig.resolve(file_values, {})
	except ConfigError as exc:
		typer.echo(f"ERROR: {exc}", err=True)
		raise typer.Exit(code=2) from exc

	# --- Fail-fast checks that need the filesystem but not the (costly) video open. ---
	if not run.input.video.is_file():
		raise typer.BadParameter(f"input video {run.input.video} does not exist or is not a file.")
	# Path-type guards the per-key flags used to enforce (dir_okay/file_okay) before they
	# were removed (vault/19): file outputs must not be directories, the anchors dir not a
	# file — caught here cleanly rather than as an opaque writer error later.
	for label, path in (
		("export.out", run.export.out),
		("export.transform_csv", run.export.transform_csv),
		("run.timing_csv", run.options.timing_csv),
	):
		if path is not None and path.is_dir():
			raise typer.BadParameter(f"{label} must be a file path, not a directory: {path}.")
	if run.export.anchors_dir is not None and run.export.anchors_dir.is_file():
		raise typer.BadParameter(
			f"export.anchors_dir must be a directory, not a file: {run.export.anchors_dir}."
		)
	if (
		run.options.timing_csv is not None
		and run.export.out.resolve() == run.options.timing_csv.resolve()
	):
		raise typer.BadParameter("run.timing_csv must differ from export.out.")
	if run.export.transform_csv is not None and run.export.transform_csv.resolve() in {
		run.export.out.resolve(),
		run.options.timing_csv.resolve() if run.options.timing_csv is not None else None,
	}:
		raise typer.BadParameter(
			"export.transform_csv must differ from export.out and run.timing_csv."
		)
	_prepare_output_path(run.export.out, force=force)
	if run.options.timing_csv is not None:
		_prepare_output_path(run.options.timing_csv, force=force)
	if run.export.transform_csv is not None:
		_prepare_output_path(run.export.transform_csv, force=force)

	with _open_video(
		run.input.video,
		start_seconds=run.window.start_seconds,
		end_seconds=run.window.end_seconds,
		process_fps=run.input.process_fps or None,
	) as source:
		try:
			scale = run.calibration.resolve_scale(source.metadata)
		except (ValueError, ConfigError) as exc:
			# A non-positive altitude (e.g. an SRT with no usable values) surfaces from
			# the calibration chain; report it cleanly rather than as a traceback.
			raise typer.BadParameter(str(exc)) from exc
		# Coordinate stabilization (MVP1.9, vault/05_75_mvp1_9.md): the detector and
		# tracker run on the raw frame; the live ORB ego-motion transform is applied to
		# the detections (not the pixels) inside the pipeline. None when stabilization
		# is off. The ORB estimator is also the DetectionObserver (masking vehicles out
		# of its own feature extraction), and — when exporting anchors — notifies a queue
		# the anchor recorder drains.
		anchor_poses: list[Transform2D] = []
		emit_anchors = run.export.anchors_dir is not None
		ego_motion: EgoMotionEstimator | None = None
		detection_observer: DetectionObserver | None = None
		if run.ego_motion.enabled:
			orb = OrbEgoMotionEstimator(
				n_features=run.ego_motion.n_features,
				match_ratio=run.ego_motion.match_ratio,
				min_matches=run.ego_motion.min_matches,
				ransac_threshold=run.ego_motion.ransac_threshold,
				min_anchor_overlap=run.ego_motion.min_anchor_overlap,
				anchor_observer=(
					(lambda _index, pose: anchor_poses.append(pose)) if emit_anchors else None
				),
			)
			ego_motion = orb
			detection_observer = orb
		det: Detector = _build_detector(run.detector, device=run.runtime.device)
		# When we stabilize coordinates ourselves, disable BoT-SORT's own camera-motion
		# compensation so it does not double-correct the already-stabilized boxes.
		tracker: Tracker = BoxmotBotSortTracker(
			source.metadata,
			det_thresh=run.tracker.det_thresh,
			compensate_camera_motion=not run.ego_motion.enabled,
		)
		# Map detections into the global frame when ego-motion is on; the pipeline's Null
		# default (pass-through) handles a non-stabilized run.
		stabilizer: DetectionStabilizer | None = (
			EgoMotionStabilizer() if run.ego_motion.enabled else None
		)
		with (
			_timing_sink(run.options.timing_csv) as sink,
			_transform_sink(run.export.transform_csv) as transform_sink,
			_anchor_sink(run.export.anchors_dir, video_label=str(run.input.video)) as anchor_sink,
		):
			# Per-step timing wraps each port once per frame (vault/15). detect/track/record
			# always run; observe/ego-motion/stabilize only on a stabilized run.
			if sink is not None:
				det = TimedDetector(det, sink)
				tracker = TimedTracker(tracker, sink)
				if detection_observer is not None:
					detection_observer = TimedDetectionObserver(detection_observer, sink)
				if ego_motion is not None:
					ego_motion = TimedEgoMotion(ego_motion, sink)  # innermost: times the ORB work
				if stabilizer is not None:
					stabilizer = TimedStabilizer(stabilizer, sink)
			# Tee the per-frame transform and/or export anchors; these wrap *outside*
			# TimedEgoMotion so their I/O is not counted as ego-motion time. The pipeline
			# stays untouched; the concrete ORB keeps serving the observer + anchor queue.
			pipeline_ego_motion: EgoMotionEstimator | None = ego_motion
			if transform_sink is not None and pipeline_ego_motion is not None:
				pipeline_ego_motion = RecordingEgoMotionEstimator(
					pipeline_ego_motion, transform_sink
				)
			if anchor_sink is not None and pipeline_ego_motion is not None:
				pipeline_ego_motion = AnchorRecordingEgoMotionEstimator(
					pipeline_ego_motion, anchor_poses, anchor_sink
				)
			# The track record is the run's output. The pipeline owns its lifecycle
			# (open on enter, close on exit), so it is passed unopened.
			track_sink: TrackSink = ParquetTrackSink(run.export.out, source.metadata, scale=scale)
			if sink is not None:
				track_sink = TimedTrackSink(track_sink, sink)
			pipeline = TrajectoryPipeline(
				video=source,
				detector=det,
				tracker=tracker,
				sink=track_sink,
				reporter=ConsoleProgressReporter(),
				detection_observer=detection_observer,
				stabilizer=stabilizer,
				ego_motion=pipeline_ego_motion,
			)
			n_frames = pipeline.run()

	typer.echo(
		f"Recorded {n_frames} frames -> {run.export.out} (scale={scale} m/px). "
		"Run tratrac-postprocess on it to produce a .trj."
	)


@contextmanager
def _open_video(
	video: Path,
	*,
	start_seconds: float | None,
	end_seconds: float | None,
	process_fps: float | None,
) -> Iterator[OpenCvVideoSource]:
	"""Open the (optionally windowed, optionally rate-capped) video source.

	Translates range ``ValueError``s raised while opening — e.g. a start past the
	video's end — into a clean ``typer.BadParameter``. Exceptions from the
	processing body pass through untouched.
	"""
	with ExitStack() as stack:
		try:
			source = OpenCvVideoSource(
				video,
				start_seconds=start_seconds,
				end_seconds=end_seconds,
				process_fps=process_fps,
			)
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


@contextmanager
def _anchor_sink(out_dir: Path | None, *, video_label: str) -> Iterator[AnchorSink | None]:
	"""Yield an anchor PNG+manifest sink when a directory is given, else ``None`` (off)."""
	if out_dir is None:
		yield None
		return
	with AnchorManifestSink(out_dir, video_label=video_label) as sink:
		yield sink


if __name__ == "__main__":
	app()
