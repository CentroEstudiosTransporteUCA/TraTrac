"""Typer CLI entry point for TraTrac.

A run is fully described by a persisted ``RunConfig`` (see
``tratrac.application.config``, whose module docstring is the schema/design reference).
There are no built-in defaults and **no per-key override flags**: every value comes
from the ``--config`` TOML, and a missing value fails the run listing exactly what is
absent. The sole flag is ``--force`` (overwrite existing outputs) — overwrite
policy is *not* a config key, since it never affects the trajectories. A complete
config replays with just ``--config``.

The run is **perception only**: it writes the track record (the raw tracked
measurements, the run's canonical output). It does not produce an SSAM ``.trj`` —
run ``tratrac-postprocess`` on the record to filter/smooth it into a ``.trj`` (src/tratrac/application/SMOOTHING.md).
This run never resolves any geometric transform itself: ``input.transforms_in``
must name a ``tratrac-preprocess estimate`` run's transforms file (which also
exported the keyframe-anchor PNGs an operator draws exclusion zones/calibration
correspondences on) — a static camera's file simply has no ego-motion rows, so
its stabilization stage is the identity. See
infrastructure/video/ego_motion_orb.py's module docstring.

Validation without running (``--check``)
=========================================

``tratrac --config run.toml --check [--json]`` validates a config and exits without
touching the pipeline — the point is to make this module the single source of truth
for validation, instead of a UI (the URBAn Tauri client) reimplementing range/coherence
rules in JS/Rust. It lives as a **flag on the single `process` command**, not a new
subcommand, keeping the config-only CLI design intact. ``--force`` is irrelevant under
``--check`` (no writes happen) and is ignored.

``--json`` emits a machine-readable report to **stdout**; without it, problems print
human-readable to stderr. Exit code ``0`` means the config is valid, ``2`` means invalid
(the same code ``ConfigError`` already uses elsewhere). The JSON shape::

    {
      "ok": false,
      "problems": [
        "detector.conf must be in [0.0, 1.0], got 1.4.",
        "input.transforms_in: file does not exist: out/highway_run3.transforms.jsonl.",
        "export.out must be a file path, not a directory: out/."
      ]
    }

``problems`` is a flat list of human strings — the same messages ``ConfigError`` and the
CLI's own guards already produce, no separate message taxonomy. A client shows them
verbatim and keys its enable/disable state off the exit code (``ok == (problems == [])``).

Three layers run, cheapest first, each gated on the previous parsing successfully:

- **L1 — TOML parse** (``load_toml``). A syntax error is a single fatal problem
  (``"config: <msg>"``); resolution can't proceed past it.
- **L2 — ``RunConfig.resolve``** (the bulk, already aggregated). Missing keys, type
  errors, ranges, and ``input.transforms_in`` existing as a file — always required, no
  ``enabled`` toggle to gate it, since ``tratrac`` never resolves scale, ego-motion, or
  a world-projection homography itself. This is exactly what a config-editor UI needs.
- **L3 — ``static_run_problems(run)``** (only reached if L1+L2 produce a ``run``): the
  post-resolve filesystem guards — video file exists, output path-types (file vs.
  directory), path collisions (``export.out`` ≠ ``run.timing_csv``). Cheap, no video
  decode. Both ``--check`` and a real run consume this same function, so both report
  every path problem aggregated in one pass rather than one-per-retry.

Deliberately **out of scope**, not silently skipped: anything that opens the video or
the network — detector checkpoint download/availability, device reachability. (GSD
scale/ego-motion/homography resolution moved out of ``tratrac`` entirely into
``tratrac-preprocess estimate``/``project``, which have no ``--check`` mode of their
own today.) These are run-time concerns, not config-shape concerns; a future
``--check-deep`` could add them, but v1 stays fast and offline enough to call on every
form edit. Also out of scope in v1: warnings for legal-but-unusual values (e.g.
``tracker.det_thresh >= detector.conf``) — errors-only for now, a ``"warnings"`` field
is a forward-compatible extension point if that's ever needed.
"""

from __future__ import annotations

import json
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
from tratrac.domain.ports import (
	DetectionStabilizer,
	Detector,
	EgoMotionEstimator,
	TimingSink,
	Tracker,
	TrackSink,
)
from tratrac.infrastructure.config.toml import load_toml
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.detection.yolo_obb import YoloObbDetector
from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector
from tratrac.infrastructure.progress.console import ConsoleProgressReporter
from tratrac.infrastructure.timing.csv import CsvTimingSink
from tratrac.infrastructure.timing.decorators import (
	TimedDetector,
	TimedEgoMotion,
	TimedStabilizer,
	TimedTracker,
	TimedTrackSink,
)
from tratrac.infrastructure.tracking.boxmot_bot_sort import BoxmotBotSortTracker
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink
from tratrac.infrastructure.transform.sink import PrecomputedEgoMotionEstimator, read_ego_motion
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
	check: Annotated[
		bool,
		typer.Option(
			"--check",
			help="Validate the config and exit (exit 0 valid, 2 invalid); never opens the "
			"video or writes outputs.",
		),
	] = False,
	json_output: Annotated[
		bool,
		typer.Option(
			"--json",
			help="With --check, emit a machine-readable JSON report to stdout.",
		),
	] = False,
) -> None:
	"""Track a video into a record file (run tratrac-postprocess on it to get a .trj).

	The run is driven entirely by ``--config``; the only operational flag is
	``--force`` (overwrite existing outputs without editing the config). Overwrite
	policy is not part of the config — it never affects the trajectories
	(``application/config.py``'s module docstring).

	``--check`` short-circuits to validation only: it parses the TOML, resolves the
	``RunConfig``, and runs the static filesystem guards, then reports every problem
	at once (``--json`` for a machine-readable report). It never opens the video,
	downloads a checkpoint, or writes anything — so a client can validate a config on
	every edit. ``--force``/``--json`` are irrelevant without their partner flag.
	"""
	if check:
		_run_check(config, as_json=json_output)
		return

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
	# Aggregated (every problem at once), so a run surfaces all path issues in one go —
	# the same shape ``--check`` reports and consistent with ConfigError. See
	# ``application/config.py``'s module docstring.
	static_problems = static_run_problems(run)
	if static_problems:
		_emit_check_report(static_problems, as_json=False)
		raise typer.Exit(code=2)
	_prepare_output_path(run.export.out, force=force)
	if run.options.timing_csv is not None:
		_prepare_output_path(run.options.timing_csv, force=force)

	try:
		ego_motion_table = read_ego_motion(run.input.transforms_in)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	with _open_video(
		run.input.video,
		start_seconds=run.window.start_seconds,
		end_seconds=run.window.end_seconds,
		process_fps=run.input.process_fps or None,
	) as source:
		# Coordinate stabilization (MVP1.9, see ego_motion_orb.py's module docstring): the
		# detector and tracker run on the raw frame; the ego-motion transform is applied to
		# the detections (not the pixels) inside the pipeline. `tratrac` never estimates
		# ego-motion itself -- `input.transforms_in` (a `tratrac-preprocess estimate` run's
		# output, "Detector-free ego-motion") is always required, so this run's single
		# detector pass never runs ORB. A static camera's file simply has no ego-motion
		# rows, so `PrecomputedEgoMotionEstimator` returns the identity for every frame --
		# `EgoMotionStabilizer` applying an identity is a no-op, so both are constructed
		# unconditionally rather than branching on whether the file happens to have rows.
		ego_motion: EgoMotionEstimator = PrecomputedEgoMotionEstimator(ego_motion_table)
		det: Detector = _build_detector(run.detector, device=run.runtime.device)
		# We always stabilize coordinates ourselves now (even if the stage turns out to be
		# the identity), so BoT-SORT's own camera-motion compensation is always disabled --
		# running both would risk double-correcting the same boxes.
		# is_obb is decided from the run's detector choice up front (Group A5/A9): boxmot
		# infers the det-array layout from only the first non-empty frame, which would
		# silently lock in AABB mode if that frame happened to have no detections.
		tracker: Tracker = BoxmotBotSortTracker(
			source.metadata,
			det_thresh=run.tracker.det_thresh,
			compensate_camera_motion=False,
			is_obb=run.detector.name is DetectorChoice.YOLO_OBB,
		)
		stabilizer: DetectionStabilizer = EgoMotionStabilizer()
		with _timing_sink(run.options.timing_csv) as sink:
			# Per-step timing wraps each port once per frame (src/tratrac/infrastructure/timing/STEP_TIMING.md).
			if sink is not None:
				det = TimedDetector(det, sink)
				tracker = TimedTracker(tracker, sink)
				ego_motion = TimedEgoMotion(ego_motion, sink)
				stabilizer = TimedStabilizer(stabilizer, sink)
			# The track record is the run's output. The pipeline owns its lifecycle
			# (open on enter, close on exit), so it is passed unopened.
			track_sink: TrackSink = ParquetTrackSink(run.export.out, source.metadata)
			if sink is not None:
				track_sink = TimedTrackSink(track_sink, sink)
			pipeline = TrajectoryPipeline(
				video=source,
				detector=det,
				tracker=tracker,
				sink=track_sink,
				reporter=ConsoleProgressReporter(),
				stabilizer=stabilizer,
				ego_motion=ego_motion,
			)
			n_frames = pipeline.run()

	typer.echo(
		f"Recorded {n_frames} frames -> {run.export.out}. "
		"Run tratrac-postprocess on it to produce a .trj."
	)


def static_run_problems(run: RunConfig) -> list[str]:
	"""Path problems detectable without opening the video, collected (not raised).

	The post-resolve guards that need only the filesystem (existence, path-type) or
	pure path-collision arithmetic — the layer that complements ``RunConfig.resolve``'s
	schema checks. Returned as a list so both the run and ``--check`` report them
	aggregated rather than one failure per re-run. The video decode, scale resolution,
	and checkpoint download stay out (those are run-time, not config-shape, concerns).
	"""
	problems: list[str] = []
	if not run.input.video.is_file():
		problems.append(f"input.video {run.input.video} does not exist or is not a file.")
	if not run.input.transforms_in.is_file():
		problems.append(
			f"input.transforms_in {run.input.transforms_in} does not exist or is not a file "
			"-- run tratrac-preprocess first."
		)
	# Path-type guards the per-key flags used to enforce (dir_okay/file_okay) before they
	# were removed (application/config.py's module docstring): file outputs must not be
	# directories — caught here cleanly rather than as an opaque writer error later.
	for label, path in (
		("export.out", run.export.out),
		("run.timing_csv", run.options.timing_csv),
	):
		if path is not None and path.is_dir():
			problems.append(f"{label} must be a file path, not a directory: {path}.")
	if (
		run.options.timing_csv is not None
		and run.export.out.resolve() == run.options.timing_csv.resolve()
	):
		problems.append("run.timing_csv must differ from export.out.")
	return problems


def _run_check(config: Path | None, *, as_json: bool) -> None:
	"""Validate ``config`` and exit (0 valid, 2 invalid) without running the pipeline.

	Layers, cheapest first, each gated on the previous parsing: TOML parse, then
	``RunConfig.resolve`` (the schema), then ``static_run_problems`` (the static path
	guards). All problems are aggregated into a single report so a client validating a
	config sees everything wrong at once.
	"""
	problems: list[str] = []
	file_values: dict[str, Any] = {}
	if config is not None:
		try:
			file_values = load_toml(config)
		except ValueError as exc:
			problems.append(f"config: {exc}")
	if not problems:
		try:
			run = RunConfig.resolve(file_values, {})
		except ConfigError as exc:
			problems.extend(exc.problems)
		else:
			problems.extend(static_run_problems(run))
	_emit_check_report(problems, as_json=as_json)
	raise typer.Exit(code=0 if not problems else 2)


def _emit_check_report(problems: list[str], *, as_json: bool) -> None:
	"""Print a validation report: JSON to stdout, or human-readable lines to stderr."""
	if as_json:
		typer.echo(json.dumps({"ok": not problems, "problems": problems}))
		return
	if not problems:
		typer.echo("config OK")
		return
	typer.echo("ERROR: invalid run configuration:", err=True)
	for problem in problems:
		typer.echo(f"  - {problem}", err=True)


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
	if detector.name is DetectorChoice.YOLO_OBB:
		return YoloObbDetector(
			checkpoint=detector.checkpoint,
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
