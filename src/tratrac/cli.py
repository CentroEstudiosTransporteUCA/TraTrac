"""Typer CLI entry point for TraTrac."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tratrac.application.orientation import EmaOrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.calibration.drone_specs import known_models, lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.calibration.srt_parser import mean_altitude
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.ports import (
	Detector,
	OrientationEstimator,
	TimingSink,
	Tracker,
	TrajectoryExporter,
)
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


class DetectorChoice(StrEnum):
	"""Available detector adapters.

	`yolov8_visdrone` is the MVP1 emergency default — community YOLOv8 fine-tuned
	on VisDrone, picked because COCO-pretrained RT-DETR fails on aerial inputs.
	`rt_detr` stays available; once a fine-tuned aerial RT-DETR checkpoint exists,
	we make it the default again and drop the YOLO option.
	"""

	YOLOV8_VISDRONE = "yolov8_visdrone"
	RT_DETR = "rt_detr"


app = typer.Typer(
	name="tratrac",
	help="Vehicle tracking and trajectory export for aerial video.",
	no_args_is_help=True,
)


@app.command()
def process(
	video: Annotated[
		Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Input video.")
	],
	out: Annotated[Path, typer.Option("--out", "-o", dir_okay=False, help="Output .trj path.")],
	detector: Annotated[
		DetectorChoice,
		typer.Option("--detector", help="Which detector adapter to use."),
	] = DetectorChoice.YOLOV8_VISDRONE,
	conf: Annotated[
		float, typer.Option("--conf", help="Detection confidence threshold.", min=0.0, max=1.0)
	] = 0.25,
	checkpoint: Annotated[
		str,
		typer.Option(
			"--checkpoint",
			help="Checkpoint name (HF repo id for either detector).",
		),
	] = "",
	device: Annotated[str, typer.Option("--device", help="torch device (cpu, cuda, mps).")] = "cpu",
	meters_per_pixel: Annotated[
		float,
		typer.Option(
			"--meters-per-pixel",
			help="Direct GSD override (metres per pixel). Skips drone metadata lookup if set.",
			min=0.0,
		),
	] = 0.0,
	drone_model: Annotated[
		str,
		typer.Option(
			"--drone-model",
			help="Drone model key (e.g. mavic_3) for sensor + focal lookup. See drone_specs registry.",
		),
	] = "",
	altitude_m: Annotated[
		float,
		typer.Option(
			"--altitude",
			help="Flight altitude in metres AGL. Used with --drone-model when no SRT file is available.",
			min=0.0,
		),
	] = 0.0,
	srt: Annotated[
		Path | None,
		typer.Option(
			"--srt",
			help="Path to DJI .SRT sidecar with per-frame altitude (default: <video>.SRT next to the video).",
		),
	] = None,
	timing_csv: Annotated[
		Path | None,
		typer.Option(
			"--timing-csv",
			dir_okay=False,
			help="Write per-frame pipeline step timings to this CSV path (profiling; off by default).",
		),
	] = None,
	start: Annotated[
		str | None,
		typer.Option(
			"--start",
			help="Start of the analysis window as a timecode (HH:MM:SS(.ms), MM:SS, or SS). "
			"Default: video start.",
		),
	] = None,
	end: Annotated[
		str | None,
		typer.Option(
			"--end",
			help="End of the analysis window (inclusive) as a timecode. Default: video end.",
		),
	] = None,
	timestep_precision: Annotated[
		float | None,
		typer.Option(
			"--timestep-precision",
			help="Minimum seconds between exported TIMESTEP records (thins the .trj). "
			"Detection and tracking still run on every frame. Default: every frame.",
		),
	] = None,
) -> None:
	"""Process a video into an SSAM .trj trajectory file."""
	# --- Fail-fast input validation. Everything here is checkable without the
	# video, so reject bad input before the overwrite prompt and the costly
	# video open + detector load. ---
	start_seconds = _parse_timecode(start) if start is not None else None
	end_seconds = _parse_timecode(end) if end is not None else None
	if end_seconds is not None and end_seconds <= 0:
		raise typer.BadParameter("--end must be greater than zero.")
	if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
		raise typer.BadParameter("--end must be after --start.")
	_validate_device(device)
	_validate_drone_model(drone_model)
	_validate_timestep_precision(timestep_precision)
	if meters_per_pixel <= 0.0 and not drone_model:
		raise _no_calibration_error()
	if timing_csv is not None and out.resolve() == timing_csv.resolve():
		raise typer.BadParameter("--timing-csv must differ from --out.")
	# Confirm/prepare outputs (create parents, confirm overwrite) before opening.
	_prepare_output_path(out)
	if timing_csv is not None:
		_prepare_output_path(timing_csv)
	with _open_video(video, start_seconds=start_seconds, end_seconds=end_seconds) as source:
		try:
			scale = _resolve_scale(
				video_path=video,
				metadata=source.metadata,
				meters_per_pixel=meters_per_pixel,
				drone_model=drone_model,
				altitude_m=altitude_m,
				srt_path=srt,
			)
		except ValueError as exc:
			# A non-positive altitude (e.g. an SRT with no usable values) surfaces
			# from the calibration chain as a domain ValueError; report it cleanly.
			raise typer.BadParameter(str(exc)) from exc
		det: Detector = _build_detector(detector, checkpoint=checkpoint, device=device, conf=conf)
		tracker: Tracker = BoxmotBotSortTracker(source.metadata)
		exporter: TrajectoryExporter = SsamTrjExporter(out, source.metadata, scale=scale)
		if timestep_precision is not None:
			# Decimate the TIMESTEP stream. Sits inside TimedExporter (below) so the
			# EXPORT step still records once per processed frame (see vault/15).
			exporter = DecimatingTrajectoryExporter(
				exporter,
				min_interval_seconds=timestep_precision,
				fps=source.metadata.fps,
			)
		orientation: OrientationEstimator = EmaOrientationEstimator(meters_per_pixel=scale)
		with _timing_sink(timing_csv) as sink:
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

	typer.echo(f"Processed {n_frames} frames -> {out} (scale={scale} m/px)")


_DEVICE_RE = re.compile(r"cpu|mps|cuda(:\d+)?")


def _validate_device(device: str) -> None:
	"""Reject device strings torch won't accept. Heuristic: cpu / mps / cuda[:N]."""
	if _DEVICE_RE.fullmatch(device) is None:
		raise typer.BadParameter(
			f"Unsupported --device {device!r}; expected cpu, mps, or cuda[:N] (e.g. cuda:0)."
		)


def _validate_drone_model(drone_model: str) -> None:
	"""Reject an unknown --drone-model up front, before the video opens."""
	if drone_model and drone_model.lower() not in known_models():
		known = ", ".join(known_models())
		raise typer.BadParameter(f"Unknown drone model {drone_model!r}. Known: {known}.")


# Above this, exported timesteps get too sparse for SSAM conflict analysis
# (vault/04: sub-second, ~0.1s, is the practical minimum). Still valid, so warn.
_COARSE_TIMESTEP_WARNING_SECONDS = 0.5


def _validate_timestep_precision(seconds: float | None) -> None:
	"""Reject a non-positive interval; warn when it is too coarse for SSAM.

	A coarse interval still yields a syntactically valid .trj, so this warns
	rather than errors. See vault/04_ssam_format.md and vault/18_timestep_precision.md.
	"""
	if seconds is None:
		return
	if seconds <= 0.0:
		raise typer.BadParameter("--timestep-precision must be greater than zero.")
	if seconds > _COARSE_TIMESTEP_WARNING_SECONDS:
		typer.echo(
			f"WARNING: --timestep-precision {seconds}s is coarse; SSAM conflict analysis "
			"wants sub-second timesteps (~0.1s). The .trj stays valid but may be too "
			"sparse for surrogate-safety metrics.",
			err=True,
		)


@contextmanager
def _open_video(
	video: Path, *, start_seconds: float | None, end_seconds: float | None
) -> Iterator[OpenCvVideoSource]:
	"""Open the (optionally windowed) video source.

	Translates range ``ValueError``s raised while opening — e.g. a --start past
	the video's end — into a clean ``typer.BadParameter``. Exceptions from the
	processing body pass through untouched.
	"""
	with ExitStack() as stack:
		try:
			source = OpenCvVideoSource(video, start_seconds=start_seconds, end_seconds=end_seconds)
			stack.enter_context(source)
		except ValueError as exc:
			raise typer.BadParameter(str(exc)) from exc
		yield source


def _resolve_scale(
	*,
	video_path: Path,
	metadata: VideoMetadata,
	meters_per_pixel: float,
	drone_model: str,
	altitude_m: float,
	srt_path: Path | None,
) -> float:
	"""Resolve the GSD (metres per pixel) from CLI args, in priority order:

	1. Explicit `--meters-per-pixel` value
	2. `--drone-model` + (`--altitude` or DJI .SRT sidecar)

	Errors out if neither is supplied. An uncalibrated run would emit physically
	meaningless metric quantities (pixels pretended to be metres), so the CLI
	refuses rather than silently produce them. The scale=1.0 pixel fallback
	remains available in the library (``EmaOrientationEstimator`` /
	``SsamTrjExporter`` defaults) for callers that knowingly want it.

	``process`` rejects the no-calibration case up front; the guard here keeps
	this helper correct on its own (and shares the same error message).
	"""
	if meters_per_pixel > 0.0:
		return meters_per_pixel

	if drone_model:
		spec = lookup(drone_model)
		altitude = altitude_m if altitude_m > 0.0 else _altitude_from_srt(video_path, srt_path)
		return ground_sample_distance(
			sensor_width_mm=spec.sensor_width_mm,
			focal_length_mm=spec.focal_length_mm,
			altitude_m=altitude,
			image_width_pixels=metadata.width,
		)

	raise _no_calibration_error()


def _parse_timecode(value: str) -> float:
	"""Parse a timecode into seconds.

	Accepts ``SS(.ms)``, ``MM:SS(.ms)``, or ``HH:MM:SS(.ms)`` (e.g. ``12.5``,
	``1:30``, ``0:01:05.250``). Components must be non-negative. Raises
	``typer.BadParameter`` on malformed input so the CLI reports it cleanly.
	"""
	parts = value.strip().split(":")
	if len(parts) > 3:
		raise typer.BadParameter(f"Timecode has too many ':'-separated parts: {value!r}.")
	try:
		seconds = float(parts[-1])
		minutes = float(parts[-2]) if len(parts) >= 2 else 0.0
		hours = float(parts[-3]) if len(parts) == 3 else 0.0
	except ValueError:
		raise typer.BadParameter(
			f"Invalid timecode {value!r}; expected SS(.ms), MM:SS, or HH:MM:SS."
		) from None
	if seconds < 0 or minutes < 0 or hours < 0:
		raise typer.BadParameter(f"Timecode components must be non-negative: {value!r}.")
	return hours * 3600.0 + minutes * 60.0 + seconds


def _no_calibration_error() -> typer.Exit:
	"""Emit the no-calibration error and return the exception for the caller to raise."""
	typer.echo(
		"ERROR: no calibration supplied. Pass --meters-per-pixel, or --drone-model "
		"with --altitude or a DJI .SRT sidecar. Without a real GSD the exported .trj "
		"would carry physically meaningless metric values.",
		err=True,
	)
	return typer.Exit(code=2)


def _altitude_from_srt(video_path: Path, srt_path: Path | None) -> float:
	"""Resolve altitude from an explicit SRT path or the conventional sidecar location."""
	if srt_path is None:
		# DJI convention: same stem, .SRT extension, next to the video.
		candidates = [video_path.with_suffix(".SRT"), video_path.with_suffix(".srt")]
		for candidate in candidates:
			if candidate.exists():
				srt_path = candidate
				break
	if srt_path is None or not srt_path.exists():
		print(
			"ERROR: --drone-model given but neither --altitude nor a readable .SRT sidecar "
			"was found. Provide one or the other.",
			file=sys.stderr,
		)
		raise typer.Exit(code=2)
	return mean_altitude(srt_path)


def _build_detector(
	choice: DetectorChoice, *, checkpoint: str, device: str, conf: float
) -> Detector:
	if choice is DetectorChoice.RT_DETR:
		return RtDetrDetector(
			checkpoint=checkpoint or "PekingU/rtdetr_r18vd",
			device=device,
			score_threshold=conf,
		)
	if choice is DetectorChoice.YOLOV8_VISDRONE:
		return YoloV8VisDroneDetector(
			repo_id=checkpoint or "Mahadih534/YoloV8-VisDrone",
			device=device,
			score_threshold=conf,
		)
	raise ValueError(f"Unknown detector choice: {choice}")


def _prepare_output_path(path: Path) -> None:
	"""Make ``path`` writable: confirm overwrite if it exists, then create parents.

	The writers open with ``"w"``/``"wb"`` and would raise if the parent directory
	is missing. Overwrite confirmation is an interactive (CLI) concern, so it lives
	here rather than in the writers.
	"""
	if path.exists():
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
