"""Typer CLI entry point for TraTrac."""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tratrac.application.orientation import OrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.calibration.drone_specs import lookup
from tratrac.calibration.gsd import ground_sample_distance
from tratrac.calibration.srt_parser import mean_altitude
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.ports import Detector
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
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
	out: Annotated[Path, typer.Option("--out", "-o", help="Output .trj path.")],
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
) -> None:
	"""Process a video into an SSAM .trj trajectory file."""
	with OpenCvVideoSource(video) as source:
		scale = _resolve_scale(
			video_path=video,
			metadata=source.metadata,
			meters_per_pixel=meters_per_pixel,
			drone_model=drone_model,
			altitude_m=altitude_m,
			srt_path=srt,
		)
		det: Detector = _build_detector(detector, checkpoint=checkpoint, device=device, conf=conf)
		tracker = BoxmotBotSortTracker(source.metadata)
		exporter = SsamTrjExporter(out, source.metadata, scale=scale)
		orientation = OrientationEstimator(meters_per_pixel=scale)
		pipeline = TrajectoryPipeline(
			video=source,
			detector=det,
			tracker=tracker,
			exporter=exporter,
			orientation=orientation,
		)
		n_frames = pipeline.run()

	typer.echo(f"Processed {n_frames} frames -> {out} (scale={scale} m/px)")


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
	3. Fall back to 1.0 with a stderr warning
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

	typer.echo(
		"WARNING: no calibration supplied (--meters-per-pixel or --drone-model + altitude). "
		"Falling back to scale=1.0 — output .trj will be syntactically valid SSAM but "
		"physically meaningless (MVP1 behaviour).",
		err=True,
	)
	return 1.0


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


if __name__ == "__main__":
	app()
