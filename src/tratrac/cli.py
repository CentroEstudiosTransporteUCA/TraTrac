"""Typer CLI entry point for TraTrac."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tratrac.application.orientation import OrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
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
) -> None:
	"""Process a video into an SSAM .trj trajectory file."""
	with OpenCvVideoSource(video) as source:
		det: Detector = _build_detector(detector, checkpoint=checkpoint, device=device, conf=conf)
		tracker = BoxmotBotSortTracker(source.metadata)
		exporter = SsamTrjExporter(out, source.metadata)
		orientation = OrientationEstimator()
		pipeline = TrajectoryPipeline(
			video=source,
			detector=det,
			tracker=tracker,
			exporter=exporter,
			orientation=orientation,
		)
		n_frames = pipeline.run()

	typer.echo(f"Processed {n_frames} frames -> {out}")


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
