#!/usr/bin/env python3
"""Probe a detector's raw output on a single video frame, bypassing tratrac's downstream
confidence threshold / class filter so every detection the model emits is visible — useful for
diagnosing whether vehicles are being missed by the detector itself or just suppressed
downstream, and for before/after comparisons when swapping detectors (see
``src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md``).

Two backends:

- ``rt_detr`` — a HuggingFace RT-DETR checkpoint (``transformers``). This is how the original
  MVP1 emergency-swap decision was made: COCO-pretrained RT-DETR-R18 was confirmed here to
  misclassify aerial cars as ``bird``/``traffic light``/other COCO classes.
- ``yolo`` — any ``ultralytics`` YOLO checkpoint, detect *or* OBB task (auto-detected from the
  result). Point ``--checkpoint`` at a local ``.pt`` file or an ultralytics-hub name
  (auto-downloads), or pass ``--repo-id``/``--filename`` to pull a checkpoint hosted as a raw
  file on the HuggingFace Hub (the shape the MVP1 YOLOv8-VisDrone checkpoint is hosted in).

Standalone by design (this repo's `scripts/` convention): does not import the `tratrac` package,
so it keeps working even when the package is broken. Only class-name heuristics are shared
conceptually with the adapters, not code.

Usage:
	uv run python scripts/probe_detector.py VIDEO --backend rt_detr [--frame N]
		[--checkpoint MODEL] [--threshold T] [--top N] [--out PATH]
	uv run python scripts/probe_detector.py VIDEO --backend yolo [--frame N]
		[--checkpoint MODEL.pt] [--repo-id ID --filename NAME]
		[--threshold T] [--top N] [--out PATH]

Run after the main pipeline has finished to avoid CPU contention.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

# COCO labels that map to tratrac's VehicleClass — highlighted in green in the dump.
_VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}
# Other COCO classes worth noticing in aerial views — confirmed misclassifications of cars
# during the original RT-DETR-vs-YOLOv8 MVP1 comparison (see DETECTOR_CHOICE.md).
_NEAR_VEHICLE_LABELS = {"boat", "train", "airplane", "bicycle", "person", "bird", "traffic light"}
# Substring match against arbitrary (non-COCO) detector vocabularies — VisDrone, UAV-OBB,
# DroneVehicle, etc. all use different label sets, so exact-match sets don't generalize.
_VEHICLE_KEYWORDS = ("car", "truck", "bus", "van", "taxi", "motor", "bike", "vehicle", "freight")


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument("video", type=Path, help="Input video file.")
	parser.add_argument(
		"--backend", choices=("rt_detr", "yolo"), default="rt_detr", help="Detector to probe."
	)
	parser.add_argument("--frame", type=int, default=0, help="Frame index to probe (0-based).")
	parser.add_argument(
		"--checkpoint",
		default=None,
		help="HuggingFace RT-DETR checkpoint (rt_detr backend, default PekingU/rtdetr_r18vd) or "
		"a local path / ultralytics-hub name (yolo backend).",
	)
	parser.add_argument(
		"--repo-id",
		default=None,
		help="yolo backend only: HuggingFace Hub repo id to download --filename from, for "
		"checkpoints hosted as a raw file rather than an ultralytics-hub name.",
	)
	parser.add_argument(
		"--filename", default=None, help="yolo backend only: filename within --repo-id."
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.0,
		help="Min score to keep (default 0.0 = show everything).",
	)
	parser.add_argument(
		"--top",
		type=int,
		default=30,
		help="Print only the top N detections (all are still drawn in the PNG).",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output PNG path (default: <video stem>_frame<N>.png next to the video).",
	)
	args = parser.parse_args()

	if not args.video.exists():
		print(f"Video not found: {args.video}", file=sys.stderr)
		return 1

	image = _read_frame(args.video, args.frame)
	if image is None:
		return 1

	if args.backend == "rt_detr":
		checkpoint = args.checkpoint or "PekingU/rtdetr_r18vd"
		detections = _probe_rt_detr(image, checkpoint, args.threshold)
	else:
		checkpoint = _resolve_yolo_checkpoint(args.checkpoint, args.repo_id, args.filename)
		if checkpoint is None:
			return 1
		detections = _probe_yolo(image, checkpoint, args.threshold)

	_report(detections, args.top)
	out_path = args.out or args.video.parent / f"{args.video.stem}_frame{args.frame}.png"
	_draw(image, detections).save(out_path)
	print(f"\nAnnotated frame -> {out_path}")
	return 0


def _read_frame(video: Path, frame_index: int) -> Image.Image | None:
	cap = cv2.VideoCapture(str(video))
	if not cap.isOpened():
		print(f"Cannot open video: {video}", file=sys.stderr)
		return None
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
	ok, bgr = cap.read()
	cap.release()
	if not ok or bgr is None:
		print(f"Could not read frame {frame_index}", file=sys.stderr)
		return None
	height, width = bgr.shape[:2]
	print(f"Frame {frame_index}: {width}x{height} px")
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	return Image.fromarray(rgb)


# A detection normalized across backends: an AABB (x1,y1,x2,y2) always; ``polygon`` additionally
# set (4 corner points) when the source was an oriented box, for drawing/reporting the rotation.
class _Det:
	__slots__ = ("box", "label", "polygon", "score")

	def __init__(
		self,
		score: float,
		label: str,
		box: tuple[float, float, float, float],
		polygon: Sequence[tuple[float, float]] | None = None,
	) -> None:
		self.score = score
		self.label = label
		self.box = box
		self.polygon = tuple(polygon) if polygon is not None else None


def _probe_rt_detr(image: Image.Image, checkpoint: str, threshold: float) -> list[_Det]:
	import torch
	from transformers import AutoImageProcessor, RTDetrForObjectDetection

	print(f"Loading {checkpoint} (CPU)…")
	processor = AutoImageProcessor.from_pretrained(checkpoint)
	model: Any = RTDetrForObjectDetection.from_pretrained(checkpoint).to("cpu")
	model.eval()
	id2label: dict[int, str] = model.config.id2label

	inputs = processor(images=image, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**inputs)
	target_size = torch.tensor([[image.height, image.width]])
	processed = processor.post_process_object_detection(
		outputs, target_sizes=target_size, threshold=threshold
	)[0]

	scores = [float(s) for s in processed["scores"].tolist()]
	labels = [int(lid) for lid in processed["labels"].tolist()]
	boxes = [[float(c) for c in b] for b in processed["boxes"].tolist()]
	return [
		_Det(score=s, label=id2label[lid], box=(b[0], b[1], b[2], b[3]))
		for s, lid, b in zip(scores, labels, boxes, strict=True)
	]


def _resolve_yolo_checkpoint(
	checkpoint: str | None, repo_id: str | None, filename: str | None
) -> str | None:
	if repo_id is not None or filename is not None:
		if repo_id is None or filename is None:
			print("--repo-id and --filename must be given together.", file=sys.stderr)
			return None
		from huggingface_hub import hf_hub_download

		return str(hf_hub_download(repo_id=repo_id, filename=filename))
	if checkpoint is None:
		print(
			"yolo backend needs --checkpoint (local/hub name) or --repo-id + --filename.",
			file=sys.stderr,
		)
		return None
	return checkpoint


def _probe_yolo(image: Image.Image, checkpoint: str, threshold: float) -> list[_Det]:
	import numpy as np
	from ultralytics import YOLO

	print(f"Loading {checkpoint} (CPU)…")
	model: Any = YOLO(checkpoint)
	results = model.predict(
		source=np.array(image), conf=max(threshold, 1e-4), device="cpu", verbose=False
	)
	if not results:
		return []
	first = results[0]
	names: dict[int, str] = first.names

	obb = getattr(first, "obb", None)
	if obb is not None and obb.shape[0] > 0:
		corners = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
		confs = obb.conf.cpu().numpy()
		cls_ids = obb.cls.cpu().numpy().astype(int)
		detections: list[_Det] = []
		for poly, conf, cls_id in zip(corners, confs, cls_ids, strict=True):
			xs, ys = poly[:, 0], poly[:, 1]
			detections.append(
				_Det(
					score=float(conf),
					label=names[int(cls_id)],
					box=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
					polygon=[(float(x), float(y)) for x, y in poly],
				)
			)
		return sorted(detections, key=lambda d: d.score, reverse=True)

	boxes = first.boxes
	if boxes is None or boxes.shape[0] == 0:
		return []
	xyxy = boxes.xyxy.cpu().numpy()
	confs = boxes.conf.cpu().numpy()
	cls_ids = boxes.cls.cpu().numpy().astype(int)
	detections = [
		_Det(score=float(conf), label=names[int(cls_id)], box=tuple(float(c) for c in xy))
		for xy, conf, cls_id in zip(xyxy, confs, cls_ids, strict=True)
	]
	return sorted(detections, key=lambda d: d.score, reverse=True)


def _classify(label: str) -> str:
	""" "vehicle" / "near-vehicle" / "" for a detection label, across arbitrary vocabularies."""
	lowered = label.lower()
	if lowered in _VEHICLE_LABELS or any(k in lowered for k in _VEHICLE_KEYWORDS):
		return "vehicle"
	if lowered in _NEAR_VEHICLE_LABELS:
		return "near-vehicle"
	return ""


def _report(detections: list[_Det], top: int) -> None:
	print(f"\n{len(detections)} detections.")
	if detections:
		print(f"\n== Top {min(top, len(detections))} by score ==")
		for i, d in enumerate(detections[:top]):
			x1, y1, x2, y2 = d.box
			kind = _classify(d.label)
			flag = f"  <- {kind}" if kind else ""
			shape = " (oriented)" if d.polygon is not None else ""
			print(
				f"  {i + 1:>3}. score={d.score:.3f}  label={d.label:<16}  "
				f"box=({x1:7.1f},{y1:7.1f})-({x2:7.1f},{y2:7.1f}){shape}{flag}"
			)

	counts = Counter(d.label for d in detections)
	vehicle_total = sum(c for label, c in counts.items() if _classify(label) == "vehicle")
	near_total = sum(c for label, c in counts.items() if _classify(label) == "near-vehicle")
	print("\n== Class counts ==")
	for label, count in counts.most_common():
		kind = _classify(label)
		marker = f"  ({kind})" if kind else ""
		print(f"  {label:<20} {count}{marker}")
	print(f"\nVehicle detections: {vehicle_total}")
	print(f"Near-vehicle:       {near_total}")
	print(f"Other:              {len(detections) - vehicle_total - near_total}")


def _draw(image: Image.Image, detections: list[_Det]) -> Image.Image:
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)
	try:
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
	except OSError:
		font = ImageFont.load_default()

	for d in detections:
		kind = _classify(d.label)
		color = (
			(0, 200, 0)
			if kind == "vehicle"
			else (255, 165, 0)
			if kind == "near-vehicle"
			else (200, 200, 0)
		)
		if d.polygon is not None:
			draw.line([*d.polygon, d.polygon[0]], fill=color, width=2)
			label_at = d.polygon[0]
		else:
			x1, y1, x2, y2 = d.box
			draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
			label_at = (x1, y1)
		draw.text(
			(label_at[0], max(label_at[1] - 14, 0)),
			f"{d.label} {d.score:.2f}",
			fill=color,
			font=font,
		)
	return annotated


if __name__ == "__main__":
	sys.exit(main())
