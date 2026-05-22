#!/usr/bin/env python3
"""Probe RT-DETR raw output on a single video frame.

Bypasses tratrac's confidence threshold and class filter so you can see every
detection the model emits — useful for diagnosing whether vehicles are being
missed by the detector or just suppressed downstream.

Usage:
	uv run python scripts/probe_detector.py VIDEO [--frame N] [--checkpoint MODEL]
		[--threshold T] [--top N] [--out PATH]

Run after the main pipeline has finished to avoid CPU contention.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, RTDetrForObjectDetection

# COCO labels that map to tratrac's VehicleClass — highlighted in green in the dump.
_VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}
# Other classes worth noticing in aerial views — possible misclassifications of cars.
_NEAR_VEHICLE_LABELS = {"boat", "train", "airplane", "bicycle", "person"}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("video", type=Path, help="Input video file.")
	parser.add_argument("--frame", type=int, default=0, help="Frame index to probe (0-based).")
	parser.add_argument(
		"--checkpoint",
		default="PekingU/rtdetr_r18vd",
		help="HuggingFace RT-DETR checkpoint.",
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

	# --- Read the chosen frame ---
	cap = cv2.VideoCapture(str(args.video))
	if not cap.isOpened():
		print(f"Cannot open video: {args.video}", file=sys.stderr)
		return 1
	cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
	ok, bgr = cap.read()
	cap.release()
	if not ok or bgr is None:
		print(f"Could not read frame {args.frame}", file=sys.stderr)
		return 1

	height, width = bgr.shape[:2]
	print(f"Frame {args.frame}: {width}x{height} px")
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(rgb)

	# --- Load model ---
	print(f"Loading {args.checkpoint} (CPU)…")
	processor = AutoImageProcessor.from_pretrained(args.checkpoint)
	model = RTDetrForObjectDetection.from_pretrained(args.checkpoint).to("cpu")
	model.eval()
	id2label: dict[int, str] = model.config.id2label

	# --- Run detection at the requested (low) threshold ---
	inputs = processor(images=image, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**inputs)
	target_size = torch.tensor([[image.height, image.width]])
	processed = processor.post_process_object_detection(
		outputs, target_sizes=target_size, threshold=args.threshold
	)[0]

	scores = [float(s) for s in processed["scores"].tolist()]
	labels = [int(lid) for lid in processed["labels"].tolist()]
	boxes = [[float(c) for c in b] for b in processed["boxes"].tolist()]

	detections = sorted(
		(
			{
				"score": s,
				"label": id2label[lid],
				"box": b,
			}
			for s, lid, b in zip(scores, labels, boxes, strict=True)
		),
		key=lambda d: float(d["score"]),
		reverse=True,
	)

	# --- Print top N ---
	print(f"\n{len(detections)} detections at threshold >= {args.threshold}.")
	if detections:
		print(f"\n== Top {min(args.top, len(detections))} by score ==")
		for i, d in enumerate(detections[: args.top]):
			x1, y1, x2, y2 = d["box"]
			label = d["label"]
			score = d["score"]
			flag = ""
			if label in _VEHICLE_LABELS:
				flag = "  <- vehicle"
			elif label in _NEAR_VEHICLE_LABELS:
				flag = "  <- near-vehicle (possible misclassified car)"
			print(
				f"  {i + 1:>3}. score={score:.3f}  label={label:<14}  "
				f"box=({x1:7.1f},{y1:7.1f})-({x2:7.1f},{y2:7.1f}){flag}"
			)

	# --- Class counts ---
	counts = Counter(d["label"] for d in detections)
	vehicle_total = sum(c for label, c in counts.items() if label in _VEHICLE_LABELS)
	near_total = sum(c for label, c in counts.items() if label in _NEAR_VEHICLE_LABELS)
	print("\n== Class counts ==")
	for label, count in counts.most_common():
		marker = ""
		if label in _VEHICLE_LABELS:
			marker = "  (vehicle)"
		elif label in _NEAR_VEHICLE_LABELS:
			marker = "  (near-vehicle)"
		print(f"  {label:<20} {count}{marker}")
	print(f"\nVehicle detections: {vehicle_total}")
	print(f"Near-vehicle:       {near_total}")
	print(f"Other:              {len(detections) - vehicle_total - near_total}")

	# --- Draw annotated frame ---
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)
	try:
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
	except OSError:
		font = ImageFont.load_default()

	for d in detections:
		x1, y1, x2, y2 = d["box"]
		label = d["label"]
		score = d["score"]
		if label in _VEHICLE_LABELS:
			color = (0, 200, 0)
		elif label in _NEAR_VEHICLE_LABELS:
			color = (255, 165, 0)
		else:
			color = (200, 200, 0)
		draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
		draw.text((x1, max(y1 - 14, 0)), f"{label} {score:.2f}", fill=color, font=font)

	out_path: Path = args.out or args.video.parent / f"{args.video.stem}_frame{args.frame}.png"
	annotated.save(out_path)
	print(f"\nAnnotated frame -> {out_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
