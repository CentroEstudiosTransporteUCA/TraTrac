"""Visualize what the ORB ego-motion stabilizer does to a video, frame by frame.

Unlike the other scripts/ tools (pure-stdlib, package-independent), this one
*intentionally* imports the package: its whole point is to show the real
``OrbEgoMotionEstimator`` + warp (the exact code path the pipeline uses), so you
can eyeball the cumulative drift before deciding how to fix it.

Two modes:
  - default (pure ORB): fast, no model download, shows the UNMASKED estimator.
  - --mask: faithful to the pipeline — runs the real detector on each warped
    frame and feeds the detections back (observer) so ORB masks vehicles out,
    exactly as mvp19 did. Slower (~detector cost per frame).

It opens a window with [ ORIGINAL | WARPED-into-reference ] side by side and a
HUD showing the live cumulative transform (translation px / rotation deg / scale).
As drift accumulates you will see the warped panel slide/rotate and grow black
borders. Keys: q = quit, space = pause/step.

Run (live window, on your display):
  uv run python scripts/visualize_stabilization.py .resources/cruce_simple.mp4
  uv run python scripts/visualize_stabilization.py .resources/cruce_simple.mp4 --mask
Headless (write a side-by-side mp4 instead of a window):
  uv run python scripts/visualize_stabilization.py VIDEO --no-window --save out.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from tratrac.infrastructure.video.stabilized import _cv2_warp

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.infrastructure.video.ego_motion_orb import OrbEgoMotionEstimator


def _decompose(t: Transform2D) -> tuple[float, float, float]:
	"""(translation magnitude px, rotation deg, uniform scale) of a similarity."""
	translation = math.hypot(t.tx, t.ty)
	rotation = math.degrees(math.atan2(t.c, t.a))
	scale = math.hypot(t.a, t.c)
	return translation, rotation, scale


def _label(img: NDArray[np.uint8], text: str, y: int) -> None:
	"""Draw HUD text with a dark outline so it reads on any background."""
	cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
	cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("video", type=Path)
	parser.add_argument("--mask", action="store_true", help="Faithful masked mode (runs detector).")
	parser.add_argument("--start-frame", type=int, default=0)
	parser.add_argument("--max-frames", type=int, default=0, help="0 = whole clip.")
	parser.add_argument("--orb-features", type=int, default=2000)
	parser.add_argument("--orb-match-ratio", type=float, default=0.75)
	parser.add_argument("--orb-min-matches", type=int, default=10)
	parser.add_argument("--orb-ransac-threshold", type=float, default=3.0)
	parser.add_argument("--checkpoint", default="Mahadih534/YoloV8-VisDrone")
	parser.add_argument("--filename", default="visDrone.pt")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--display-scale", type=float, default=0.5, help="Resize the panel pair.")
	parser.add_argument("--delay", type=int, default=1, help="waitKey ms; 0 = step on keypress.")
	parser.add_argument("--save", type=Path, default=None, help="Write the side-by-side to mp4.")
	parser.add_argument("--no-window", action="store_true", help="Headless (no imshow).")
	parser.add_argument(
		"--diff",
		action="store_true",
		help="Add a 3rd panel: |warped - reference(frame 0)|. Static bg dark = aligned; "
		"glowing doubled edges = drift. The sensitive view for slow drift.",
	)
	parser.add_argument(
		"--amplify", type=float, default=4.0, help="Brightness gain for the --diff panel."
	)
	args = parser.parse_args()

	estimator = OrbEgoMotionEstimator(
		n_features=args.orb_features,
		match_ratio=args.orb_match_ratio,
		min_matches=args.orb_min_matches,
		ransac_threshold=args.orb_ransac_threshold,
	)
	detector = None
	if args.mask:
		from tratrac.infrastructure.detection.yolov8_visdrone import YoloV8VisDroneDetector

		detector = YoloV8VisDroneDetector(
			repo_id=args.checkpoint,
			filename=args.filename,
			device=args.device,
			score_threshold=args.conf,
		)

	cap = cv2.VideoCapture(str(args.video))
	if not cap.isOpened():
		raise SystemExit(f"Could not open {args.video}")
	if args.start_frame:
		cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.start_frame))

	writer = None
	paused = False
	index = args.start_frame
	processed = 0
	reference: NDArray[np.uint8] | None = None
	mode = "MASKED (faithful)" if args.mask else "PURE ORB (unmasked)"
	try:
		while True:
			if args.max_frames and processed >= args.max_frames:
				break
			ok, pixels = cap.read()
			if not ok:
				break
			transform = estimator.estimate(Frame(index=index, pixels=pixels))
			warped = _cv2_warp(pixels, transform)
			if reference is None:
				reference = warped.copy()  # frame 0: its transform is identity

			# Difference vs the FIXED reference, computed from the clean warp before
			# any overlay. On a static camera a perfect warp cancels the background
			# (only moving cars glow); drift makes static edges light up.
			diff_panel = None
			if args.diff:
				diff_panel = cv2.convertScaleAbs(cv2.absdiff(warped, reference), alpha=args.amplify)

			if detector is not None:
				detections = detector.detect(Frame(index=index, pixels=warped))
				estimator.observe(detections)
				for det in detections:
					b = det.bbox
					cv2.rectangle(
						warped,
						(int(b.x), int(b.y)),
						(int(b.x + b.width), int(b.y + b.height)),
						(0, 0, 255),
						2,
					)

			tmag, rot, scale = _decompose(transform)
			original = pixels.copy()
			_label(original, "ORIGINAL", 30)
			_label(warped, "WARPED -> reference", 30)
			_label(warped, f"frame {index}  [{mode}]", 60)
			_label(warped, f"translation {tmag:6.1f}px", 90)
			_label(warped, f"rotation {rot:+5.2f}deg  scale {scale:5.3f}", 120)

			panels = [original, warped]
			if diff_panel is not None:
				_label(diff_panel, "DIFF vs REF (dark=aligned)", 30)
				panels.append(diff_panel)
			combined = cv2.hconcat(panels)
			if args.display_scale != 1.0:
				combined = cv2.resize(combined, None, fx=args.display_scale, fy=args.display_scale)

			if args.save is not None:
				if writer is None:
					h, w = combined.shape[:2]
					fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
					writer = cv2.VideoWriter(
						str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
					)
				writer.write(combined)

			if not args.no_window:
				cv2.imshow("stabilization (ORIGINAL | WARPED)", combined)
				key = cv2.waitKey(0 if paused else args.delay) & 0xFF
				if key == ord("q"):
					break
				if key == ord(" "):
					paused = not paused

			index += 1
			processed += 1
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		if not args.no_window:
			cv2.destroyAllWindows()

	print(f"Processed {processed} frames. Final cumulative: {_decompose(estimator._cumulative)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
