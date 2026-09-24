#!/usr/bin/env python3
"""Propose candidate world-projection calibration points from visible road markings.

Group C4 (GitHub Issues): automatic road-geometry calibration, scoped to
**correspondence proposal only** — TraTrac finds candidate calibratable image points, a human
(via an external interactive tool, e.g. URBAn — the repo-boundary split this scope decision
settled on) confirms/adjusts them and supplies the real-world coordinates. This is deliberately
not a fully-automatic replacement: the cited method this follows (Popov, Trukhina & Vashkelis,
*Mobile Traffic Camera Calibration from Road Geometry for UAV-Based Traffic Surveillance*,
arXiv:2605.11900, 2026 — the source `DETECTOR_CHOICE.md`-style "cited source" for
"manual validation still outperforms automatic" already referenced in this project's planning
docs) reports that a fully automatic homography from line detection alone was "plausible but
geometrically imperfect," while human-in-the-loop calibration was the actually robust path.

**What this produces**: candidate straight-line road-marking segments (lane lines, crosswalk
stripe edges, road borders — all high-contrast, nearly-straight features against asphalt) via
classical edge + line detection, **not** semantic classification of which marking is which —
telling "this is a crosswalk stripe" from "this is a lane line" reliably needs more than Canny
+ Hough, and guessing wrong would be worse than not guessing. Each detected line's endpoints
become **image-only** candidate correspondences (schema-compatible with
``infrastructure/world/calibration.py``'s ``calibration.json``, ``world`` left ``null``) plus an
annotated overlay PNG, so a human can visually pick genuine calibration points and supply their
metric coordinates — the interactive step this script deliberately does not attempt.

Standalone (stdlib + cv2 + numpy only, no ``tratrac`` package import) per the ``scripts/``
convention — this only needs classical image processing, no domain types.

How it works: **Canny edge detection** + a **probabilistic Hough transform**
(``cv2.HoughLinesP``) over the still frame finds every fairly-straight edge, most of which
are *not* road markings (shadows, curbs, vehicle bodies, building edges, sidewalk pavers). A
**brightness + local-contrast filter** narrows that down to segments that actually look like a
painted marking: bright (``--min-brightness``, road paint is white/yellow) *and* meaningfully
brighter than the surface sampled a few pixels perpendicular to the line on either side
(``--min-contrast``/``--contrast-offset``). Brightness alone isn't enough — it also passes
uniformly light surfaces (sidewalks, concrete plazas, rooftops) with no real photometric edge;
requiring *local* contrast against the surrounding surface is what distinguishes "a painted
line on darker asphalt" from "a bright area." Surviving segments' endpoints become the
candidate correspondences plus the optional annotated overlay.

Validated against real footage: run against a real nadir intersection frame (1920x1080),
4,286 raw Hough segments -> 976 pass a brightness-only filter (too permissive — catches
sidewalks, plazas, rooftops) -> **32 pass brightness + local contrast**, visually confirmed
(overlay review) to correctly land on real crosswalk stripes, a lane divider line, and a road
border, with imperfect recall (some crosswalk stripes and one far-side crossing missed) and a
couple of false positives (a vehicle roofline, a shadow edge). This imperfect-but-genuinely-
informative result matches the cited paper's own "plausible but geometrically imperfect"
characterization of automatic line-based calibration — exactly why the output is scoped as a
proposal a human reviews, not a calibration TraTrac would use unattended.

Scope boundaries:

- Parameter tuning is a starting point, not validated across sites. Defaults were checked
  against one real frame from one intersection; different lighting, marking condition, or
  camera altitude would likely need different ``--canny-*``/``--min-brightness``/
  ``--min-contrast`` values. No sweep or cross-site validation was done.
- No semantic labeling of candidates (which marking is which) — Canny + Hough finds edges, not
  semantics, and a wrong semantic guess would mislead a reviewer worse than an unlabeled
  candidate would.
- No automatic world-coordinate assignment, by design, per the cited evidence above.
- No integration with ``tratrac-postprocess`` — this is a standalone pre-processing step whose
  output (after human review/completion) becomes an ordinary ``calibration.json`` for the
  already-shipped ``--calibration`` flag; no CLI flag was added to consume a "proposal" file
  directly, since it isn't usable until a human has filled in ``world`` anyway.
- The interactive confirm/adjust UI is explicitly out of scope per the resolved repo-boundary
  decision — that's URBAn's job.

Usage:
	uv run python scripts/propose_calibration.py FRAME.png --out proposal.json
		[--overlay overlay.png] [--canny-low N] [--canny-high N]
		[--hough-threshold N] [--min-line-length N] [--max-line-gap N]
		[--min-brightness N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_LINE_COLOR = (0, 255, 0)  # BGR green, drawn on the overlay for visual review.


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument("frame", type=Path, help="A single still frame (e.g. an exported anchor).")
	parser.add_argument("--out", type=Path, required=True, help="Output proposal JSON.")
	parser.add_argument(
		"--overlay", type=Path, default=None, help="Optional annotated PNG for visual review."
	)
	parser.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold.")
	parser.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold.")
	parser.add_argument(
		"--hough-threshold", type=int, default=40, help="HoughLinesP vote threshold."
	)
	parser.add_argument(
		"--min-line-length", type=int, default=30, help="HoughLinesP minimum segment length, px."
	)
	parser.add_argument(
		"--max-line-gap", type=int, default=8, help="HoughLinesP maximum gap to bridge, px."
	)
	parser.add_argument(
		"--min-brightness",
		type=int,
		default=150,
		help="Reject a candidate line whose mean pixel value along it is below this (0-255) — "
		"road markings are painted white/yellow, so a dim 'edge' is more likely shadow/clutter "
		"than a real marking.",
	)
	parser.add_argument(
		"--min-contrast",
		type=int,
		default=45,
		help="Reject a candidate line that isn't at least this much brighter (0-255) than the "
		"surface a few pixels to either side of it. A painted marking is a bright stripe on "
		"darker asphalt; a uniformly bright surface (sidewalk, concrete plaza, rooftop) has "
		"~zero perpendicular contrast even though it passes --min-brightness, so this is what "
		"actually distinguishes 'marking' from 'just a bright area'.",
	)
	parser.add_argument(
		"--contrast-offset",
		type=int,
		default=6,
		help="Perpendicular offset (px) each side of a line to sample as its 'background'.",
	)
	args = parser.parse_args()

	image = cv2.imread(str(args.frame))
	if image is None:
		print(f"Could not read image: {args.frame}", file=sys.stderr)
		return 1
	height, width = image.shape[:2]
	print(f"{args.frame}: {width}x{height}")

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, args.canny_low, args.canny_high)
	raw_lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=np.pi / 180,
		threshold=args.hough_threshold,
		minLineLength=args.min_line_length,
		maxLineGap=args.max_line_gap,
	)
	print(f"{0 if raw_lines is None else len(raw_lines)} raw Hough segments")

	segments = _marking_segments(
		gray,
		raw_lines,
		min_brightness=args.min_brightness,
		min_contrast=args.min_contrast,
		contrast_offset=args.contrast_offset,
	)
	print(f"{len(segments)} segments look like painted markings (bright + locally contrasty)")

	proposal = {
		"note": "PROPOSAL ONLY — image-side candidates from automatic line detection "
		"(scripts/propose_calibration.py); world coordinates are not filled in and every "
		"point needs human confirmation before use. See src/tratrac/application/"
		"scripts/propose_calibration.py's module docstring.",
		"source_frame": str(args.frame),
		"correspondences": [
			{
				"reference_frame": 0,
				"label": f"candidate_{i}",
				"image": [round(x, 1), round(y, 1)],
				"world": None,
			}
			for i, (x, y) in enumerate(_endpoints(segments))
		],
	}
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(proposal, indent=2))
	print(f"Wrote {len(proposal['correspondences'])} candidate points -> {args.out}")

	if args.overlay is not None:
		overlay = image.copy()
		for x1, y1, x2, y2 in segments:
			cv2.line(overlay, (x1, y1), (x2, y2), _LINE_COLOR, 2)
		args.overlay.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(args.overlay), overlay)
		print(f"Wrote overlay -> {args.overlay}")
	return 0


def _marking_segments(
	gray: np.ndarray,
	raw_lines: np.ndarray | None,
	*,
	min_brightness: int,
	min_contrast: int,
	contrast_offset: int,
) -> list[tuple[int, int, int, int]]:
	"""Keep only Hough segments that look like a painted road marking: bright, **and**
	brighter than the surface a few pixels either side of it. Brightness alone also passes
	uniformly light surfaces (sidewalks, plazas, rooftops) that have no real edge to mark —
	the perpendicular-contrast check is what actually rejects those.
	"""
	if raw_lines is None:
		return []
	kept: list[tuple[int, int, int, int]] = []
	for line in raw_lines:
		x1, y1, x2, y2 = (int(v) for v in line[0])
		brightness = _mean_brightness_along(gray, x1, y1, x2, y2, offset=0)
		if brightness < min_brightness:
			continue
		left = _mean_brightness_along(gray, x1, y1, x2, y2, offset=contrast_offset)
		right = _mean_brightness_along(gray, x1, y1, x2, y2, offset=-contrast_offset)
		if brightness - max(left, right) >= min_contrast:
			kept.append((x1, y1, x2, y2))
	return kept


def _mean_brightness_along(
	gray: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, offset: int
) -> float:
	"""Mean pixel value sampled along the segment, shifted ``offset`` px perpendicular to it
	(``offset=0`` samples the line itself)."""
	n = max(abs(x2 - x1), abs(y2 - y1), 1)
	xs = np.linspace(x1, x2, n)
	ys = np.linspace(y1, y2, n)
	if offset:
		length = max(math.hypot(x2 - x1, y2 - y1), 1e-6)
		nx, ny = -(y2 - y1) / length, (x2 - x1) / length  # unit normal
		xs = xs + nx * offset
		ys = ys + ny * offset
	xs = xs.round().astype(int).clip(0, gray.shape[1] - 1)
	ys = ys.round().astype(int).clip(0, gray.shape[0] - 1)
	return float(gray[ys, xs].mean())


def _endpoints(segments: list[tuple[int, int, int, int]]) -> list[tuple[float, float]]:
	points: list[tuple[float, float]] = []
	for x1, y1, x2, y2 in segments:
		points.append((float(x1), float(y1)))
		points.append((float(x2), float(y2)))
	return points


if __name__ == "__main__":
	sys.exit(main())
