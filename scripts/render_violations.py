#!/usr/bin/env python3
"""Render validator violations onto a video.

Standalone — only stdlib + cv2. Takes a video and a violations CSV produced by
``scripts/validate_trj.py`` and marks each non-compliant instance in red on the
frame where it occurs, aligned by ``round(timestamp_s * fps)``.

Trajectory drawing now lives in the pipeline's ``OverlayVideoExporter``
(``export.video_out``, see vault/20_video_export.md), which writes a video of each
(already-stabilized, if ego-motion is on) frame with bumpers/IDs/trails drawn.
This script's sole remaining job is the violation overlay — point it at that
overlay ``.mp4`` to get "warped frame + trajectories + violations".

Alignment: a violation row is placed on ``round(timestamp_s * fps)`` counted from
the passed video's frame 0. So the video's frame 0 must correspond to absolute
time 0 of the run (true for an untrimmed run, e.g. the overlay videos in
.configs/). For a ``--start``-trimmed run, render on the raw source video and pass
a matching ``--start`` so the seek re-aligns the absolute frame index. The CSV's
own ``frame_index`` ordinal is deliberately NOT used (it is the in-file timestep
count, which diverges under ``--timestep-precision`` decimation).

Coordinates: when the run used ego-motion stabilization, the ``.trj`` — and so the
violations CSV — carries positions in the *global* stabilization frame, not raw
pixels. Pass ``--transforms-csv`` (the run's ``export.transform_csv``) so each mark
is mapped from the global frame back onto the raw video, the same inverse-transform
the overlay video applies to its bumpers/trails. Without it, marks are drawn as-is,
which is correct only when stabilization was off (global == raw).

Usage:
	uv run python scripts/render_violations.py VIDEO --violations-csv CSV [--out OUT]
		[--transforms-csv TCSV] [--start HH:MM:SS] [--end HH:MM:SS] [--checks appearance,...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

_VIOLATION_COLOR = (0, 0, 255)  # BGR red — stands out against the overlay's track colours.


@dataclass(frozen=True, slots=True)
class ViolationMark:
	"""One vehicle's violations on a single frame, ready to draw (image-space, y-down).

	Every check that failed for this (frame, vehicle) collapses into `label`."""

	vehicle_id: int
	cx: int
	cy: int
	label: str


def _parse_timecode(value: str) -> float:
	"""Parse an HH:MM:SS timecode into seconds (mirrors the tratrac CLI).

	Accepts ``HH:MM:SS(.ms)`` and the shorter ``MM:SS(.ms)`` / ``SS(.ms)`` forms
	(e.g. ``00:01:30``, ``1:30``, ``12.5``). Raises ``argparse.ArgumentTypeError``
	on malformed input so argparse reports it cleanly.
	"""
	parts = value.strip().split(":")
	if len(parts) > 3:
		raise argparse.ArgumentTypeError(f"Timecode has too many ':'-separated parts: {value!r}.")
	try:
		seconds = float(parts[-1])
		minutes = float(parts[-2]) if len(parts) >= 2 else 0.0
		hours = float(parts[-3]) if len(parts) == 3 else 0.0
	except ValueError:
		raise argparse.ArgumentTypeError(
			f"Invalid timecode {value!r}; expected HH:MM:SS, MM:SS, or SS."
		) from None
	if seconds < 0 or minutes < 0 or hours < 0:
		raise argparse.ArgumentTypeError(f"Timecode components must be non-negative: {value!r}.")
	return hours * 3600.0 + minutes * 60.0 + seconds


_REQUIRED_VIOLATION_COLUMNS = ("timestamp_s", "vehicle_id", "check", "centroid_x", "centroid_y")

# 2x3 affine coefficients (a, b, c, d, tx, ty): maps a raw frame's pixels into the
# global stabilization frame (the space the .trj/violations positions live in when
# ego-motion is on). Its inverse maps those positions back onto the raw frame.
Transform = tuple[float, float, float, float, float, float]
_REQUIRED_TRANSFORM_COLUMNS = ("frame", "a", "b", "c", "d", "tx", "ty")


def parse_transforms_csv(path: Path) -> dict[int, Transform]:
	"""Load a tratrac per-frame transform CSV (export.transform_csv) by frame index.

	Each row is the current-frame -> global ego-motion transform for that frame. The
	`frame` column is the absolute Frame.index, which matches round(timestamp_s * fps)
	for the violation rows, so the two align on the same integer key.
	"""
	transforms: dict[int, Transform] = {}
	with path.open(newline="") as fh:
		reader = csv.DictReader(fh)
		missing = [c for c in _REQUIRED_TRANSFORM_COLUMNS if c not in (reader.fieldnames or [])]
		if missing:
			raise ValueError(f"{path}: missing columns {missing}; not a tratrac transform CSV?")
		for row in reader:
			transforms[int(row["frame"])] = (
				float(row["a"]),
				float(row["b"]),
				float(row["c"]),
				float(row["d"]),
				float(row["tx"]),
				float(row["ty"]),
			)
	return transforms


def global_to_raw(transform: Transform, x: float, y: float) -> tuple[float, float]:
	"""Map a global-frame point back onto the raw frame by inverting `transform`.

	Mirrors domain.geometry.Transform2D.inverse().apply(); reimplemented here in
	plain arithmetic so this script stays standalone (stdlib + cv2 only). The 2x2
	linear part of a 4-DOF similarity is always non-singular (det = squared scale).
	"""
	a, b, c, d, tx, ty = transform
	det = a * d - b * c
	if det == 0.0:
		raise ValueError("Cannot invert a transform with a singular linear part.")
	ia, ib = d / det, -b / det
	ic, id_ = -c / det, a / det
	itx = -(ia * tx + ib * ty)
	ity = -(ic * tx + id_ * ty)
	return (ia * x + ib * y + itx, ic * x + id_ * y + ity)


def parse_violations_csv(
	path: Path, fps: float, checks: set[str] | None, transforms: dict[int, Transform] | None = None
) -> dict[int, list[ViolationMark]]:
	"""Group a validate_trj.py violations CSV by video frame.

	Each row is placed on round(timestamp_s * fps). When `checks` is given, rows
	whose `check` is not in it are dropped. Multiple checks failing for one vehicle
	on one frame collapse into a single mark.

	Positions are image-space (y-down), so no Y-flip is needed. When `transforms` is
	given (a run with ego-motion on), each position is in the global stabilization
	frame and is mapped back onto the raw frame via that frame's inverse transform
	before rounding; without it the positions are drawn as-is (correct when
	stabilization was off, where global == raw).
	"""
	checks_by_key: dict[tuple[int, int], set[str]] = defaultdict(set)
	centroid_by_key: dict[tuple[int, int], tuple[int, int]] = {}
	with path.open(newline="") as fh:
		reader = csv.DictReader(fh)
		missing = [c for c in _REQUIRED_VIOLATION_COLUMNS if c not in (reader.fieldnames or [])]
		if missing:
			raise ValueError(f"{path}: missing columns {missing}; not a validate_trj.py CSV?")
		for row in reader:
			check = row["check"]
			if checks is not None and check not in checks:
				continue
			frame_idx = round(float(row["timestamp_s"]) * fps)
			key = (frame_idx, int(row["vehicle_id"]))
			cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
			if transforms is not None:
				transform = transforms.get(frame_idx)
				if transform is not None:
					cx, cy = global_to_raw(transform, cx, cy)
			checks_by_key[key].add(check)
			centroid_by_key[key] = (round(cx), round(cy))

	by_frame: dict[int, list[ViolationMark]] = defaultdict(list)
	for (frame_idx, vid), names in checks_by_key.items():
		cx, cy = centroid_by_key[(frame_idx, vid)]
		by_frame[frame_idx].append(ViolationMark(vid, cx, cy, " / ".join(sorted(names))))
	return by_frame


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"video", type=Path, help="Video to mark; typically the pipeline's overlay .mp4."
	)
	parser.add_argument(
		"--violations-csv",
		type=Path,
		required=True,
		help="Violations CSV produced by scripts/validate_trj.py.",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output .mp4 (default: <video>_with_violations.mp4 next to video).",
	)
	parser.add_argument(
		"--start",
		type=_parse_timecode,
		default=0.0,
		help="Start time as a timecode (HH:MM:SS, MM:SS, or SS). Default: video start.",
	)
	parser.add_argument(
		"--end",
		type=_parse_timecode,
		default=None,
		help="End time as a timecode (HH:MM:SS, MM:SS, or SS). Default: video end.",
	)
	parser.add_argument(
		"--checks",
		type=str,
		default=None,
		help="Comma-separated subset of check names to overlay (e.g. "
		"appearance,disappearance,heading_switch). Default: every check in the CSV.",
	)
	parser.add_argument(
		"--transforms-csv",
		type=Path,
		default=None,
		help="Per-frame ego-motion transform CSV (tratrac export.transform_csv). Required to "
		"place marks correctly when the run used ego-motion stabilization; the positions are "
		"then mapped from the global frame back onto the raw video. Omit for non-stabilized runs.",
	)
	args = parser.parse_args()

	if not args.video.exists():
		print(f"Video not found: {args.video}", file=sys.stderr)
		return 1
	if not args.violations_csv.exists():
		print(f"Violations CSV not found: {args.violations_csv}", file=sys.stderr)
		return 1
	if args.transforms_csv is not None and not args.transforms_csv.exists():
		print(f"Transforms CSV not found: {args.transforms_csv}", file=sys.stderr)
		return 1

	cap = cv2.VideoCapture(str(args.video))
	if not cap.isOpened():
		print(f"Cannot open video: {args.video}", file=sys.stderr)
		return 1
	fps = cap.get(cv2.CAP_PROP_FPS)
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f"Video: {width}x{height} @ {fps:.2f} fps, {total} frames.")

	checks_filter = (
		{c.strip() for c in args.checks.split(",") if c.strip()} if args.checks else None
	)
	transforms = parse_transforms_csv(args.transforms_csv) if args.transforms_csv else None
	if transforms is not None:
		print(f"Loaded {len(transforms)} per-frame transforms; mapping marks back onto raw frames.")
	violations_by_frame = parse_violations_csv(args.violations_csv, fps, checks_filter, transforms)
	mark_count = sum(len(v) for v in violations_by_frame.values())
	print(f"Loaded {mark_count} violation marks from {args.violations_csv.name}.")

	start_frame = max(0, int(args.start * fps))
	end_frame = total if args.end is None else min(total, int(args.end * fps))
	print(f"Rendering frames [{start_frame}, {end_frame}).")

	out_path = args.out or args.video.parent / f"{args.video.stem}_with_violations.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	rendered = 0

	try:
		for frame_idx in range(start_frame, end_frame):
			ok, frame = cap.read()
			if not ok:
				break

			for vm in violations_by_frame.get(frame_idx, []):
				cv2.circle(frame, (vm.cx, vm.cy), 16, _VIOLATION_COLOR, 2)
				cv2.putText(
					frame,
					vm.label,
					(vm.cx + 18, vm.cy + 4),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					_VIOLATION_COLOR,
					1,
					cv2.LINE_AA,
				)

			writer.write(frame)
			rendered += 1
			if rendered % 1000 == 0:
				print(f"  rendered {rendered}/{end_frame - start_frame} frames")
	finally:
		cap.release()
		writer.release()

	print(f"\nWrote {rendered} frames -> {out_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
