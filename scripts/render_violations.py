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

Usage:
	uv run python scripts/render_violations.py VIDEO --violations-csv CSV [--out OUT]
		[--start HH:MM:SS] [--end HH:MM:SS] [--checks appearance,...]
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


def parse_violations_csv(
	path: Path, fps: float, checks: set[str] | None
) -> dict[int, list[ViolationMark]]:
	"""Group a validate_trj.py violations CSV by video frame.

	Rows already carry image-space (y-down) positions, so no Y-flip is needed.
	Each row is placed on round(timestamp_s * fps). When `checks` is given, rows
	whose `check` is not in it are dropped. Multiple checks failing for one vehicle
	on one frame collapse into a single mark.
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
			key = (round(float(row["timestamp_s"]) * fps), int(row["vehicle_id"]))
			checks_by_key[key].add(check)
			centroid_by_key[key] = (
				round(float(row["centroid_x"])),
				round(float(row["centroid_y"])),
			)

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
	args = parser.parse_args()

	if not args.video.exists():
		print(f"Video not found: {args.video}", file=sys.stderr)
		return 1
	if not args.violations_csv.exists():
		print(f"Violations CSV not found: {args.violations_csv}", file=sys.stderr)
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
	violations_by_frame = parse_violations_csv(args.violations_csv, fps, checks_filter)
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
