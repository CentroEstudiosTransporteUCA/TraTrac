#!/usr/bin/env python3
"""Render an SSAM .trj file's trajectories overlaid on the source video.

Standalone — only stdlib + cv2 + numpy. Reads the binary .trj directly (see
vault/04_ssam_format.md for the spec), maps each TIMESTEP to the corresponding
frame index, and draws front/rear bumpers, orientation lines, vehicle IDs,
and per-track trajectory trails coloured deterministically per track.

By default each trail accumulates the vehicle's whole path and is drawn until
the vehicle disappears (drops the moment the track is no longer detected). Pass
--trail N to cap it to a rolling window of N frames instead.

Usage:
	uv run python scripts/render_trajectories.py VIDEO TRJ [--out OUT]
		[--start SEC] [--end SEC] [--trail FRAMES]

Performance note: 1920x1080 @ 30fps is roughly 50-150 frames/s on CPU with
the default codec. A 15-minute video usually renders in 3-10 min.
"""

from __future__ import annotations

import argparse
import colorsys
import struct
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2


def _color_for(vehicle_id: int) -> tuple[int, int, int]:
	"""Deterministic BGR color from vehicle id (golden-ratio hue spread)."""
	hue = (vehicle_id * 0.618033988749895) % 1.0
	r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
	return (int(b * 255), int(g * 255), int(r * 255))


def parse_trj(
	path: Path,
) -> tuple[int, float, list[tuple[float, list[dict[str, Any]]]]]:
	"""Parse a .trj file. Returns (image_height, scale, timesteps)."""
	data = path.read_bytes()
	pos = 0

	if data[pos] != 0:
		raise ValueError(f"Expected FORMAT record at offset 0, got {data[pos]}.")
	pos += 1
	endian_byte = data[pos]
	pos += 1
	end = "<" if endian_byte == ord("L") else ">"
	(version,) = struct.unpack_from(f"{end}f", data, pos)
	pos += 4
	z_option = 0
	if version >= 2.999:
		z_option = data[pos]
		pos += 1

	if data[pos] != 1:
		raise ValueError(f"Expected DIMENSIONS at offset {pos}, got {data[pos]}.")
	pos += 1
	pos += 1  # units byte (not needed for rendering)
	(scale, _min_x, _min_y, _max_x, max_y) = struct.unpack_from(f"{end}fiiii", data, pos)
	pos += 20
	image_height = max_y

	vehicle_fmt = f"{end}iiBffffffff"
	vehicle_body = struct.calcsize(vehicle_fmt)
	z_body = struct.calcsize(f"{end}ff")

	timesteps: list[tuple[float, list[dict[str, Any]]]] = []
	current_ts: float | None = None
	current_vs: list[dict[str, Any]] = []

	while pos < len(data):
		rt = data[pos]
		pos += 1
		if rt == 2:
			if current_ts is not None:
				timesteps.append((current_ts, current_vs))
			(current_ts,) = struct.unpack_from(f"{end}f", data, pos)
			pos += 4
			current_vs = []
		elif rt == 3:
			(vid, _link, _lane, fx, fy, rx, ry, length, width, speed, accel) = struct.unpack_from(
				vehicle_fmt, data, pos
			)
			pos += vehicle_body
			if z_option:
				pos += z_body
			current_vs.append(
				{
					"vid": vid,
					"front": (fx, fy),
					"rear": (rx, ry),
					"length": length,
					"width": width,
					"speed": speed,
					"accel": accel,
				}
			)
		else:
			break
	if current_ts is not None:
		timesteps.append((current_ts, current_vs))

	return image_height, scale, timesteps


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("video", type=Path, help="Source video file.")
	parser.add_argument("trj", type=Path, help="SSAM .trj file produced by tratrac.")
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output .mp4 (default: <video>_with_trajectories.mp4 next to video).",
	)
	parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
	parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
	parser.add_argument(
		"--trail",
		type=int,
		default=0,
		help="Rolling trail length in frames. 0 (default) draws the whole trajectory "
		"until the vehicle disappears.",
	)
	args = parser.parse_args()

	if not args.video.exists():
		print(f"Video not found: {args.video}", file=sys.stderr)
		return 1
	if not args.trj.exists():
		print(f"TRJ not found: {args.trj}", file=sys.stderr)
		return 1

	trj_height, scale, timesteps = parse_trj(args.trj)
	print(f"Parsed {len(timesteps)} timesteps from {args.trj.name} (scale={scale}).")

	cap = cv2.VideoCapture(str(args.video))
	if not cap.isOpened():
		print(f"Cannot open video: {args.video}", file=sys.stderr)
		return 1
	fps = cap.get(cv2.CAP_PROP_FPS)
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f"Video: {width}x{height} @ {fps:.2f} fps, {total} frames.")

	if height != trj_height:
		print(
			f"WARNING: trj DIMENSIONS height={trj_height} != video height={height}. "
			f"Y-flip will be off.",
			file=sys.stderr,
		)

	# Build frame_index -> [vehicle dicts] lookup.
	by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
	for ts, vehicles in timesteps:
		if not vehicles:
			continue
		frame_idx = round(ts * fps)
		for v in vehicles:
			by_frame[frame_idx].append(v)

	start_frame = max(0, int(args.start * fps))
	end_frame = total if args.end is None else min(total, int(args.end * fps))
	print(f"Rendering frames [{start_frame}, {end_frame}).")

	out_path = args.out or args.video.parent / f"{args.video.stem}_with_trajectories.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

	# trail=0 -> unbounded: the deque accumulates the vehicle's full path so far.
	# A positive value caps it to a rolling window of that many frames.
	trail_maxlen = args.trail if args.trail > 0 else None
	trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_maxlen))

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	rendered = 0

	try:
		for frame_idx in range(start_frame, end_frame):
			ok, frame = cap.read()
			if not ok:
				break

			current_vids: set[int] = set()
			for v in by_frame.get(frame_idx, []):
				vid = int(v["vid"])
				current_vids.add(vid)
				color = _color_for(vid)
				fx, fy = v["front"]
				rx, ry = v["rear"]
				# The exporter writes file coords as (centroid_world / scale), so
				# the stored values are pixel grid units regardless of `scale`.
				# `trj_height` is DIMENSIONS.MaxY in the same grid units (pixels).
				# Y-flip back into image-space (y grows down) is a straight
				# pixel-domain subtraction — no scale multiplication.
				fx_px = round(fx)
				fy_px = round(trj_height - fy)
				rx_px = round(rx)
				ry_px = round(trj_height - ry)

				cv2.line(frame, (rx_px, ry_px), (fx_px, fy_px), color, 2)
				cv2.circle(frame, (fx_px, fy_px), 5, color, -1)
				cv2.circle(frame, (rx_px, ry_px), 5, color, 2)
				cv2.putText(
					frame,
					f"v{vid}  {v['speed']:.0f}",
					(fx_px + 8, max(fy_px - 8, 14)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					color,
					1,
					cv2.LINE_AA,
				)

				cx = (fx_px + rx_px) // 2
				cy = (fy_px + ry_px) // 2
				trails[vid].append((cx, cy))

			# Draw trails only for currently-visible vehicles so dead tracks don't ghost.
			for vid in current_vids:
				points = trails[vid]
				if len(points) < 2:
					continue
				color = _color_for(vid)
				for i in range(1, len(points)):
					cv2.line(frame, points[i - 1], points[i], color, 1)

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
