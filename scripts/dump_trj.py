#!/usr/bin/env python3
"""Dump an SSAM .trj v1.04 / v3.0 binary trajectory file to human-readable text.

Standalone — uses only the standard library, so it works even when the tratrac
package is broken. Spec lives in vault/04_ssam_format.md.

Usage:
	uv run python scripts/dump_trj.py PATH [--max-frames N] [--summary]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_FORMAT = 0
_DIMENSIONS = 1
_TIMESTEP = 2
_VEHICLE = 3


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("path", type=Path, help="Path to the .trj file.")
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Only print the first N timesteps (summary still totals all).",
	)
	parser.add_argument(
		"--summary",
		action="store_true",
		help="Skip per-record output; print only the summary block.",
	)
	args = parser.parse_args()

	data = args.path.read_bytes()
	if not data:
		print(f"{args.path}: file is empty.", file=sys.stderr)
		return 1

	pos = 0

	# --- FORMAT record ---
	if data[pos] != _FORMAT:
		print(f"Expected FORMAT record (0) at offset 0, got {data[pos]}.", file=sys.stderr)
		return 1
	pos += 1
	endian_byte = data[pos]
	pos += 1
	if endian_byte == ord("L"):
		end = "<"
		endian_name = "little"
	elif endian_byte == ord("B"):
		end = ">"
		endian_name = "big"
	else:
		print(f"Unknown endian byte: 0x{endian_byte:02x}.", file=sys.stderr)
		return 1
	(version,) = struct.unpack_from(f"{end}f", data, pos)
	pos += 4

	z_option = 0
	if version >= 2.999:
		z_option = data[pos]
		pos += 1

	print("== FORMAT ==")
	print(f"  endian:   {chr(endian_byte)} ({endian_name})")
	print(f"  version:  {version:.2f}")
	if z_option:
		print(f"  z_option: {z_option} (Front Z + Rear Z appended per vehicle)")
	else:
		print("  z_option: 0 (no elevation)")
	print()

	# --- DIMENSIONS record ---
	if data[pos] != _DIMENSIONS:
		print(f"Expected DIMENSIONS record (1) at offset {pos}, got {data[pos]}.", file=sys.stderr)
		return 1
	pos += 1
	units = data[pos]
	pos += 1
	(scale, min_x, min_y, max_x, max_y) = struct.unpack_from(f"{end}fiiii", data, pos)
	pos += 20

	units_name = {0: "english (ft, ft/s, ft/s²)", 1: "metric (m, m/s, m/s²)"}.get(
		units, f"unknown ({units})"
	)
	print("== DIMENSIONS ==")
	print(f"  units:  {units_name}")
	print(f"  scale:  {scale}")
	print(f"  bounds: x=[{min_x}, {max_x}] y=[{min_y}, {max_y}]")
	print()

	# --- Walk records ---
	vehicle_fmt = f"{end}iiBffffffff"
	vehicle_body = struct.calcsize(vehicle_fmt)  # = 41 (excludes record-type byte)
	z_fmt = f"{end}ff"
	z_body = struct.calcsize(z_fmt)  # = 8

	timestep_count = 0
	vehicle_count = 0
	track_ids: set[int] = set()
	last_time = 0.0
	verbose = not args.summary

	while pos < len(data):
		rt = data[pos]
		pos += 1
		if rt == _TIMESTEP:
			if pos + 4 > len(data):
				print(f"Truncated TIMESTEP at offset {pos - 1}.", file=sys.stderr)
				break
			(timestep_s,) = struct.unpack_from(f"{end}f", data, pos)
			pos += 4
			last_time = timestep_s
			if verbose and (args.max_frames is None or timestep_count < args.max_frames):
				print(f"== TIMESTEP {timestep_count}  t={timestep_s:.3f}s ==")
			timestep_count += 1
		elif rt == _VEHICLE:
			needed = vehicle_body + (z_body if z_option else 0)
			if pos + needed > len(data):
				print(f"Truncated VEHICLE at offset {pos - 1}.", file=sys.stderr)
				break
			(vid, link_id, lane_id, fx, fy, rx, ry, length, width, speed, accel) = (
				struct.unpack_from(vehicle_fmt, data, pos)
			)
			pos += vehicle_body
			fz: float | None = None
			rz: float | None = None
			if z_option:
				(fz, rz) = struct.unpack_from(z_fmt, data, pos)
				pos += z_body
			vehicle_count += 1
			track_ids.add(vid)
			if verbose and (args.max_frames is None or timestep_count <= args.max_frames):
				z_str = f"  z=({fz:.2f},{rz:.2f})" if fz is not None and rz is not None else ""
				print(
					f"  v{vid:<5d} link={link_id:<3d} lane={lane_id:<3d}  "
					f"front=({fx:8.2f},{fy:8.2f})  rear=({rx:8.2f},{ry:8.2f}){z_str}  "
					f"L={length:6.2f} W={width:6.2f}  v={speed:7.2f} a={accel:7.2f}"
				)
		else:
			print(
				f"Unknown record type byte {rt} (0x{rt:02x}) at offset {pos - 1}. Stopping.",
				file=sys.stderr,
			)
			break

	print()
	print("== Summary ==")
	print(f"  timesteps:     {timestep_count}")
	print(f"  vehicle recs:  {vehicle_count}")
	print(f"  unique IDs:    {len(track_ids)}")
	print(f"  last t:        {last_time:.3f}s")
	print(f"  file size:     {len(data)} bytes")
	if pos != len(data):
		print(f"  unread bytes:  {len(data) - pos} (after early stop)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
