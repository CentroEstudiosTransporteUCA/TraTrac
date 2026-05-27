#!/usr/bin/env python3
"""Semantically validate an SSAM .trj file as an end-to-end check on the
video -> trajectory pipeline.

Standalone — only the standard library, so it works even when the tratrac
package is broken (same constraint as scripts/dump_trj.py). The byte-level
format is specified in vault/04_ssam_format.md.

Unlike dump_trj.py (which prints records) this script asks whether the
trajectories make *physical sense* and reports a compliance table. Three
families of checks, each measured as "compliant instances / total instances":

  1. Continuity — a track must not appear or disappear in the frame interior.
     Every contiguous run of a track has a start and an end; each is compliant
     only if it happens at the global first/last timestep (already in view /
     leaves at the end of the clip) or near an image boundary. Internal gaps
     (a track that vanishes mid-clip and reappears) therefore show up as an
     interior disappearance + interior appearance and are penalised.

  2. Orientation smoothness — the front/rear axis (heading) must not jump
     between consecutive observations of a track. Compliant if the wrapped
     angular change is below the threshold.

  3. Kinematic plausibility — the stored Speed and Acceleration are in
     DIMENSIONS.Units (m/s, m/s^2 or ft/s, ft/s^2) per the SSAM spec, NOT in
     pixels and NOT scaled by Scale. Each is checked against a real-world
     physical ceiling (configurable; unit-aware defaults). Speed must also be
     finite and non-negative. Caveat: MVP1 writes pixel-displacement into these
     fields while declaring metric units (vault/04_ssam_format.md), so MVP1
     output is expected to fail these bounds wholesale — that is the intended
     signal, not a bug in the validator.

Position/heading checks (continuity, orientation) compare in the file's grid
units (pixels) against the DIMENSIONS bounds. Continuity's "near a boundary"
test also needs the vehicle's body size, which SSAM stores in DIMENSIONS.Units
(metres), not pixels — so it converts Length/Width to pixels via Scale. The
result is physically scale-independent (a car spans the same pixels whether
Scale is 1.0 for MVP1 pixels-as-metres or a real GSD for MVP1.75+).

Usage:
	uv run python scripts/validate_trj.py PATH [options]
	uv run python scripts/validate_trj.py PATH --fail-under 90        # CI gate
	uv run python scripts/validate_trj.py PATH --violations-csv bad.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_FORMAT = 0
_DIMENSIONS = 1
_TIMESTEP = 2
_VEHICLE = 3

# --- Default thresholds -----------------------------------------------------
_DEFAULT_BOUNDARY_MARGIN = 33  # slack added to half the vehicle's major axis
_DEFAULT_MAX_HEADING_STEP_DEG = 20.0  # heading change above this = a sudden switch

# Plausibility ceilings for a road vehicle, in DIMENSIONS.Units. The
# Speed/Acceleration fields are physical (m/s, m/s^2 or ft/s, ft/s^2), so the
# right bound depends on the file's unit byte — resolved at runtime by
# _default_kinematic_bounds. Override per dataset with --max-speed / --max-accel.
# NOTE: the metric ceilings were tightened for urban-intersection footage and no
# longer match the English ones (which keep the original "reject the impossible"
# limits): metric 22 m/s (79 km/h) / 15 m/s^2 (~1.5 g) vs English ~70 m/s
# (252 km/h) / ~12 m/s^2 (~1.2 g). Raise --max-speed for fast-road clips.
_MAX_SPEED_METRIC = 22.0  # m/s   (79 km/h)
_MAX_ACCEL_METRIC = 15.0  # m/s^2 (~1.5 g)
_MAX_SPEED_ENGLISH = 230.0  # ft/s   (~70 m/s, 252 km/h)
_MAX_ACCEL_ENGLISH = 40.0  # ft/s^2 (~12 m/s^2, ~1.2 g)


@dataclass(frozen=True, slots=True)
class VehicleRecord:
	"""One VEHICLE record, in the file's grid units (SSAM y-up, scaled-out)."""

	vehicle_id: int
	front_x: float
	front_y: float
	rear_x: float
	rear_y: float
	length: float
	width: float
	speed: float
	accel: float

	@property
	def centroid(self) -> tuple[float, float]:
		"""Grid-unit centroid: the exporter writes front/rear as centroid +/-
		heading*(length/2), so the midpoint recovers the centroid exactly."""
		return ((self.front_x + self.rear_x) / 2.0, (self.front_y + self.rear_y) / 2.0)

	@property
	def heading_angle(self) -> float | None:
		"""Heading as atan2(front - rear), radians. None if front == rear."""
		dx = self.front_x - self.rear_x
		dy = self.front_y - self.rear_y
		if dx == 0.0 and dy == 0.0:
			return None
		return math.atan2(dy, dx)

	@property
	def major_axis(self) -> float:
		return max(self.length, self.width)


@dataclass(frozen=True, slots=True)
class TrjData:
	version: float
	endian: str  # "L" or "B"
	units: int
	scale: float
	bounds: tuple[int, int, int, int]  # min_x, min_y, max_x, max_y
	timesteps: list[tuple[float, list[VehicleRecord]]]

	@property
	def total_records(self) -> int:
		return sum(len(vs) for _, vs in self.timesteps)


@dataclass(frozen=True, slots=True)
class CheckResult:
	name: str
	compliant: int
	total: int

	@property
	def pct(self) -> float | None:
		if self.total == 0:
			return None
		return 100.0 * self.compliant / self.total


@dataclass(frozen=True, slots=True)
class Violation:
	"""One non-compliant instance, located so it can be found in the source video."""

	frame_index: int  # timestep ordinal == frame index (pipeline emits one per frame)
	timestamp_s: float
	vehicle_id: int
	check: str  # which check failed, e.g. "appearance", "heading_switch"
	detail: str  # human-readable reason
	record: VehicleRecord


# --- Parsing ----------------------------------------------------------------


def parse_trj(path: Path) -> TrjData:
	"""Parse a binary SSAM .trj v1.04 / v3.0 file. Mirrors scripts/dump_trj.py."""
	data = path.read_bytes()
	if not data:
		raise ValueError(f"{path}: file is empty.")
	pos = 0

	if data[pos] != _FORMAT:
		raise ValueError(f"Expected FORMAT record (0) at offset 0, got {data[pos]}.")
	pos += 1
	endian_byte = data[pos]
	pos += 1
	if endian_byte == ord("L"):
		end = "<"
	elif endian_byte == ord("B"):
		end = ">"
	else:
		raise ValueError(f"Unknown endian byte: 0x{endian_byte:02x}.")
	(version,) = struct.unpack_from(f"{end}f", data, pos)
	pos += 4
	z_option = 0
	if version >= 2.999:
		z_option = data[pos]
		pos += 1

	if data[pos] != _DIMENSIONS:
		raise ValueError(f"Expected DIMENSIONS (1) at offset {pos}, got {data[pos]}.")
	pos += 1
	units = data[pos]
	pos += 1
	(scale, min_x, min_y, max_x, max_y) = struct.unpack_from(f"{end}fiiii", data, pos)
	pos += 20

	vehicle_fmt = f"{end}iiBffffffff"
	vehicle_body = struct.calcsize(vehicle_fmt)
	z_body = struct.calcsize(f"{end}ff")

	timesteps: list[tuple[float, list[VehicleRecord]]] = []
	current_t: float | None = None
	current_vs: list[VehicleRecord] = []

	while pos < len(data):
		rt = data[pos]
		pos += 1
		if rt == _TIMESTEP:
			if pos + 4 > len(data):
				raise ValueError(f"Truncated TIMESTEP at offset {pos - 1}.")
			if current_t is not None:
				timesteps.append((current_t, current_vs))
			(current_t,) = struct.unpack_from(f"{end}f", data, pos)
			pos += 4
			current_vs = []
		elif rt == _VEHICLE:
			needed = vehicle_body + (z_body if z_option else 0)
			if pos + needed > len(data):
				raise ValueError(f"Truncated VEHICLE at offset {pos - 1}.")
			(vid, _link, _lane, fx, fy, rx, ry, length, width, speed, accel) = struct.unpack_from(
				vehicle_fmt, data, pos
			)
			pos += vehicle_body
			if z_option:
				pos += z_body
			current_vs.append(VehicleRecord(vid, fx, fy, rx, ry, length, width, speed, accel))
		else:
			raise ValueError(f"Unknown record type {rt} (0x{rt:02x}) at offset {pos - 1}.")

	if current_t is not None:
		timesteps.append((current_t, current_vs))

	return TrjData(
		version=version,
		endian=chr(endian_byte),
		units=units,
		scale=scale,
		bounds=(min_x, min_y, max_x, max_y),
		timesteps=timesteps,
	)


# --- Geometry helpers -------------------------------------------------------


def _edge_distance(record: VehicleRecord, bounds: tuple[int, int, int, int]) -> float:
	"""Distance from the vehicle centroid to the nearest image edge, grid units."""
	min_x, min_y, max_x, max_y = bounds
	cx, cy = record.centroid
	return min(cx - min_x, max_x - cx, cy - min_y, max_y - cy)


def _near_boundary(
	record: VehicleRecord, bounds: tuple[int, int, int, int], margin: float, scale: float
) -> bool:
	"""True if the vehicle's body can reach an image edge: centroid distance to
	the nearest edge is within `margin` plus half the vehicle's major axis.

	`major_axis` (Length/Width) is stored in DIMENSIONS.Units — metres for a
	metric file — while positions are grid units (pixels = world / Scale). So it
	is converted to pixels via `/ scale` before joining the pixel margin. No-op
	when Scale is 1.0 (MVP1 pixels-as-metres); only bites once real GSD
	calibration lands (MVP1.75+)."""
	return _edge_distance(record, bounds) <= margin + 0.5 * record.major_axis / scale


def _angle_delta_deg(a: float, b: float) -> float:
	"""Smallest unsigned angular difference between two angles (radians), in
	degrees, wrapped to [0, 180]."""
	diff = math.degrees(abs(a - b)) % 360.0
	return min(diff, 360.0 - diff)


# --- Checks -----------------------------------------------------------------


def _segments(ordinals: list[int]) -> list[tuple[int, int]]:
	"""Split a sorted list of present-frame ordinals into contiguous runs,
	returned as (start_index, end_index) pairs into the input list."""
	runs: list[tuple[int, int]] = []
	start = 0
	for i in range(1, len(ordinals)):
		if ordinals[i] != ordinals[i - 1] + 1:
			runs.append((start, i - 1))
			start = i
	runs.append((start, len(ordinals) - 1))
	return runs


def check_continuity(
	trj: TrjData, margin: float
) -> tuple[CheckResult, CheckResult, list[Violation]]:
	"""Appearances and disappearances. One instance per contiguous run start/end."""
	last_ordinal = len(trj.timesteps) - 1
	# track_id -> ordered (ordinal, record)
	tracks: dict[int, list[tuple[int, VehicleRecord]]] = defaultdict(list)
	for ordinal, (_t, vehicles) in enumerate(trj.timesteps):
		for record in vehicles:
			tracks[record.vehicle_id].append((ordinal, record))

	appear_ok = appear_total = 0
	disappear_ok = disappear_total = 0
	violations: list[Violation] = []
	for observations in tracks.values():
		ordinals = [o for o, _ in observations]
		for start_i, end_i in _segments(ordinals):
			start_ord, start_rec = observations[start_i]
			end_ord, end_rec = observations[end_i]
			appear_total += 1
			if start_ord == 0 or _near_boundary(start_rec, trj.bounds, margin, trj.scale):
				appear_ok += 1
			else:
				dist = _edge_distance(start_rec, trj.bounds)
				violations.append(
					Violation(
						start_ord,
						trj.timesteps[start_ord][0],
						start_rec.vehicle_id,
						"appearance",
						f"interior appearance, {dist:.0f}px from edge",
						start_rec,
					)
				)
			disappear_total += 1
			if end_ord == last_ordinal or _near_boundary(end_rec, trj.bounds, margin, trj.scale):
				disappear_ok += 1
			else:
				dist = _edge_distance(end_rec, trj.bounds)
				violations.append(
					Violation(
						end_ord,
						trj.timesteps[end_ord][0],
						end_rec.vehicle_id,
						"disappearance",
						f"interior disappearance, {dist:.0f}px from edge",
						end_rec,
					)
				)

	return (
		CheckResult("Appearances (start/boundary)", appear_ok, appear_total),
		CheckResult("Disappearances (end/boundary)", disappear_ok, disappear_total),
		violations,
	)


def check_orientation(trj: TrjData, max_step_deg: float) -> tuple[CheckResult, list[Violation]]:
	"""Heading smoothness across consecutive observations of each track."""
	tracks: dict[int, list[tuple[int, float, VehicleRecord]]] = defaultdict(list)
	for ordinal, (t, vehicles) in enumerate(trj.timesteps):
		for record in vehicles:
			tracks[record.vehicle_id].append((ordinal, t, record))

	ok = total = 0
	violations: list[Violation] = []
	for observations in tracks.values():
		for (_po, _pt, prev), (ordinal, t, curr) in pairwise(observations):
			a, b = prev.heading_angle, curr.heading_angle
			if a is None or b is None:
				continue
			total += 1
			delta = _angle_delta_deg(a, b)
			if delta <= max_step_deg:
				ok += 1
			else:
				violations.append(
					Violation(
						ordinal,
						t,
						curr.vehicle_id,
						"heading_switch",
						f"heading jumped {delta:.1f} deg",
						curr,
					)
				)
	return CheckResult("Heading smoothness", ok, total), violations


def _default_kinematic_bounds(units: int) -> tuple[float, float]:
	"""Physical (max_speed, max_accel) defaults for the file's unit system."""
	if units == 0:  # English: ft/s, ft/s^2
		return (_MAX_SPEED_ENGLISH, _MAX_ACCEL_ENGLISH)
	return (_MAX_SPEED_METRIC, _MAX_ACCEL_METRIC)  # Metric (1) or unknown


def check_plausibility(
	trj: TrjData, *, max_speed: float, max_accel: float
) -> tuple[CheckResult, CheckResult, list[Violation]]:
	"""Physical plausibility of the stored Speed/Acceleration fields. These are
	in DIMENSIONS.Units (m/s, m/s^2 or ft/s, ft/s^2) per the SSAM spec, so the
	bounds are real-world limits, not pixel heuristics. Speed must be finite and
	in [0, max_speed]; acceleration finite with |a| <= max_accel."""
	speed_unit = "ft/s" if trj.units == 0 else "m/s"
	accel_unit = "ft/s^2" if trj.units == 0 else "m/s^2"
	speed_ok = accel_ok = total = 0
	violations: list[Violation] = []
	for ordinal, (t, vehicles) in enumerate(trj.timesteps):
		for record in vehicles:
			total += 1
			if math.isfinite(record.speed) and 0.0 <= record.speed <= max_speed:
				speed_ok += 1
			else:
				if not math.isfinite(record.speed):
					detail = f"speed not finite: {record.speed}"
				elif record.speed < 0:
					detail = f"speed negative: {record.speed:.2f} {speed_unit}"
				else:
					detail = f"speed {record.speed:.2f} > {max_speed:.2f} {speed_unit}"
				violations.append(
					Violation(ordinal, t, record.vehicle_id, "speed_implausible", detail, record)
				)
			if math.isfinite(record.accel) and abs(record.accel) <= max_accel:
				accel_ok += 1
			else:
				if not math.isfinite(record.accel):
					detail = f"accel not finite: {record.accel}"
				else:
					detail = f"|accel| {abs(record.accel):.2f} > {max_accel:.2f} {accel_unit}"
				violations.append(
					Violation(ordinal, t, record.vehicle_id, "accel_implausible", detail, record)
				)
	return (
		CheckResult("Speed plausibility", speed_ok, total),
		CheckResult("Accel plausibility", accel_ok, total),
		violations,
	)


# --- Reporting --------------------------------------------------------------


def _format_row(result: CheckResult) -> str:
	pct = result.pct
	pct_str = "   n/a" if pct is None else f"{pct:6.2f}%"
	return f"  {result.name:<32s}{result.compliant:>10d} / {result.total:<10d}{pct_str}"


def print_report(trj: TrjData, groups: list[tuple[str, list[CheckResult]]]) -> None:
	min_x, min_y, max_x, max_y = trj.bounds
	units_name = {0: "english", 1: "metric"}.get(trj.units, f"unknown({trj.units})")
	tracks = {r.vehicle_id for _t, vs in trj.timesteps for r in vs}
	print(
		f"format v{trj.version:.2f}  endian {trj.endian}  units {units_name}  "
		f"scale {trj.scale}  bounds x[{min_x},{max_x}] y[{min_y},{max_y}]"
	)
	print(
		f"{len(trj.timesteps)} timesteps, {trj.total_records} vehicle records, {len(tracks)} tracks"
	)
	print()
	width = 64
	print(f"  {'CHECK':<32s}{'COMPLIANT':>10s}   {'TOTAL':<10s}{'%':>7s}")
	for title, results in groups:
		print(f"{title} " + "-" * (width - len(title) - 1))
		for result in results:
			print(_format_row(result))
	print("-" * width)


_CSV_HEADER = (
	"frame_index",
	"timestamp_s",
	"vehicle_id",
	"check",
	"detail",
	"centroid_x",
	"centroid_y",
	"front_x",
	"front_y",
	"rear_x",
	"rear_y",
	"length",
	"width",
	"speed",
	"accel",
)


def write_violations_csv(path: Path, trj: TrjData, violations: list[Violation]) -> None:
	"""Write every non-compliant instance, one row each. Positions are converted
	to IMAGE space (pixels, y grows down, matching the video) so a row points
	straight at the frame and spot to inspect. Sorted by frame then vehicle."""
	max_y = trj.bounds[3]
	rows = sorted(violations, key=lambda v: (v.frame_index, v.vehicle_id, v.check))
	with path.open("w", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(_CSV_HEADER)
		for v in rows:
			r = v.record
			cx, cy = r.centroid
			writer.writerow(
				(
					v.frame_index,
					f"{v.timestamp_s:.3f}",
					v.vehicle_id,
					v.check,
					v.detail,
					f"{cx:.1f}",
					f"{max_y - cy:.1f}",
					f"{r.front_x:.1f}",
					f"{max_y - r.front_y:.1f}",
					f"{r.rear_x:.1f}",
					f"{max_y - r.rear_y:.1f}",
					f"{r.length:.2f}",
					f"{r.width:.2f}",
					f"{r.speed:.3f}",
					f"{r.accel:.3f}",
				)
			)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument("path", type=Path, help="Path to the .trj file.")
	parser.add_argument(
		"--boundary-margin",
		type=float,
		default=_DEFAULT_BOUNDARY_MARGIN,
		help="Grid-unit slack added to half the vehicle major axis when deciding "
		"if an appearance/disappearance touches a boundary.",
	)
	parser.add_argument(
		"--max-heading-step",
		type=float,
		default=_DEFAULT_MAX_HEADING_STEP_DEG,
		help="Heading change (degrees) above which a transition between "
		"consecutive observations counts as a sudden switch.",
	)
	parser.add_argument(
		"--max-speed",
		type=float,
		default=None,
		help="Plausibility ceiling for Speed, in the file's units (m/s or ft/s). "
		"Default depends on DIMENSIONS.Units (22 m/s or 230 ft/s).",
	)
	parser.add_argument(
		"--max-accel",
		type=float,
		default=None,
		help="Plausibility ceiling for |Acceleration|, in the file's units "
		"(m/s^2 or ft/s^2). Default depends on DIMENSIONS.Units (15 or 40).",
	)
	parser.add_argument(
		"--violations-csv",
		type=Path,
		default=None,
		help="Write every non-compliant instance to this CSV (frame, timestamp, "
		"vehicle id, which check failed, why, and image-space position) so each "
		"can be located in the source video. Filter by the 'check' column.",
	)
	parser.add_argument(
		"--fail-under",
		type=float,
		default=None,
		help="Exit non-zero if any check's compliance percentage is below this "
		"value. Omit to always exit 0 (report only).",
	)
	args = parser.parse_args()

	try:
		trj = parse_trj(args.path)
	except (ValueError, OSError) as exc:
		print(f"Failed to parse {args.path}: {exc}", file=sys.stderr)
		return 1

	default_speed, default_accel = _default_kinematic_bounds(trj.units)
	max_speed = args.max_speed if args.max_speed is not None else default_speed
	max_accel = args.max_accel if args.max_accel is not None else default_accel

	appearances, disappearances, continuity_v = check_continuity(trj, args.boundary_margin)
	orientation, orientation_v = check_orientation(trj, args.max_heading_step)
	speed_plaus, accel_plaus, plausibility_v = check_plausibility(
		trj, max_speed=max_speed, max_accel=max_accel
	)

	groups = [
		("Continuity", [appearances, disappearances]),
		("Orientation", [orientation]),
		("Kinematics", [speed_plaus, accel_plaus]),
	]
	print_report(trj, groups)

	if args.violations_csv is not None:
		violations = continuity_v + orientation_v + plausibility_v
		write_violations_csv(args.violations_csv, trj, violations)
		print(f"\nWrote {len(violations)} violations to {args.violations_csv}")

	if args.fail_under is not None:
		failing = [
			r
			for _title, results in groups
			for r in results
			if r.pct is not None and r.pct < args.fail_under
		]
		if failing:
			names = ", ".join(r.name for r in failing)
			print(f"\nFAIL: below {args.fail_under:.1f}% -> {names}", file=sys.stderr)
			return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
