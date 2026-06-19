"""Plot the per-run diagnostics from a TraTrac outputs folder.

Standalone (stdlib + numpy + matplotlib; no package imports, so it works even if the
package is broken). Walks an outputs directory, finds every baseline ``.trj`` (and its
``_smooth.trj`` / ``_tracks.csv`` siblings, if present), and writes two figures next to
each:

  <stem>_smoothing.png  — the de-jitter story (cat. 1): speed(t), acceleration(t) with a
                          plausibility band, a jerk histogram, and trajectory paths,
                          baseline vs RTS-smoothed.
  <stem>_tracking.png   — tracking / continuity (cat. 2): track-lifespan Gantt,
                          track-length histogram, active tracks over time, track
                          birth/death positions, and a detection-confidence histogram
                          (when the tracks sidecar is present).

The `.trj` is parsed inline (SSAM v1.04): positions are in image pixels; speed/accel are
metric (m/s, m/s^2) when the run had a real GSD scale.

Pass ``--video FILE`` (or a folder of clips, matched to each run by name prefix) to draw
a frame of the filmed scene behind the trajectory and birth/death panels.

Usage:
    uv run python scripts/plot_run.py OUTPUTS_DIR [--out DIR] [--accel-bound 8.0]
        [--video CLIP_OR_FOLDER]
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
	import matplotlib

	matplotlib.use("Agg")  # headless: write PNGs, never open a window
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment guard
	sys.exit("matplotlib is required: `uv add matplotlib` (or it ships transitively).")

# SSAM v1.04 record structs (see infrastructure/export/ssam_trj.py).
_FORMAT = struct.Struct("<BBf")  # type, endian, version
_DIMENSIONS = struct.Struct("<BBfiiii")  # type, units, scale, min_x, min_y, max_x, max_y
_TIMESTEP = struct.Struct("<Bf")  # type, timestamp
_VEHICLE = struct.Struct(
	"<BiiBffffffff"
)  # type, id, link, lane, fx, fy, rx, ry, len, wid, spd, acc
_TIMESTEP_TYPE = 2
_VEHICLE_TYPE = 3


@dataclass
class Trj:
	"""One parsed .trj: image bounds + per-vehicle (t, x, y, speed, accel) arrays."""

	bounds: tuple[int, int, int, int]  # min_x, min_y, max_x, max_y (pixels)
	by_vehicle: dict[int, np.ndarray]  # vehicle_id -> (n, 5) columns [t, x, y, speed, accel]
	timestamps: np.ndarray  # one per TIMESTEP record (frame)
	counts: np.ndarray  # active vehicles at each timestamp


def read_trj(path: Path) -> Trj:
	data = path.read_bytes()
	offset = _FORMAT.size + _DIMENSIONS.size  # skip FORMAT; read DIMENSIONS for bounds
	_, _, _, min_x, min_y, max_x, max_y = _DIMENSIONS.unpack_from(data, _FORMAT.size)
	rows: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)
	timestamps: list[float] = []
	counts: list[int] = []
	current_t = 0.0
	while offset < len(data):
		record_type = data[offset]
		if record_type == _TIMESTEP_TYPE:
			_, current_t = _TIMESTEP.unpack_from(data, offset)
			offset += _TIMESTEP.size
			timestamps.append(current_t)
			counts.append(0)
		elif record_type == _VEHICLE_TYPE:
			v = _VEHICLE.unpack_from(data, offset)
			offset += _VEHICLE.size
			_, vid, _link, _lane, fx, fy, rx, ry, _len, _wid, speed, accel = v
			rows[vid].append((current_t, (fx + rx) / 2.0, (fy + ry) / 2.0, speed, accel))
			if counts:
				counts[-1] += 1
		else:  # unknown/trailing byte: stop
			break
	by_vehicle = {vid: np.array(sorted(r)) for vid, r in rows.items()}
	return Trj(
		bounds=(min_x, min_y, max_x, max_y),
		by_vehicle=by_vehicle,
		timestamps=np.array(timestamps),
		counts=np.array(counts),
	)


def read_track_scores(path: Path) -> np.ndarray:
	"""Detection confidence scores from a track sidecar (header line then CSV)."""
	with path.open(newline="") as handle:
		handle.readline()  # skip the "# fps=... meters_per_pixel=..." header line
		return np.array([float(row["score"]) for row in csv.DictReader(handle)])


_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def background_for(stem: str, video_arg: Path | None) -> np.ndarray | None:
	"""Grab one RGB frame of the clip for this run (a file, or the best match in a folder).

	Folder match is by name prefix up to the first underscore (``cruce_ema`` -> ``cruce_*``),
	which lines a multi-config outputs tree up with a ``resources/`` video folder.
	"""
	if video_arg is None:
		return None
	if video_arg.is_file():
		path: Path | None = video_arg
	elif video_arg.is_dir():
		token = stem.split("_")[0]
		matches = [
			p
			for p in sorted(video_arg.iterdir())
			if p.suffix.lower() in _VIDEO_SUFFIXES and p.stem.split("_")[0] == token
		]
		path = matches[0] if matches else None
	else:
		path = None
	if path is None:
		return None
	capture = cv2.VideoCapture(str(path))
	ok, frame = capture.read()
	capture.release()
	if not ok:
		return None
	rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return rgb


def _draw_background(ax: object, frame: np.ndarray | None) -> None:
	"""Place the filmed scene behind a spatial panel.

	The frame is flipped vertically and drawn with ``origin='lower'`` so it lines up with
	the .trj's y-up image coordinates (the exporter stores ``height - y``).
	"""
	if frame is None:
		return
	height, width = frame.shape[:2]
	ax.imshow(  # type: ignore[attr-defined]
		frame[::-1], extent=(0.0, float(width), 0.0, float(height)), origin="lower", alpha=0.55
	)
	ax.set_xlim(0, width)  # type: ignore[attr-defined]
	ax.set_ylim(0, height)  # type: ignore[attr-defined]


def _all_jerks(trj: Trj) -> np.ndarray:
	"""Jerk = d(accel)/dt across every track (m/s^3)."""
	chunks: list[np.ndarray] = []
	for arr in trj.by_vehicle.values():
		if len(arr) < 3:
			continue
		dt = np.diff(arr[:, 0])
		jerk = np.divide(np.diff(arr[:, 4]), dt, out=np.full_like(dt, np.nan), where=dt > 0)
		chunks.append(jerk[np.isfinite(jerk)])
	return np.concatenate(chunks) if chunks else np.array([])


def _representative_track(base: Trj, smooth: Trj | None) -> int:
	"""The longest track (present in both series when a smoothed one exists)."""
	candidates = base.by_vehicle
	if smooth is not None:
		shared = {v: a for v, a in base.by_vehicle.items() if v in smooth.by_vehicle}
		candidates = shared or base.by_vehicle
	return max(candidates, key=lambda v: len(candidates[v]))


def plot_smoothing(
	base: Trj,
	smooth: Trj | None,
	stem: str,
	out: Path,
	accel_bound: float,
	background: np.ndarray | None = None,
) -> None:
	vid = _representative_track(base, smooth)
	bt = base.by_vehicle[vid]
	st = smooth.by_vehicle[vid] if smooth is not None and vid in smooth.by_vehicle else None
	fig, axes = plt.subplots(2, 2, figsize=(14, 9))
	fig.suptitle(f"{stem} — smoothing (track {vid} shown; jerk over all tracks)", fontsize=13)

	ax = axes[0, 0]
	ax.plot(bt[:, 0], bt[:, 3], color="#d62728", alpha=0.8, lw=1, label="baseline")
	if st is not None:
		ax.plot(st[:, 0], st[:, 3], color="#1f77b4", lw=1.6, label="RTS smoothed")
	ax.set(title="Speed vs time", xlabel="t (s)", ylabel="speed (m/s)")
	ax.legend(fontsize=8)

	ax = axes[0, 1]
	ax.axhspan(-accel_bound, accel_bound, color="#2ca02c", alpha=0.08)
	ax.axhline(accel_bound, color="#2ca02c", lw=0.8, ls="--")
	ax.axhline(-accel_bound, color="#2ca02c", lw=0.8, ls="--")
	ax.plot(bt[:, 0], bt[:, 4], color="#d62728", alpha=0.8, lw=1, label="baseline")
	if st is not None:
		ax.plot(st[:, 0], st[:, 4], color="#1f77b4", lw=1.6, label="RTS smoothed")
	ax.set(
		title=f"Acceleration vs time (±{accel_bound:g} m/s² plausible)",
		xlabel="t (s)",
		ylabel="accel (m/s²)",
	)
	ax.legend(fontsize=8)

	ax = axes[1, 0]
	bj = _all_jerks(base)
	series = [bj[np.abs(bj) < 200]] if bj.size else []
	labels = ["baseline"]
	if smooth is not None:
		sj = _all_jerks(smooth)
		series.append(sj[np.abs(sj) < 200] if sj.size else np.array([]))
		labels.append("RTS smoothed")
	if series and any(s.size for s in series):
		ax.hist(
			series, bins=60, log=True, color=["#d62728", "#1f77b4"][: len(series)], label=labels
		)
		ax.legend(fontsize=8)
	ax.set(title="Jerk distribution (all tracks)", xlabel="jerk (m/s³)", ylabel="count (log)")

	ax = axes[1, 1]
	ax.set_aspect("equal")
	_draw_background(ax, background)
	longest = sorted(base.by_vehicle, key=lambda v: -len(base.by_vehicle[v]))[:6]
	for v in longest:
		a = base.by_vehicle[v]
		ax.plot(a[:, 1], a[:, 2], color="#d62728", alpha=0.45, lw=1.0, zorder=2)
		if smooth is not None and v in smooth.by_vehicle:
			s = smooth.by_vehicle[v]
			ax.plot(s[:, 1], s[:, 2], color="#00e5ff", lw=1.4, zorder=3)
	scene = " over scene" if background is not None else ""
	ax.set(
		title=f"Trajectories{scene}: baseline (red) vs smoothed (cyan)",
		xlabel="x (px)",
		ylabel="y (px)",
	)

	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.savefig(out / f"{stem}_smoothing.png", dpi=120)
	plt.close(fig)


def plot_tracking(
	base: Trj,
	scores: np.ndarray | None,
	stem: str,
	out: Path,
	background: np.ndarray | None = None,
) -> None:
	veh = base.by_vehicle
	starts = {v: a[0, 0] for v, a in veh.items()}
	order = sorted(veh, key=lambda v: starts[v])
	fig, axes = plt.subplots(2, 3, figsize=(18, 9))
	fig.suptitle(f"{stem} — tracking / continuity ({len(veh)} tracks)", fontsize=13)

	ax = axes[0, 0]
	for i, v in enumerate(order):
		a = veh[v]
		ax.hlines(i, a[0, 0], a[-1, 0], color="#1f77b4", lw=0.8)
	ax.set(title="Track lifespans (Gantt)", xlabel="t (s)", ylabel="track (by start)")

	ax = axes[0, 1]
	durations = np.array([a[-1, 0] - a[0, 0] for a in veh.values()])
	ax.hist(durations, bins=40, color="#1f77b4")
	ax.set(title="Track-length distribution", xlabel="duration (s)", ylabel="tracks")

	ax = axes[0, 2]
	if base.timestamps.size:
		ax.plot(base.timestamps, base.counts, color="#1f77b4", lw=1)
	ax.set(title="Active tracks over time", xlabel="t (s)", ylabel="vehicles in frame")

	ax = axes[1, 0]
	ax.set_aspect("equal")
	_draw_background(ax, background)
	births = np.array([a[0, 1:3] for a in veh.values()])
	deaths = np.array([a[-1, 1:3] for a in veh.values()])
	if births.size:
		marker = {"s": 22, "edgecolors": "white", "linewidths": 0.5, "zorder": 3}
		ax.scatter(births[:, 0], births[:, 1], c="#39ff14", label="birth", **marker)
		ax.scatter(deaths[:, 0], deaths[:, 1], c="#ff1744", label="death", **marker)
	if background is None:
		min_x, min_y, max_x, max_y = base.bounds
		ax.set_xlim(min_x, max_x)
		ax.set_ylim(min_y, max_y)
	ax.set(title="Track birth/death positions", xlabel="x (px)", ylabel="y (px)")
	ax.legend(fontsize=8)

	ax = axes[1, 1]
	if scores is not None and scores.size:
		ax.hist(scores, bins=40, color="#1f77b4")
		ax.set(title="Detection confidence", xlabel="score", ylabel="detections")
	else:
		ax.text(0.5, 0.5, "no tracks sidecar\n(export.tracks off)", ha="center", va="center")
		ax.set_axis_off()

	axes[1, 2].set_axis_off()
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.savefig(out / f"{stem}_tracking.png", dpi=120)
	plt.close(fig)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("outputs", type=Path, help="Run outputs folder (walked recursively).")
	parser.add_argument(
		"--out", type=Path, default=None, help="Write PNGs here (default: beside each .trj)."
	)
	parser.add_argument(
		"--accel-bound", type=float, default=8.0, help="±plausible accel band (m/s²)."
	)
	parser.add_argument(
		"--video",
		type=Path,
		default=None,
		help="Clip (or folder of clips, matched by name prefix) drawn behind the spatial panels.",
	)
	args = parser.parse_args()

	if not args.outputs.exists():
		return print(f"no such folder: {args.outputs}") or 1
	baselines = sorted(p for p in args.outputs.rglob("*.trj") if not p.stem.endswith("_smooth"))
	if not baselines:
		return print(f"no .trj files under {args.outputs}") or 1

	for trj_path in baselines:
		stem = trj_path.stem
		smooth_path = trj_path.with_name(f"{stem}_smooth.trj")
		tracks_path = trj_path.with_name(f"{stem}_tracks.csv")
		base = read_trj(trj_path)
		if not base.by_vehicle:
			print(f"skip {stem}: no vehicles")
			continue
		smooth = read_trj(smooth_path) if smooth_path.exists() else None
		scores = read_track_scores(tracks_path) if tracks_path.exists() else None
		out_dir = args.out if args.out is not None else trj_path.parent
		out_dir.mkdir(parents=True, exist_ok=True)
		background = background_for(stem, args.video)
		plot_smoothing(base, smooth, stem, out_dir, args.accel_bound, background)
		plot_tracking(base, scores, stem, out_dir, background)
		notes = (" (+smooth)" if smooth else "") + (" (+scene)" if background is not None else "")
		print(f"{stem}: {len(base.by_vehicle)} tracks{notes} -> {out_dir}/{stem}_*.png")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
