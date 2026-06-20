"""Plot the per-run diagnostics from a TraTrac outputs folder.

Standalone (stdlib + numpy + matplotlib; no package imports, so it works even if the
package is broken). Walks an outputs directory, finds every baseline ``.trj`` (and its
``_smooth.trj`` / ``_tracks.csv`` siblings, if present), and writes **one PNG per graph**
into a per-run folder ``<stem>/``:

  01_speed_vs_time · 02_acceleration · 03_jerk_distribution · 04_trajectories
  05_track_lifespans · 06_track_length · 07_active_tracks · 08_birth_death
  09_detection_confidence (only when the tracks sidecar is present)

01-04 are the de-jitter story (baseline vs RTS-smoothed); 05-09 are tracking/continuity.

The `.trj` is parsed inline (SSAM v1.04): positions are in image pixels; speed/accel are
metric (m/s, m/s^2) when the run had a real GSD scale.

For the scene behind the trajectory and birth/death panels:
  - ``--video FILE`` (or a folder, prefix-matched) draws one still frame — fine for a
    static camera.
  - ``--scout-dir DIR`` draws a **mosaic of the whole swept area**: the scout's
    ``frame_*.png`` anchors warped by the run's ``<stem>_transforms.csv`` into the .trj's
    global frame, with transparent gaps where no frame covered. Use this for moving-drone
    runs whose trajectories span more than one frame.

Usage:
    uv run python scripts/plot_run.py OUTPUTS_DIR [--out DIR] [--accel-bound 8.0]
        [--video CLIP_OR_FOLDER] [--scout-dir SCOUT_OR_PARENT]
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import defaultdict
from collections.abc import Callable
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

# A draw function takes a matplotlib Axes (untyped: matplotlib is unstubbed here).
DrawFn = Callable[[object], None]


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
Pose = tuple[float, float, float, float, float, float]  # a, b, c, d, tx, ty


@dataclass
class Background:
	"""An image to sit behind a spatial panel, already placed in the .trj's coordinates.

	``image`` is drawn with ``imshow(extent=..., origin=...)``; ``clip`` forces the axes
	to the image bounds (single frame) or lets them autoscale to also fit the data (the
	mosaic, which may be smaller than the trajectories' full extent).
	"""

	image: np.ndarray
	extent: tuple[float, float, float, float]  # left, right, bottom, top (trj coords)
	origin: str
	clip: bool


def _read_transforms(path: Path) -> dict[int, Pose]:
	"""Per-frame ego-motion poses (frame -> a,b,c,d,tx,ty) from a run's transform CSV."""
	out: dict[int, Pose] = {}
	with path.open(newline="") as handle:
		for row in csv.DictReader(handle):
			out[int(row["frame"])] = (
				float(row["a"]),
				float(row["b"]),
				float(row["c"]),
				float(row["d"]),
				float(row["tx"]),
				float(row["ty"]),
			)
	return out


def _nearest_pose(transforms: dict[int, Pose], index: int) -> Pose | None:
	"""The pose at ``index``, or the closest available frame's (anchors may be skipped)."""
	if not transforms:
		return None
	if index in transforms:
		return transforms[index]
	return transforms[min(transforms, key=lambda k: abs(k - index))]


def _anchor_frames(scout_dir: Path) -> list[tuple[int, Path]]:
	"""(frame_index, png) pairs from a scout dir's ``frame_<index>.png`` anchors."""
	anchors: list[tuple[int, Path]] = []
	for png in sorted(scout_dir.glob("frame_*.png")):
		try:
			anchors.append((int(png.stem.split("_")[1]), png))
		except (IndexError, ValueError):
			continue
	return anchors


def _scout_dir_for(stem: str, scout_arg: Path | None) -> Path | None:
	"""The scout dir for this run: ``scout_arg`` itself, or its prefix-matched subdir."""
	if scout_arg is None or not scout_arg.is_dir():
		return None
	if any(scout_arg.glob("frame_*.png")):
		return scout_arg
	token = stem.split("_")[0]
	for sub in sorted(scout_arg.iterdir()):
		if sub.is_dir() and sub.name.split("_")[0] == token and any(sub.glob("frame_*.png")):
			return sub
	return None


def _build_mosaic(
	anchors: list[tuple[int, Path]], transforms: dict[int, Pose], h_raw: int, max_dim: int = 4000
) -> Background | None:
	"""Warp each anchor frame by the run's pose into one RGBA canvas (transparent gaps).

	The result is in the .trj's global frame (the run's transforms define it), flipped to
	the .trj's y-up convention via the extent, so trajectories overlay directly.
	"""
	placements: list[tuple[np.ndarray, Pose]] = []
	xs: list[float] = []
	ys: list[float] = []
	for index, png in anchors:
		pose = _nearest_pose(transforms, index)
		bgr = cv2.imread(str(png))
		if pose is None or bgr is None:
			continue
		a, b, c, d, tx, ty = pose
		img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		h, w = img.shape[:2]
		for x, y in ((0, 0), (w, 0), (0, h), (w, h)):
			xs.append(a * x + b * y + tx)
			ys.append(c * x + d * y + ty)
		placements.append((img, pose))
	if not placements:
		return None
	gx0, gx1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
	gy0, gy1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
	width, height = max(1, gx1 - gx0), max(1, gy1 - gy0)
	scale = min(1.0, max_dim / max(width, height))  # cap canvas resolution; extent is unchanged
	cw, ch = max(1, round(width * scale)), max(1, round(height * scale))
	canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
	for img, (a, b, c, d, tx, ty) in placements:
		matrix = np.array(
			[[a * scale, b * scale, (tx - gx0) * scale], [c * scale, d * scale, (ty - gy0) * scale]]
		)
		warped = cv2.warpAffine(img, matrix, (cw, ch))
		mask = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), matrix, (cw, ch))
		covered = mask > 0
		canvas[covered, :3] = warped[covered]
		canvas[covered, 3] = 255
	extent = (float(gx0), float(gx1), float(h_raw - gy1), float(h_raw - gy0))
	return Background(canvas, extent, "upper", clip=False)


def _single_frame(stem: str, video_arg: Path | None) -> Background | None:
	"""One frame of the clip (a file, or the prefix-matched video in a folder)."""
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
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	height, width = rgb.shape[:2]
	return Background(rgb[::-1], (0.0, float(width), 0.0, float(height)), "lower", clip=True)


def background_for(
	stem: str,
	video_arg: Path | None,
	scout_arg: Path | None,
	transforms_path: Path | None,
	h_raw: int,
) -> Background | None:
	"""Build the scene background: a scout-anchor mosaic if available, else a single frame.

	The mosaic needs scout anchor PNGs (``--scout-dir``) plus the run's own transform CSV
	(``<stem>_transforms.csv``); together they place the swept area in the .trj's frame.
	"""
	scout_dir = _scout_dir_for(stem, scout_arg)
	if scout_dir is not None and transforms_path is not None and transforms_path.exists():
		mosaic = _build_mosaic(_anchor_frames(scout_dir), _read_transforms(transforms_path), h_raw)
		if mosaic is not None:
			return mosaic
	return _single_frame(stem, video_arg)


def _draw_background(ax, background: Background | None) -> None:
	"""Place the scene (single frame or scout mosaic) behind a spatial panel."""
	if background is None:
		return
	ax.imshow(background.image, extent=background.extent, origin=background.origin, alpha=0.55)
	if background.clip:
		left, right, bottom, top = background.extent
		ax.set_xlim(left, right)
		ax.set_ylim(bottom, top)


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


def _draw_speed(ax, bt: np.ndarray, st: np.ndarray | None) -> None:
	ax.plot(bt[:, 0], bt[:, 3], color="#d62728", alpha=0.8, lw=1, label="baseline")
	if st is not None:
		ax.plot(st[:, 0], st[:, 3], color="#1f77b4", lw=1.6, label="RTS smoothed")
	ax.set(title="Speed vs time", xlabel="t (s)", ylabel="speed (m/s)")
	ax.legend(fontsize=8)


def _draw_accel(ax, bt: np.ndarray, st: np.ndarray | None, bound: float) -> None:
	ax.axhspan(-bound, bound, color="#2ca02c", alpha=0.08)
	ax.axhline(bound, color="#2ca02c", lw=0.8, ls="--")
	ax.axhline(-bound, color="#2ca02c", lw=0.8, ls="--")
	ax.plot(bt[:, 0], bt[:, 4], color="#d62728", alpha=0.8, lw=1, label="baseline")
	if st is not None:
		ax.plot(st[:, 0], st[:, 4], color="#1f77b4", lw=1.6, label="RTS smoothed")
	ax.set(
		title=f"Acceleration vs time (±{bound:g} m/s² plausible)",
		xlabel="t (s)",
		ylabel="accel (m/s²)",
	)
	ax.legend(fontsize=8)


def _draw_jerk(ax, base: Trj, smooth: Trj | None) -> None:
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


def _draw_trajectories(ax, base: Trj, smooth: Trj | None, background: Background | None) -> None:
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


def _draw_gantt(ax, veh: dict[int, np.ndarray]) -> None:
	order = sorted(veh, key=lambda v: veh[v][0, 0])
	for i, v in enumerate(order):
		a = veh[v]
		ax.hlines(i, a[0, 0], a[-1, 0], color="#1f77b4", lw=0.8)
	ax.set(title="Track lifespans (Gantt)", xlabel="t (s)", ylabel="track (by start)")


def _draw_length(ax, veh: dict[int, np.ndarray]) -> None:
	durations = np.array([a[-1, 0] - a[0, 0] for a in veh.values()])
	ax.hist(durations, bins=40, color="#1f77b4")
	ax.set(title="Track-length distribution", xlabel="duration (s)", ylabel="tracks")


def _draw_active(ax, base: Trj) -> None:
	if base.timestamps.size:
		ax.plot(base.timestamps, base.counts, color="#1f77b4", lw=1)
	ax.set(title="Active tracks over time", xlabel="t (s)", ylabel="vehicles in frame")


def _draw_birth_death(
	ax,
	veh: dict[int, np.ndarray],
	bounds: tuple[int, int, int, int],
	background: Background | None,
) -> None:
	ax.set_aspect("equal")
	_draw_background(ax, background)
	births = np.array([a[0, 1:3] for a in veh.values()])
	deaths = np.array([a[-1, 1:3] for a in veh.values()])
	if births.size:
		marker = {"s": 22, "edgecolors": "white", "linewidths": 0.5, "zorder": 3}
		ax.scatter(births[:, 0], births[:, 1], c="#39ff14", label="birth", **marker)
		ax.scatter(deaths[:, 0], deaths[:, 1], c="#ff1744", label="death", **marker)
	if background is None:
		min_x, min_y, max_x, max_y = bounds
		ax.set_xlim(min_x, max_x)
		ax.set_ylim(min_y, max_y)
	ax.set(title="Track birth/death positions", xlabel="x (px)", ylabel="y (px)")
	ax.legend(fontsize=8)


def _draw_confidence(ax, scores: np.ndarray) -> None:
	ax.hist(scores, bins=40, color="#1f77b4")
	ax.set(title="Detection confidence", xlabel="score", ylabel="detections")


def _render(folder: Path, name: str, stem: str, draw: DrawFn, figsize: tuple[float, float]) -> None:
	"""One graph -> one PNG. Prepends the run name to whatever title the draw set."""
	fig, ax = plt.subplots(figsize=figsize)
	draw(ax)
	ax.set_title(f"{stem} — {ax.get_title()}")
	fig.tight_layout()
	fig.savefig(folder / f"{name}.png", dpi=120)
	plt.close(fig)


def generate(
	base: Trj,
	smooth: Trj | None,
	scores: np.ndarray | None,
	stem: str,
	folder: Path,
	accel_bound: float,
	background: Background | None,
) -> int:
	"""Write one PNG per graph into ``folder``; returns the number of PNGs written."""
	folder.mkdir(parents=True, exist_ok=True)
	vid = _representative_track(base, smooth)
	bt = base.by_vehicle[vid]
	st = smooth.by_vehicle[vid] if smooth is not None and vid in smooth.by_vehicle else None
	veh = base.by_vehicle
	graphs: list[tuple[str, DrawFn, tuple[float, float]]] = [
		("01_speed_vs_time", lambda ax: _draw_speed(ax, bt, st), (7.0, 5.0)),
		("02_acceleration", lambda ax: _draw_accel(ax, bt, st, accel_bound), (7.0, 5.0)),
		("03_jerk_distribution", lambda ax: _draw_jerk(ax, base, smooth), (7.0, 5.0)),
		(
			"04_trajectories",
			lambda ax: _draw_trajectories(ax, base, smooth, background),
			(8.0, 6.0),
		),
		("05_track_lifespans", lambda ax: _draw_gantt(ax, veh), (8.0, 6.0)),
		("06_track_length", lambda ax: _draw_length(ax, veh), (7.0, 5.0)),
		("07_active_tracks", lambda ax: _draw_active(ax, base), (9.0, 4.5)),
		(
			"08_birth_death",
			lambda ax: _draw_birth_death(ax, veh, base.bounds, background),
			(8.0, 6.0),
		),
	]
	if scores is not None and scores.size:
		graphs.append(
			("09_detection_confidence", lambda ax: _draw_confidence(ax, scores), (7.0, 5.0))
		)
	for name, draw, figsize in graphs:
		_render(folder, name, stem, draw, figsize)
	return len(graphs)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("outputs", type=Path, help="Run outputs folder (walked recursively).")
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Write per-run folders here (default: beside each .trj).",
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
	parser.add_argument(
		"--scout-dir",
		type=Path,
		default=None,
		help="Scout dir (or parent of per-clip dirs) of frame_*.png anchors; warped by the "
		"run's <stem>_transforms.csv into a swept-area mosaic background.",
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
		transforms_path = trj_path.with_name(f"{stem}_transforms.csv")
		transforms = transforms_path if transforms_path.exists() else None
		base_dir = args.out if args.out is not None else trj_path.parent
		folder = base_dir / stem
		background = background_for(stem, args.video, args.scout_dir, transforms, base.bounds[3])
		count = generate(base, smooth, scores, stem, folder, args.accel_bound, background)
		scene = "" if background is None else (" (+mosaic)" if not background.clip else " (+scene)")
		notes = (" (+smooth)" if smooth else "") + scene
		print(f"{stem}: {len(base.by_vehicle)} tracks{notes} -> {folder}/ ({count} png)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
