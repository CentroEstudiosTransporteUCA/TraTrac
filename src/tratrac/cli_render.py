"""Typer entry point for the trajectory renderer (``tratrac-render``).

Post-hoc visualization: draws an SSAM ``.trj``'s bumpers/IDs/trails over its source
clip and writes an overlay video. This is the rendering that used to run *inside*
the live pipeline (``export.video_out``) and in ``tratrac-smooth --video-out`` —
pulled out into its own step so a run only detects/tracks/exports and never pays the
per-frame video-encode cost (see vault/20_video_export.md).

It reuses the pipeline's ``OverlayVideoExporter`` drawing engine; the only thing
that changes from the old in-pipeline path is the *source* of the vehicle states:
``read_trj`` instead of live tracking. Frames are aligned by ``round(timestamp * fps)``
(fps from the video, since the ``.trj`` carries time but not fps).

With ``--violations`` it also marks each non-compliant instance from a ``validate_trj.py``
violations CSV (in red, in the same pass) — so one render gives "frame + trajectories +
violations" without a second encode.

For an ego-motion run the ``.trj`` (and violation) positions are in the global
stabilization frame; pass ``--transforms`` (the run's ``export.transform_csv``) so each
frame's inverse transform maps everything back onto the raw video.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.overlay_video import OverlayVideoExporter
from tratrac.infrastructure.export.ssam_trj import read_trj
from tratrac.infrastructure.transform.csv import read_transforms
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac-render",
	help="Render an SSAM .trj's trajectories (and optional violations) over its source clip.",
	no_args_is_help=True,
)

# Seconds of clip kept past the last TIMESTEP so its frame is included (the .trj carries
# time but not fps, so the window is set in seconds). Small over-render; bounds the output.
_TAIL_BUFFER_SECONDS = 0.5

_VIOLATION_COLOR = (0, 0, 255)  # BGR red — stands out against the trajectory colours.
_REQUIRED_VIOLATION_COLUMNS = ("timestamp_s", "vehicle_id", "check", "centroid_x", "centroid_y")


class _CurrentTransform:
	"""Mutable holder so the overlay's ``transform_source`` reads the frame's transform."""

	def __init__(self) -> None:
		self.value = Transform2D.identity()

	def get(self) -> Transform2D:
		return self.value


@dataclass(frozen=True, slots=True)
class _ViolationMark:
	"""One vehicle's violations on a frame, in global-frame image coords (mapped at draw)."""

	cx: float
	cy: float
	label: str


@app.command()
def render(
	video: Annotated[
		Path, typer.Argument(exists=True, dir_okay=False, help="Source clip the .trj was run on.")
	],
	trj: Annotated[
		Path,
		typer.Option("--trj", exists=True, dir_okay=False, help="SSAM .trj to draw."),
	],
	out: Annotated[
		Path, typer.Option("--out", "-o", dir_okay=False, help="Overlay video output (mp4).")
	],
	transforms: Annotated[
		Path | None,
		typer.Option(
			"--transforms",
			dir_okay=False,
			help="The run's transform CSV (export.transform_csv); maps global-frame coords back "
			"onto the raw video for an ego-motion run. Omit for a non-stabilized run.",
		),
	] = None,
	violations: Annotated[
		Path | None,
		typer.Option(
			"--violations",
			exists=True,
			dir_okay=False,
			help="validate_trj.py violations CSV; marks each non-compliant instance in red.",
		),
	] = None,
	checks: Annotated[
		str | None,
		typer.Option(
			"--checks",
			help="Comma-separated subset of violation check names to draw (default: all in the CSV).",
		),
	] = None,
	trail: Annotated[
		int, typer.Option("--trail", help="Trail length in frames; 0 = whole path.")
	] = 0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite an existing output.")
	] = False,
) -> None:
	"""Draw TRJ's trajectories (and optional --violations) over VIDEO into an overlay --out."""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	try:
		recording = read_trj(trj)
		transforms_map = read_transforms(transforms) if transforms is not None else {}
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	if not recording.frames:
		typer.echo(f"{trj} has no trajectories; nothing to render.")
		return

	check_filter = {c.strip() for c in checks.split(",") if c.strip()} if checks else None

	out.parent.mkdir(parents=True, exist_ok=True)
	# Render only the clip span the .trj covers, not the whole source — otherwise a short
	# analysis window on a long clip would re-encode the entire video. TIMESTEPs are
	# absolute seconds (vault/17), so they bound the window directly; the tail buffer keeps
	# the last covered frame. Windowed frames keep their absolute index, so the
	# round(timestamp * fps) alignment below still holds.
	start_seconds = min(f.timestamp_seconds for f in recording.frames)
	end_seconds = max(f.timestamp_seconds for f in recording.frames) + _TAIL_BUFFER_SECONDS
	try:
		source = OpenCvVideoSource(video, start_seconds=start_seconds, end_seconds=end_seconds)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	with source:
		fps = source.metadata.fps
		# Bucket states (and violation marks) onto absolute video frames by round(ts * fps).
		states_by_frame: dict[int, list[VehicleState]] = defaultdict(list)
		for trj_frame in recording.frames:
			states_by_frame[round(trj_frame.timestamp_seconds * fps)].extend(trj_frame.states)

		violations_by_frame: dict[int, list[_ViolationMark]] = {}
		if violations is not None:
			try:
				violations_by_frame = _parse_violations(violations, fps, check_filter)
			except (ValueError, OSError) as exc:
				raise typer.BadParameter(str(exc)) from exc

		current = _CurrentTransform()
		exporter = OverlayVideoExporter(
			out,
			source.metadata,
			scale=recording.scale,
			trail_length=trail,
			transform_source=current.get,
			annotate=_violation_annotator(violations_by_frame) if violations is not None else None,
		)
		with exporter:
			for frame in source.frames():
				current.value = transforms_map.get(frame.index, Transform2D.identity())
				exporter.emit_frame(frame.index / fps, states_by_frame.get(frame.index, []), frame)

	mark_total = sum(len(v) for v in violations_by_frame.values())
	violation_note = f", {mark_total} violation marks" if violations is not None else ""
	typer.echo(
		f"Rendered {trj} over {video} -> {out} "
		f"({len(recording.frames)} timesteps, trail={trail or 'full'}{violation_note})."
	)


def _parse_violations(
	path: Path, fps: float, checks: set[str] | None
) -> dict[int, list[_ViolationMark]]:
	"""Group a validate_trj.py violations CSV by absolute video frame (round(timestamp * fps)).

	Positions are kept in the CSV's (global-frame) image coordinates and mapped to the raw
	frame at draw time via the same transform the trajectories use. Multiple checks failing
	for one vehicle on one frame collapse into a single mark; ``checks`` filters by name.
	"""
	checks_by_key: dict[tuple[int, int], set[str]] = defaultdict(set)
	centroid_by_key: dict[tuple[int, int], tuple[float, float]] = {}
	with path.open(newline="") as handle:
		reader = csv.DictReader(handle)
		missing = [c for c in _REQUIRED_VIOLATION_COLUMNS if c not in (reader.fieldnames or [])]
		if missing:
			raise ValueError(f"{path}: missing columns {missing}; not a validate_trj.py CSV.")
		for row in reader:
			check = row["check"]
			if checks is not None and check not in checks:
				continue
			key = (round(float(row["timestamp_s"]) * fps), int(row["vehicle_id"]))
			checks_by_key[key].add(check)
			centroid_by_key[key] = (float(row["centroid_x"]), float(row["centroid_y"]))

	by_frame: dict[int, list[_ViolationMark]] = defaultdict(list)
	for key, names in checks_by_key.items():
		cx, cy = centroid_by_key[key]
		by_frame[key[0]].append(_ViolationMark(cx, cy, " / ".join(sorted(names))))
	return dict(by_frame)


def _violation_annotator(
	violations_by_frame: dict[int, list[_ViolationMark]],
) -> Callable[[NDArray[np.uint8], int, Transform2D], None]:
	"""Build the overlay annotate hook that draws a frame's violation marks (mapped to raw)."""

	def annotate(canvas: NDArray[np.uint8], frame_index: int, to_raw: Transform2D) -> None:
		import cv2  # lazy: keep the module import-light, matching OverlayVideoExporter

		for mark in violations_by_frame.get(frame_index, []):
			raw = to_raw.apply(Point2D(mark.cx, mark.cy))
			center = (round(raw.x), round(raw.y))
			cv2.circle(canvas, center, 16, _VIOLATION_COLOR, 2)
			cv2.putText(
				canvas,
				mark.label,
				(center[0] + 18, center[1] + 4),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				_VIOLATION_COLOR,
				1,
				cv2.LINE_AA,
			)

	return annotate


if __name__ == "__main__":
	app()
