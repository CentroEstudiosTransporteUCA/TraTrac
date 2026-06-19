"""Typer entry point for the smoothing post-pass (``tratrac-smooth``).

Reads a track-observation sidecar (export B, written by a run with ``export.tracks``),
runs the forward+RTS constant-acceleration Kalman smoother per track, and writes a
de-jittered SSAM ``.trj``. Offline and zero-phase — the second pass of the two-pass
smoothing design (see vault/22_smoothing.md). Re-running with different ``--pos-noise``/
``--jerk`` re-tunes the filter with no re-detection.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from tratrac.application.track_smoothing import TrackSample, smooth_to_states
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Point2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.tracks.csv import TrackObservation, TrackRecording, read_tracks

app = typer.Typer(
	name="tratrac-smooth",
	help="Smooth a track-observation file into a de-jittered SSAM .trj (forward+RTS Kalman).",
	no_args_is_help=True,
)

# SsamTrjExporter is a data exporter and ignores the frame pixels; a 1x1 placeholder
# satisfies the port without carrying real imagery into the post-pass.
_PLACEHOLDER_PIXELS: NDArray[np.uint8] = np.zeros((1, 1, 3), dtype=np.uint8)


@app.command()
def smooth(
	tracks: Annotated[
		Path, typer.Argument(exists=True, dir_okay=False, help="Track-observation file (export B).")
	],
	out: Annotated[Path, typer.Option("--out", "-o", dir_okay=False, help="Output .trj path.")],
	pos_noise: Annotated[
		float,
		typer.Option("--pos-noise", help="Measurement-noise std in px (detector center jitter)."),
	] = 2.0,
	jerk: Annotated[
		float,
		typer.Option("--jerk", help="Process jerk spectral density; higher = more responsive."),
	] = 20.0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite an existing output.")
	] = False,
) -> None:
	"""Smooth TRACKS into a de-jittered .trj at --out."""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	try:
		recording = read_tracks(tracks)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	states_by_frame = _smooth_recording(recording, pos_noise=pos_noise, jerk=jerk)

	out.parent.mkdir(parents=True, exist_ok=True)
	fps = recording.metadata.fps
	with SsamTrjExporter(out, recording.metadata, scale=recording.scale) as exporter:
		for frame_index in sorted(states_by_frame):
			exporter.emit_frame(
				frame_index / fps,
				states_by_frame[frame_index],
				Frame(index=frame_index, pixels=_PLACEHOLDER_PIXELS),
			)
	typer.echo(
		f"Smoothed {tracks} -> {out} "
		f"({len(states_by_frame)} frames, pos_noise={pos_noise}px, jerk={jerk})."
	)


def _smooth_recording(
	recording: TrackRecording, *, pos_noise: float, jerk: float
) -> dict[int, list[VehicleState]]:
	"""Group observations by track, smooth each, and regroup the states by frame index."""
	by_track: dict[int, list[TrackObservation]] = defaultdict(list)
	for observation in recording.observations:
		by_track[observation.track_id].append(observation)

	fps = recording.metadata.fps
	states_by_frame: dict[int, list[VehicleState]] = defaultdict(list)
	for track_id, observations in by_track.items():
		observations.sort(key=lambda o: o.frame_index)
		samples = [
			TrackSample(
				frame_index=o.frame_index,
				timestamp_seconds=o.frame_index / fps,
				center=Point2D(o.cx, o.cy),
				width=o.width,
				height=o.height,
			)
			for o in observations
		]
		states = smooth_to_states(
			track_id, samples, recording.scale, pos_noise=pos_noise, jerk=jerk
		)
		for sample, state in zip(samples, states, strict=True):
			states_by_frame[sample.frame_index].append(state)
	return states_by_frame


if __name__ == "__main__":
	app()
