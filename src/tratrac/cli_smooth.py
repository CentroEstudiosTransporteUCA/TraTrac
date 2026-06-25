"""Typer entry point for the smoothing post-pass (``tratrac-smooth``).

Reads a track-observation sidecar (export B, written by a run with ``export.tracks``),
runs the forward+RTS constant-acceleration Kalman smoother per track, and writes a
de-jittered SSAM ``.trj``. Offline and zero-phase — the second pass of the two-pass
smoothing design (see vault/22_smoothing.md). Re-running with different ``--pos-noise``/
``--jerk`` re-tunes the filter with no re-detection.

Rendering is a separate step: to draw the smoothed trajectories over the clip, run
``tratrac-render`` on the smoothed ``.trj`` (see vault/20_video_export.md).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from tratrac.application.track_smoothing import TrackSample, smooth_to_states
from tratrac.domain.geometry import Point2D
from tratrac.domain.ports import TrajectoryExporter
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording, read_tracks

app = typer.Typer(
	name="tratrac-smooth",
	help="Smooth a track-observation file into a de-jittered SSAM .trj (forward+RTS Kalman).",
	no_args_is_help=True,
)


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
	timestep_precision: Annotated[
		float,
		typer.Option(
			"--timestep-precision",
			help="Min seconds between exported TIMESTEPs; 0 = every frame. Thins only the .trj "
			"output (the record keeps every frame).",
		),
	] = 0.0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite existing outputs.")
	] = False,
) -> None:
	"""Smooth TRACKS into a de-jittered .trj at --out.

	To draw the smoothed trajectories over the clip, render the output with
	``tratrac-render`` (see vault/20_video_export.md).
	"""
	if out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	if timestep_precision < 0.0:
		raise typer.BadParameter("--timestep-precision must be >= 0 (0 = every frame).")
	try:
		recording = read_tracks(tracks)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	states_by_frame = _smooth_recording(recording, pos_noise=pos_noise, jerk=jerk)
	out.parent.mkdir(parents=True, exist_ok=True)
	_emit_trj(out, recording, states_by_frame, timestep_precision=timestep_precision)

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


def _emit_trj(
	out: Path,
	recording: TrackRecording,
	states_by_frame: dict[int, list[VehicleState]],
	*,
	timestep_precision: float,
) -> None:
	"""Write the smoothed .trj from the per-frame states, optionally decimating TIMESTEPs."""
	fps = recording.metadata.fps
	exporter: TrajectoryExporter = SsamTrjExporter(out, recording.metadata, scale=recording.scale)
	if timestep_precision > 0.0:
		# Thin the exported TIMESTEP stream; the smoothing still uses every observation.
		exporter = DecimatingTrajectoryExporter(
			exporter, min_interval_seconds=timestep_precision, fps=fps
		)
	with exporter:
		for frame_index in sorted(states_by_frame):
			exporter.emit_frame(frame_index / fps, states_by_frame[frame_index])


if __name__ == "__main__":
	app()
