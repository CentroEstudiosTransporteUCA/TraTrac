"""Typer entry point for the smoothing post-pass (``tratrac-smooth``).

Reads a track-observation sidecar (export B, written by a run with ``export.tracks``),
runs the forward+RTS constant-acceleration Kalman smoother per track, and writes a
de-jittered SSAM ``.trj``. Offline and zero-phase — the second pass of the two-pass
smoothing design (see vault/22_smoothing.md). Re-running with different ``--pos-noise``/
``--jerk`` re-tunes the filter with no re-detection.

With ``--video``/``--video-out`` it also draws the *smoothed* trajectories onto the source
clip (reusing ``OverlayVideoExporter``), so the smoothed variants get a real trajectory
overlay — not just violation marks. For an ego-motion run pass ``--transforms`` (the run's
transform CSV) so the smoothed global-frame coordinates map back onto the raw video,
exactly as the pipeline overlay does.
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
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.composite import CompositeTrajectoryExporter
from tratrac.infrastructure.export.overlay_video import OverlayVideoExporter
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.tracks.csv import TrackObservation, TrackRecording, read_tracks
from tratrac.infrastructure.transform.csv import read_transforms
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac-smooth",
	help="Smooth a track-observation file into a de-jittered SSAM .trj (forward+RTS Kalman).",
	no_args_is_help=True,
)

# SsamTrjExporter is a data exporter and ignores the frame pixels; a 1x1 placeholder
# satisfies the port when no real video is drawn.
_PLACEHOLDER_PIXELS: NDArray[np.uint8] = np.zeros((1, 1, 3), dtype=np.uint8)


class _CurrentTransform:
	"""Mutable holder so the overlay's ``transform_source`` reads the frame's transform."""

	def __init__(self) -> None:
		self.value = Transform2D.identity()

	def get(self) -> Transform2D:
		return self.value


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
	video: Annotated[
		Path | None,
		typer.Option(
			"--video",
			exists=True,
			dir_okay=False,
			help="Source clip to draw the smoothed trajectories on; required with --video-out.",
		),
	] = None,
	video_out: Annotated[
		Path | None,
		typer.Option(
			"--video-out", dir_okay=False, help="Overlay video of the smoothed trajectories."
		),
	] = None,
	transforms: Annotated[
		Path | None,
		typer.Option(
			"--transforms",
			dir_okay=False,
			help="The run's transform CSV; maps smoothed global-frame coords onto the raw video "
			"(ego-motion runs).",
		),
	] = None,
	video_trail: Annotated[
		int, typer.Option("--video-trail", help="Overlay trail length in frames; 0 = whole path.")
	] = 0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite existing outputs.")
	] = False,
) -> None:
	"""Smooth TRACKS into a de-jittered .trj at --out (optionally an overlay video)."""
	if video_out is not None and video is None:
		raise typer.BadParameter("--video-out requires --video (the source clip to draw on).")
	for target in (out, video_out):
		if target is not None and target.exists() and not force:
			raise typer.BadParameter(f"{target} already exists; pass --force to overwrite.")
	try:
		recording = read_tracks(tracks)
		transforms_map = read_transforms(transforms) if transforms is not None else {}
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	states_by_frame = _smooth_recording(recording, pos_noise=pos_noise, jerk=jerk)
	out.parent.mkdir(parents=True, exist_ok=True)

	if video_out is None:
		_emit_trj(out, recording, states_by_frame)
	else:
		assert video is not None  # guaranteed by the check above
		video_out.parent.mkdir(parents=True, exist_ok=True)
		_emit_trj_and_overlay(
			out, video, video_out, recording, states_by_frame, transforms_map, video_trail
		)

	overlay_note = f" + overlay {video_out}" if video_out is not None else ""
	typer.echo(
		f"Smoothed {tracks} -> {out}{overlay_note} "
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
	out: Path, recording: TrackRecording, states_by_frame: dict[int, list[VehicleState]]
) -> None:
	"""Write the smoothed .trj only (no video); data exporter ignores the frame pixels."""
	fps = recording.metadata.fps
	with SsamTrjExporter(out, recording.metadata, scale=recording.scale) as exporter:
		for frame_index in sorted(states_by_frame):
			exporter.emit_frame(
				frame_index / fps,
				states_by_frame[frame_index],
				Frame(index=frame_index, pixels=_PLACEHOLDER_PIXELS),
			)


def _emit_trj_and_overlay(
	out: Path,
	video: Path,
	video_out: Path,
	recording: TrackRecording,
	states_by_frame: dict[int, list[VehicleState]],
	transforms_map: dict[int, Transform2D],
	video_trail: int,
) -> None:
	"""Write the .trj and an overlay video of the smoothed trajectories over the source clip."""
	if not states_by_frame:
		_emit_trj(out, recording, states_by_frame)
		return
	fps = recording.metadata.fps
	lo, hi = min(states_by_frame), max(states_by_frame)
	current = _CurrentTransform()
	try:
		source = OpenCvVideoSource(video, start_seconds=lo / fps, end_seconds=(hi + 0.5) / fps)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc
	with source:
		exporter = CompositeTrajectoryExporter(
			[
				SsamTrjExporter(out, recording.metadata, scale=recording.scale),
				OverlayVideoExporter(
					video_out,
					recording.metadata,
					scale=recording.scale,
					trail_length=video_trail,
					transform_source=current.get,
				),
			]
		)
		with exporter:
			for frame in source.frames():
				current.value = transforms_map.get(frame.index, Transform2D.identity())
				exporter.emit_frame(frame.index / fps, states_by_frame.get(frame.index, []), frame)


if __name__ == "__main__":
	app()
