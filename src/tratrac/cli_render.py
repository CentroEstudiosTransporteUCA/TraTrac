"""Typer entry point for the trajectory renderer (``tratrac-render``).

Post-hoc visualization: draws an SSAM ``.trj``'s bumpers/IDs/trails over its source
clip and writes an overlay video. This is the rendering that used to run *inside*
the live pipeline (``export.video_out``) and in ``tratrac-smooth --video-out`` —
pulled out into its own step so a run only detects/tracks/exports and never pays the
per-frame video-encode cost (see vault/20_video_export.md).

It reuses the pipeline's ``OverlayVideoExporter`` drawing engine; the only thing
that changes from the old in-pipeline path is the *source* of the vehicle states:
``read_trj`` instead of live tracking. Frames are aligned by ``round(timestamp * fps)``
(fps from the video, since the ``.trj`` carries time but not fps), the same absolute
alignment ``scripts/render_violations.py`` uses, so a violation overlay can stack on
top of this output.

For an ego-motion run the ``.trj`` positions are in the global stabilization frame;
pass ``--transforms`` (the run's ``export.transform_csv``) so each frame's inverse
transform maps the trajectories back onto the raw video.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from tratrac.domain.geometry import Transform2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.overlay_video import OverlayVideoExporter
from tratrac.infrastructure.export.ssam_trj import read_trj
from tratrac.infrastructure.transform.csv import read_transforms
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

app = typer.Typer(
	name="tratrac-render",
	help="Render an SSAM .trj's trajectories over its source clip into an overlay video.",
	no_args_is_help=True,
)

# Seconds of clip kept past the last TIMESTEP so its frame is included (the .trj carries
# time but not fps, so the window is set in seconds). Small over-render; bounds the output.
_TAIL_BUFFER_SECONDS = 0.5


class _CurrentTransform:
	"""Mutable holder so the overlay's ``transform_source`` reads the frame's transform."""

	def __init__(self) -> None:
		self.value = Transform2D.identity()

	def get(self) -> Transform2D:
		return self.value


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
	trail: Annotated[
		int, typer.Option("--trail", help="Trail length in frames; 0 = whole path.")
	] = 0,
	force: Annotated[
		bool, typer.Option("--force/--no-force", help="Overwrite an existing output.")
	] = False,
) -> None:
	"""Draw TRJ's trajectories over VIDEO into an overlay --out."""
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

	out.parent.mkdir(parents=True, exist_ok=True)
	# Render only the clip span the .trj covers, not the whole source — otherwise a short
	# analysis window on a long clip would re-encode the entire video. TIMESTEPs are
	# absolute seconds (vault/17), so they bound the window directly; the tail buffer keeps
	# the last covered frame. Windowed frames keep their absolute index, so the
	# round(timestamp * fps) alignment below (the same render_violations uses) still holds.
	start_seconds = min(f.timestamp_seconds for f in recording.frames)
	end_seconds = max(f.timestamp_seconds for f in recording.frames) + _TAIL_BUFFER_SECONDS
	try:
		source = OpenCvVideoSource(video, start_seconds=start_seconds, end_seconds=end_seconds)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	with source:
		fps = source.metadata.fps
		# Bucket states onto absolute video frames. The .trj carries time, not fps, so
		# fps comes from the clip — the same round(timestamp * fps) alignment used by
		# render_violations, which keeps overlays on the correct absolute frame even for
		# a --start-trimmed run (whose TIMESTEPs stay absolute, see vault/17).
		states_by_frame: dict[int, list[VehicleState]] = defaultdict(list)
		for trj_frame in recording.frames:
			index = round(trj_frame.timestamp_seconds * fps)
			states_by_frame[index].extend(trj_frame.states)

		current = _CurrentTransform()
		exporter = OverlayVideoExporter(
			out,
			source.metadata,
			scale=recording.scale,
			trail_length=trail,
			transform_source=current.get,
		)
		with exporter:
			for frame in source.frames():
				current.value = transforms_map.get(frame.index, Transform2D.identity())
				exporter.emit_frame(frame.index / fps, states_by_frame.get(frame.index, []), frame)

	typer.echo(
		f"Rendered {trj} over {video} -> {out} "
		f"({len(recording.frames)} timesteps, trail={trail or 'full'})."
	)


if __name__ == "__main__":
	app()
