"""Typer entry point for FiftyOne dataset export (``tratrac-fiftyone``).

Post-hoc visualization: builds a FiftyOne video dataset from already-produced outputs — the
raw track record (``infrastructure/tracks/parquet.py``) and/or a smoothed record
(``infrastructure/tracks/smoothed_parquet.py``, ``tratrac-postprocess --smoothed-record``) — so
both can be inspected/compared in the FiftyOne App. A reader, not a pipeline stage: nothing
here runs detection or tracking; it reads what ``tratrac``/``tratrac-postprocess`` already
wrote, the same "post-hoc tool over existing outputs" shape as ``scripts/plot_run.py`` and
``scripts/validate_trj.py``. See ``FIFTYONE.md`` for the full design.

**Not the SSAM ``.trj``, deliberately.** An earlier version of this module read ``--trj``
directly and normalized its coordinates as if they were always raw video pixels. That's true
only for an uncalibrated ``.trj`` — a calibrated one (``tratrac-postprocess --calibration``,
MVP2) carries real-world metres from a homography the ``.trj`` file itself doesn't store, and
dividing metres by the video's pixel width/height silently produced boxes clustered near the
origin, disconnected from where the vehicle actually was. Confirmed visually in the FiftyOne
App, not just in theory. ``--smoothed-record`` exists precisely to give this tool something
that's always genuinely pixel-space, whether or not the run was calibrated — see
``application/SMOOTHING.md``'s "Dual-space export" section for the full mechanism (one Kalman/
RTS smoothing pass, inverted back through the same projector for this output, not a second
smoothing pass in pixel space).

``fiftyone`` is an optional extra (``uv sync --extra fiftyone``), not a core dependency — see
``pyproject.toml``'s comment on ``[project.optional-dependencies]`` for why. This module is
only importable when it's installed.

**Verified end-to-end against real footage** (``cruce.mp4``/``out/cruce.parquet`` — a real
27,319-frame dataset with ``record_detections`` populated correctly; re-verify
``smoothed_detections`` the same way once a ``--smoothed-record`` output exists for real
footage). The conversion logic below (``record_frame_detections``,
``smoothed_frame_detections``) is pure — no ``fiftyone`` import — and is separately
unit-tested without a live FiftyOne/MongoDB instance.

**On Linux, ``fiftyone``'s bundled MongoDB doesn't exist** — ``fiftyone-db`` stopped publishing
Linux wheels after version 0.4.5; every version since (including whatever this resolves to)
ships macOS/Windows binaries only, and
``fiftyone.core.service.DatabaseService.find_mongod()`` never falls back to a system ``mongod``
on ``PATH``. Set ``FIFTYONE_DATABASE_URI`` to point at a ``mongod`` you run yourself (any
Linux-native install, or MongoDB's official static binary run through the same raw-ELF-loader
trick this project's NixOS sessions use for other generic-glibc binaries, e.g. ``ruff`` —
``curl``/``openssl``'s shared libs from ``nixpkgs`` cover its only two missing deps). See
``CLAUDE.md`` Dependency Notes. ``scripts/run_fiftyone.sh`` automates this.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tratrac.domain.geometry import oriented_box_to_aabb
from tratrac.infrastructure.tracks.parquet import TrackRecording, read_tracks
from tratrac.infrastructure.tracks.smoothed_parquet import SmoothedRecording, read_smoothed_tracks

if TYPE_CHECKING:
	import fiftyone as fo

app = typer.Typer(
	name="tratrac-fiftyone",
	help="Build a FiftyOne dataset from a track record and/or a smoothed record.",
	no_args_is_help=True,
)


@dataclass(frozen=True, slots=True)
class FrameDetection:
	"""One vehicle's detection on one frame, as a normalized FiftyOne-shaped box.

	Pure value object — no ``fiftyone`` import — so the conversion functions below are
	unit-testable without a live FiftyOne/MongoDB instance (see the module docstring).
	"""

	track_id: int
	# FiftyOne's [x, y, width, height] convention: top-left origin, each normalized to
	# [0, 1] by the frame's own width/height.
	x: float
	y: float
	width: float
	height: float
	label: str
	confidence: float | None = None


def record_frame_detections(recording: TrackRecording) -> dict[int, list[FrameDetection]]:
	"""Group a track record's raw observations by frame, as normalized boxes.

	Always axis-aligned (the record's ``w``/``h`` are AABB dimensions even when the
	detector also reported an OBB angle) — this is the raw-bbox path; see
	``smoothed_frame_detections`` for the OBB-aware smoothed path.
	"""
	width, height = float(recording.metadata.width), float(recording.metadata.height)
	by_frame: dict[int, list[FrameDetection]] = defaultdict(list)
	for obs in recording.observations:
		by_frame[obs.frame_index].append(
			FrameDetection(
				track_id=obs.track_id,
				x=(obs.cx - obs.width / 2.0) / width,
				y=(obs.cy - obs.height / 2.0) / height,
				width=obs.width / width,
				height=obs.height / height,
				label=obs.vehicle_class.value,
				confidence=obs.score,
			)
		)
	return dict(by_frame)


def smoothed_frame_detections(recording: SmoothedRecording) -> dict[int, list[FrameDetection]]:
	"""Group a smoothed record's observations by frame, as normalized boxes.

	Always image-space pixels by construction (``application/SMOOTHING.md``'s "Dual-space
	export" section) regardless of whether the run producing it used ``--calibration`` — unlike
	the SSAM ``.trj``, there's no coordinate-space ambiguity to guard against here. Each
	observation's ``(cx, cy, angle, length, width)`` are collapsed to an enclosing AABB via
	``oriented_box_to_aabb`` — FiftyOne's plain ``Detection`` box is axis-aligned; the true
	rotated footprint isn't drawn here.
	"""
	width, height = float(recording.metadata.width), float(recording.metadata.height)
	by_frame: dict[int, list[FrameDetection]] = defaultdict(list)
	for obs in recording.observations:
		# oriented_box_to_aabb's (w, h) convention is (length, width) — the axis that rotates
		# with `angle` is the vehicle's forward/length axis (matches track_smoothing.py's
		# `oriented_size` unpacking: `size_length, size_width = oriented_size`).
		box = oriented_box_to_aabb(obs.cx, obs.cy, obs.length, obs.width, obs.angle)
		by_frame[obs.frame_index].append(
			FrameDetection(
				track_id=obs.track_id,
				x=box.x / width,
				y=box.y / height,
				width=box.width / width,
				height=box.height / height,
				label="vehicle",
			)
		)
	return dict(by_frame)


@app.command()
def build(
	video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source clip.")],
	dataset_name: Annotated[
		str, typer.Option("--dataset-name", help="FiftyOne dataset name to create.")
	],
	record: Annotated[
		Path | None,
		typer.Option("--record", exists=True, dir_okay=False, help="Raw track record (Parquet)."),
	] = None,
	smoothed_record: Annotated[
		Path | None,
		typer.Option(
			"--smoothed-record",
			exists=True,
			dir_okay=False,
			help="Smoothed record (Parquet, tratrac-postprocess --smoothed-record). Always "
			"image-space, unlike a --calibration'd .trj -- see the module docstring.",
		),
	] = None,
	overwrite: Annotated[
		bool,
		typer.Option(
			"--overwrite/--no-overwrite", help="Replace an existing dataset of the same name."
		),
	] = False,
	launch: Annotated[
		bool, typer.Option("--launch/--no-launch", help="Open the FiftyOne App after building.")
	] = False,
) -> None:
	"""Build DATASET_NAME in FiftyOne from --record and/or --smoothed-record (>= 1 required).

	Passing both adds two label fields per frame (``record_detections``,
	``smoothed_detections``) so the raw and smoothed trajectories can be compared directly in
	the FiftyOne App.
	"""
	if record is None and smoothed_record is None:
		raise typer.BadParameter("Pass at least one of --record or --smoothed-record.")

	record_by_frame: dict[int, list[FrameDetection]] = {}
	smoothed_by_frame: dict[int, list[FrameDetection]] = {}
	try:
		if record is not None:
			record_by_frame = record_frame_detections(read_tracks(record))
		if smoothed_record is not None:
			smoothed_by_frame = smoothed_frame_detections(read_smoothed_tracks(smoothed_record))
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	_build_dataset(
		video=video,
		dataset_name=dataset_name,
		record_by_frame=record_by_frame,
		smoothed_by_frame=smoothed_by_frame,
		overwrite=overwrite,
		launch=launch,
	)


def _build_dataset(
	*,
	video: Path,
	dataset_name: str,
	record_by_frame: dict[int, list[FrameDetection]],
	smoothed_by_frame: dict[int, list[FrameDetection]],
	overwrite: bool,
	launch: bool,
) -> None:
	"""Write the converted frame detections into a real FiftyOne dataset.

	Isolated from the pure conversion functions above so only this one function needs a
	live ``fiftyone`` (and its MongoDB backing store) to exercise — see the module
	docstring for why that isn't verified in this environment.
	"""
	import fiftyone as fo

	if fo.dataset_exists(dataset_name):
		if not overwrite:
			raise typer.BadParameter(
				f'Dataset "{dataset_name}" already exists; pass --overwrite to replace it.'
			)
		fo.delete_dataset(dataset_name)

	dataset = fo.Dataset(name=dataset_name, persistent=True)
	sample = fo.Sample(filepath=str(video))
	all_frames = sorted(set(record_by_frame) | set(smoothed_by_frame))
	for frame_index in all_frames:
		frame = fo.Frame()
		if frame_index in record_by_frame:
			frame["record_detections"] = fo.Detections(
				detections=[_to_fo_detection(d) for d in record_by_frame[frame_index]]
			)
		if frame_index in smoothed_by_frame:
			frame["smoothed_detections"] = fo.Detections(
				detections=[_to_fo_detection(d) for d in smoothed_by_frame[frame_index]]
			)
		# FiftyOne video frame numbers are 1-indexed; our frame_index is 0-indexed.
		sample.frames[frame_index + 1] = frame
	dataset.add_sample(sample)

	typer.echo(f'Built FiftyOne dataset "{dataset_name}" ({len(all_frames)} frames).')
	if launch:
		session = fo.launch_app(dataset)
		session.wait()


def _to_fo_detection(d: FrameDetection) -> fo.Detection:
	import fiftyone as fo

	kwargs: dict[str, Any] = {"index": d.track_id}
	if d.confidence is not None:
		kwargs["confidence"] = d.confidence
	return fo.Detection(label=d.label, bounding_box=[d.x, d.y, d.width, d.height], **kwargs)


if __name__ == "__main__":
	app()
