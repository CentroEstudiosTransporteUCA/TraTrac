"""Typer entry point for FiftyOne dataset export (``tratrac-fiftyone``).

Post-hoc visualization: builds a FiftyOne video dataset from already-produced outputs — the
raw track record (``infrastructure/tracks/parquet.py``) and/or a smoothed SSAM ``.trj``
(``infrastructure/export/ssam_trj.py``) — so both can be inspected/compared in the FiftyOne
App. A reader, not a pipeline stage: nothing here runs detection or tracking; it reads what
``tratrac``/``tratrac-postprocess`` already wrote, the same "post-hoc tool over existing
outputs" shape as ``scripts/plot_run.py`` and ``scripts/validate_trj.py``. See
``docs/roadmap/mvp7.md``'s "Exploration pass" for the design this follows.

``fiftyone`` is an optional extra (``uv sync --extra fiftyone``), not a core dependency — see
``pyproject.toml``'s comment on ``[project.optional-dependencies]`` for why. This module is
only importable when it's installed.

**Verified end-to-end against real footage** (``cruce.mp4``/``out/cruce.parquet``/
``out/cruce.trj`` — a real 27,319-frame dataset with both ``record_detections`` and
``trj_detections`` populated correctly). The conversion logic below
(``record_frame_detections``, ``trj_frame_detections``) is pure — no ``fiftyone`` import — and
is separately unit-tested without a live FiftyOne/MongoDB instance.

**On Linux, ``fiftyone``'s bundled MongoDB doesn't exist** — ``fiftyone-db`` stopped publishing
Linux wheels after version 0.4.5; every version since (including whatever this resolves to)
ships macOS/Windows binaries only, and
``fiftyone.core.service.DatabaseService.find_mongod()`` never falls back to a system ``mongod``
on ``PATH``. Set ``FIFTYONE_DATABASE_URI`` to point at a ``mongod`` you run yourself (any
Linux-native install, or MongoDB's official static binary run through the same raw-ELF-loader
trick this project's NixOS sessions use for other generic-glibc binaries, e.g. ``ruff`` —
``curl``/``openssl``'s shared libs from ``nixpkgs`` cover its only two missing deps). See
``CLAUDE.md`` Dependency Notes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tratrac.domain.geometry import oriented_box_to_aabb
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.ssam_trj import TrjRecording, read_trj
from tratrac.infrastructure.tracks.parquet import TrackRecording, read_tracks
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

if TYPE_CHECKING:
	import fiftyone as fo

app = typer.Typer(
	name="tratrac-fiftyone",
	help="Build a FiftyOne dataset from a track record and/or a smoothed .trj.",
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
	``trj_frame_detections`` for the OBB-aware smoothed path.
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


def trj_frame_detections(trj: TrjRecording, fps: float) -> dict[int, list[FrameDetection]]:
	"""Bucket a smoothed ``.trj``'s vehicle states onto video frames, as normalized boxes.

	Frames are aligned by ``round(timestamp * fps)``, the same convention ``cli_render.py``
	uses (the ``.trj`` carries time but not fps). Each state's oriented heading/dimensions
	are collapsed to an enclosing AABB via ``oriented_box_to_aabb`` (Group A4/A5) —
	FiftyOne's plain ``Detection`` box is axis-aligned; the true rotated footprint isn't
	drawn here.
	"""
	width, height = float(trj.width), float(trj.height)
	by_frame: dict[int, list[FrameDetection]] = defaultdict(list)
	for frame in trj.frames:
		frame_index = round(frame.timestamp_seconds * fps)
		for state in frame.states:
			by_frame[frame_index].append(_trj_state_to_detection(state, width, height))
	return dict(by_frame)


def _trj_state_to_detection(state: VehicleState, width: float, height: float) -> FrameDetection:
	angle = math.atan2(state.heading.dy, state.heading.dx)
	# oriented_box_to_aabb's (w, h) convention is (length, width) — the axis that rotates
	# with `angle` is the vehicle's forward/length axis (matches track_smoothing.py's
	# `oriented_size` unpacking: `size_length, size_width = oriented_size`).
	box = oriented_box_to_aabb(
		state.centroid.x, state.centroid.y, state.dimensions.length, state.dimensions.width, angle
	)
	return FrameDetection(
		track_id=state.vehicle_id,
		x=box.x / width,
		y=box.y / height,
		width=box.width / width,
		height=box.height / height,
		label="vehicle",
	)


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
	trj: Annotated[
		Path | None,
		typer.Option("--trj", exists=True, dir_okay=False, help="Smoothed SSAM .trj."),
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
	"""Build DATASET_NAME in FiftyOne from --record and/or --trj (at least one required).

	Passing both adds two label fields per frame (``record_detections``, ``trj_detections``)
	so the raw and smoothed trajectories can be compared directly in the FiftyOne App.
	"""
	if record is None and trj is None:
		raise typer.BadParameter("Pass at least one of --record or --trj.")

	record_by_frame: dict[int, list[FrameDetection]] = {}
	trj_by_frame: dict[int, list[FrameDetection]] = {}
	fps: float | None = None
	try:
		if record is not None:
			recording = read_tracks(record)
			fps = recording.metadata.fps
			record_by_frame = record_frame_detections(recording)
		if trj is not None:
			trj_recording = read_trj(trj)
			if fps is None:
				# .trj carries time but not fps; probe it from the video itself, same as a
				# bare --trj cli_render.py invocation would need.
				fps = _probe_fps(video)
			trj_by_frame = trj_frame_detections(trj_recording, fps)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	_build_dataset(
		video=video,
		dataset_name=dataset_name,
		record_by_frame=record_by_frame,
		trj_by_frame=trj_by_frame,
		overwrite=overwrite,
		launch=launch,
	)


def _probe_fps(video: Path) -> float:
	with OpenCvVideoSource(video) as source:
		return source.metadata.fps


def _build_dataset(
	*,
	video: Path,
	dataset_name: str,
	record_by_frame: dict[int, list[FrameDetection]],
	trj_by_frame: dict[int, list[FrameDetection]],
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
	all_frames = sorted(set(record_by_frame) | set(trj_by_frame))
	for frame_index in all_frames:
		frame = fo.Frame()
		if frame_index in record_by_frame:
			frame["record_detections"] = fo.Detections(
				detections=[_to_fo_detection(d) for d in record_by_frame[frame_index]]
			)
		if frame_index in trj_by_frame:
			frame["trj_detections"] = fo.Detections(
				detections=[_to_fo_detection(d) for d in trj_by_frame[frame_index]]
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
