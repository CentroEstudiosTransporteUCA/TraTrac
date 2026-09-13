"""Typer entry point for the post-process pass (``tratrac-postprocess``).

Reads the perception run's track record (Parquet), optionally **filters** out whole tracks
that fall inside exclusion zones, runs the forward+RTS constant-acceleration Kalman smoother
per surviving track **once**, and writes it to either or both of two independent, optional
outputs: ``--out`` (a de-jittered SSAM ``.trj``) and/or ``--smoothed-record`` (a de-jittered
Parquet record, always image-space pixels regardless of world projection — see
``application/SMOOTHING.md``'s "Dual-space export" section). At least one is required. Offline
and zero-phase — the second pass of the two-pass design (src/tratrac/application/SMOOTHING.md). Re-running with
different ``--pos-noise``/``--jerk``/``--exclusion-*`` re-tunes with no re-detection.

Exclusion is **track-aware** (src/tratrac/application/EXCLUSION_ZONES.md): a track is dropped when the majority of its
observations fall inside a zone. Zones are authored on the anchor PNGs a
``tratrac-preprocess`` run exported (``--anchors-dir``); pass that same run's
transforms file via ``--transforms`` so each zone's ``reference_frame`` is
mapped into the global frame by that frame's pose. This tool never fits a
homography or resolves scale itself — both come from ``--transforms``'
projection-stage rows, written by ``tratrac-preprocess estimate``/``project``
(MVP2, src/tratrac/application/WORLD_PROJECTION.md). Every observation is
projected through the file's ``TransformTable``, per its own
``(point, frame_index)``, **before** smoothing — one code path regardless of
whether the rows are ``scale`` (one zone, or several for something like a
fisheye lens' radially-varying GSD) or ``homography``: the table always
resolves the right row for that point, so there is nothing left here to branch
on. The ``.trj`` always carries ``DIMENSIONS.Scale = 1.0``. Whether a track's
oriented (OBB) size/heading survives the projection is likewise never decided
by row kind here — it's ``TransformFunction.preserves_shape``, a capability
each function declares about itself (true for identity/scale, false for a
general homography perspective distortion), queried on whichever function
actually answered that observation's lookup. ``--smoothed-record`` stays
image-space either way — not yet supported when the projection isn't
invertible (more than one zone matches a given frame — see "Dual-space
export" above for why).

With ``--reid-merge`` (Group D2, ``docs/IMPLEMENTATION_PLAN.md``) a pre-computed ReID merge
decision (``application/reid_merge.py``) remaps ``track_id`` on the record **first**, before
anything else track-lifetime-aware runs — fragments the tracker split across an occlusion are
stitched into one continuous track, which the existing per-track smoothing then handles with
no new code (it already groups purely by ``track_id``).

Rendering is a separate step: ``tratrac-render`` on the ``.trj`` (src/tratrac/infrastructure/export/VIDEO_EXPORT.md).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, cast

import typer

from tratrac.application.coordinate_transforms import (
	TransformTable,
	TranslationTransform,
	compose,
	compose_invertible,
	local_scale_at,
)
from tratrac.application.exclusion import excluded_track_ids, to_global_polygons
from tratrac.application.road_graph import (
	lane_id_for_point,
	link_id_for_point,
	to_global_lane_polygons,
	to_global_link_polygons,
)
from tratrac.application.track_smoothing import TrackSample, invert_state_to_image, smooth_to_states
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Point2D, Polygon, oriented_extent
from tratrac.domain.ports import (
	CoordinateTransform,
	InvertibleCoordinateTransform,
	TrajectoryExporter,
)
from tratrac.domain.road_graph import LaneZones, LinkZones
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.exclusion.json import load_exclusion_zones
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.reid.json import load_reid_merge
from tratrac.infrastructure.road_graph.json import load_lane_zones, load_link_zones
from tratrac.infrastructure.tracks.footprint_parquet import read_footprints
from tratrac.infrastructure.tracks.parquet import TrackObservation, TrackRecording, read_tracks
from tratrac.infrastructure.tracks.smoothed_parquet import (
	SmoothedObservation,
	SmoothedTrackParquetSink,
)
from tratrac.infrastructure.transform.records import HomographyFunction, ScaleFunction
from tratrac.infrastructure.transform.sink import read_ego_motion, read_transform_table

app = typer.Typer(
	name="tratrac-postprocess",
	help="Filter + smooth a track record into a de-jittered SSAM .trj (forward+RTS Kalman).",
	no_args_is_help=True,
)


@app.command()
def postprocess(
	tracks: Annotated[
		Path, typer.Argument(exists=True, dir_okay=False, help="Track record (export.out).")
	],
	transforms: Annotated[
		Path,
		typer.Option(
			"--transforms",
			exists=True,
			dir_okay=False,
			help="The tratrac-preprocess run's transforms file (estimate's --out, optionally "
			"updated by project). Maps zones/correspondences authored on any frame into the "
			"global frame by that frame's pose, and supplies the scale-or-homography rows "
			"trajectories are projected/scaled with — always required.",
		),
	],
	out: Annotated[
		Path | None,
		typer.Option(
			"--out",
			"-o",
			dir_okay=False,
			help="Output .trj path. Optional -- pass --smoothed-record instead/as well for a "
			"pixel-space output. At least one of the two is required.",
		),
	] = None,
	smoothed_record: Annotated[
		Path | None,
		typer.Option(
			"--smoothed-record",
			dir_okay=False,
			help="Output path for the de-jittered record in image-space pixels (Parquet) -- "
			"the same smoothing pass as --out, just always in raw pixels regardless of any "
			"world projection, for tools that overlay on the raw video (e.g. tratrac-fiftyone). "
			"See src/tratrac/application/SMOOTHING.md's 'Dual-space export' section. Not yet "
			"supported when --transforms' homography rows aren't invertible (a multi-zone "
			"per-plane projection).",
		),
	] = None,
	reid_merge: Annotated[
		Path | None,
		typer.Option(
			"--reid-merge",
			exists=True,
			dir_okay=False,
			help="A resolved ReID merge decision (application/reid_merge.py's resolve_merges, "
			"see infrastructure/reid/json.py); remaps track_id on the record before anything "
			"else track-lifetime-aware runs, stitching occlusion-split fragments into one track.",
		),
	] = None,
	exclusion_zones: Annotated[
		Path | None,
		typer.Option(
			"--exclusion-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space ROI polygons; tracks mostly inside are dropped.",
		),
	] = None,
	link_zones: Annotated[
		Path | None,
		typer.Option(
			"--link-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space Link ID polygons (src/tratrac/domain/road_graph.py); "
			"surviving observations are classified per-frame and stamped onto VehicleState.link_id.",
		),
	] = None,
	lane_zones: Annotated[
		Path | None,
		typer.Option(
			"--lane-zones",
			exists=True,
			dir_okay=False,
			help="Sidecar JSON of image-space Lane ID polygons (src/tratrac/domain/road_graph.py); "
			"surviving observations are classified per-frame and stamped onto VehicleState.lane_id.",
		),
	] = None,
	footprint: Annotated[
		Path | None,
		typer.Option(
			"--footprint",
			exists=True,
			dir_okay=False,
			help="Segmentation footprint sidecar (Group D1, infrastructure/tracks/footprint_parquet.py "
			"-- a polygon per (track_id, frame_index)); replaces bbox/OBB-derived Dimensions with "
			"mask-derived ones before smoothing, for observations it covers.",
		),
	] = None,
	exclusion_min_fraction: Annotated[
		float,
		typer.Option(
			"--exclusion-min-fraction",
			help="Drop a track when this fraction of its observations are inside a zone, (0, 1].",
		),
	] = 0.5,
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
	"""Filter (optional) + smooth TRACKS into --out (.trj) and/or --smoothed-record (Parquet)."""
	if out is None and smoothed_record is None:
		raise typer.BadParameter("Pass at least one of --out or --smoothed-record.")
	if out is not None and out.exists() and not force:
		raise typer.BadParameter(f"{out} already exists; pass --force to overwrite.")
	if smoothed_record is not None and smoothed_record.exists() and not force:
		raise typer.BadParameter(f"{smoothed_record} already exists; pass --force to overwrite.")
	if timestep_precision < 0.0:
		raise typer.BadParameter("--timestep-precision must be >= 0 (0 = every frame).")
	if not 0.0 < exclusion_min_fraction <= 1.0:
		raise typer.BadParameter("--exclusion-min-fraction must be in (0, 1].")
	try:
		recording = read_tracks(tracks)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	try:
		projection = read_transform_table(transforms, kinds=(ScaleFunction, HomographyFunction))
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc
	kinds_present = projection.kinds()
	if HomographyFunction not in kinds_present and ScaleFunction not in kinds_present:
		raise typer.BadParameter(
			f"{transforms} has no scale or homography rows -- run tratrac-preprocess estimate "
			"(and project, for world coordinates) first."
		)
	if smoothed_record is not None and not projection.is_invertible:
		raise typer.BadParameter(
			"--smoothed-record isn't supported with a non-invertible multi-zone projection "
			"(more than one zone matches a frame, e.g. a per-plane homography or several "
			"scale zones) -- see src/tratrac/application/SMOOTHING.md's 'Dual-space export' "
			"section."
		)

	footprint_count = 0
	if footprint is not None:
		recording, footprint_count = _apply_footprint(recording, footprint)

	merged_count = 0
	if reid_merge is not None:
		recording, merged_count = _apply_reid_merge(recording, reid_merge)

	dropped = 0
	if exclusion_zones is not None:
		recording, dropped = _filter_excluded(
			recording, exclusion_zones, transforms, exclusion_min_fraction
		)

	link_ids: dict[tuple[int, int], int] = {}
	if link_zones is not None:
		link_ids = _assign_labels(
			recording,
			link_zones,
			transforms,
			load_zones=load_link_zones,
			to_global_polygons=to_global_link_polygons,
			label_for_point=link_id_for_point,
		)

	lane_ids: dict[tuple[int, int], int] = {}
	if lane_zones is not None:
		lane_ids = _assign_labels(
			recording,
			lane_zones,
			transforms,
			load_zones=load_lane_zones,
			to_global_polygons=to_global_lane_polygons,
			label_for_point=lane_id_for_point,
		)

	# Every observation is projected through the file's own TransformTable, per its own
	# (point, frame_index) -- never summarized into one representative value first. This
	# is the same call regardless of whether the table holds scale rows (one zone or
	# several, e.g. a fisheye lens' radially-varying GSD) or homography rows: the table
	# itself already resolves the right row for each point, so there is nothing left for
	# this function to branch on.
	recording, pos_noise, jerk, projector = _project_to_world(
		recording, projection, pos_noise, jerk
	)
	scale_transform = ScaleFunction(1.0)

	states_by_frame = _smooth_recording(recording, scale_transform, pos_noise=pos_noise, jerk=jerk)
	if link_ids:
		states_by_frame = _apply_link_ids(states_by_frame, link_ids)
	if lane_ids:
		states_by_frame = _apply_lane_ids(states_by_frame, lane_ids)

	if smoothed_record is not None:
		smoothed_record.parent.mkdir(parents=True, exist_ok=True)
		_emit_smoothed_record(
			smoothed_record, recording.metadata, states_by_frame, projector=projector
		)

	if out is not None:
		out.parent.mkdir(parents=True, exist_ok=True)
		_emit_trj(out, recording, states_by_frame, timestep_precision=timestep_precision)

	notes = []
	if footprint is not None:
		notes.append(f"replaced dimensions for {footprint_count} observations from footprints")
	if reid_merge is not None:
		notes.append(f"merged {merged_count} ReID track ids")
	if exclusion_zones is not None:
		notes.append(f"dropped {dropped} excluded tracks")
	if link_zones is not None:
		notes.append(f"assigned link ids to {len(link_ids)} observations")
	if lane_zones is not None:
		notes.append(f"assigned lane ids to {len(lane_ids)} observations")
	if HomographyFunction in kinds_present:
		notes.append("projected to world coordinates")
	note = f", {', '.join(notes)}" if notes else ""
	outputs = ", ".join(str(p) for p in (out, smoothed_record) if p is not None)
	typer.echo(
		f"Post-processed {tracks} -> {outputs} "
		f"({len(states_by_frame)} frames, pos_noise={pos_noise:g}, jerk={jerk:g}{note})."
	)


def _apply_footprint(recording: TrackRecording, footprint_path: Path) -> tuple[TrackRecording, int]:
	"""Replace bbox/OBB-derived dimensions with footprint-derived ones (Group D1) by writing
	into ``obb_w``/``obb_h`` — the same slot Group A's OBB detector already occupies, so every
	downstream consumer (``build_state`` preferring ``oriented_size`` over the bbox,
	``_project_observation`` correctly dropping it post-projection) treats a footprint-derived
	size exactly like an OBB one, with no new special-casing.

	Runs **before** ``--reid-merge`` (keyed by the *original* ``track_id`` the segmentation run
	saw — the same one a footprint sidecar would be authored against — not any later remapped
	canonical id) so a merged track's footprint coverage still matches by the time it's looked
	up. An observation absent from the footprint sidecar is left unchanged.
	"""
	try:
		observations = read_footprints(footprint_path)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc
	polygon_by_key: dict[tuple[int, int], Polygon] = {
		(f.track_id, f.frame_index): f.polygon for f in observations
	}
	if not polygon_by_key:
		return recording, 0

	updated: list[TrackObservation] = []
	replaced = 0
	for o in recording.observations:
		polygon = polygon_by_key.get((o.track_id, o.frame_index))
		if polygon is None:
			updated.append(o)
			continue
		length, width = oriented_extent(polygon, o.angle)
		updated.append(replace(o, obb_w=length, obb_h=width))
		replaced += 1
	return (
		TrackRecording(metadata=recording.metadata, observations=updated),
		replaced,
	)


def _apply_reid_merge(
	recording: TrackRecording, reid_merge_path: Path
) -> tuple[TrackRecording, int]:
	"""Remap ``track_id`` on every observation through a resolved ReID merge decision.

	Runs before exclusion filtering / Link-Lane-Plane assignment / world projection (Group D2's
	composition-root position, ``docs/IMPLEMENTATION_PLAN.md``): those stages are all
	track-lifetime-aware, so occlusion-split fragments must already be one track_id by the time
	they run, or e.g. exclusion's majority vote would see two short, separately-judged tracks
	instead of the vehicle's whole life. A track id absent from the mapping is left unchanged
	(``load_reid_merge`` only returns entries for track ids that were actually merged).
	"""
	try:
		merges = load_reid_merge(reid_merge_path)
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc
	if not merges:
		return recording, 0
	remapped = [
		replace(o, track_id=merges.get(o.track_id, o.track_id)) for o in recording.observations
	]
	return TrackRecording(metadata=recording.metadata, observations=remapped), len(merges)


def _filter_excluded(
	recording: TrackRecording, zones_path: Path, transforms_path: Path, min_fraction: float
) -> tuple[TrackRecording, int]:
	"""Drop whole tracks mostly inside an exclusion zone; return the survivors + drop count."""
	try:
		zones = load_exclusion_zones(zones_path)
		polygons = to_global_polygons(zones, _pose_for(transforms_path))
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	excluded = excluded_track_ids(
		((o.track_id, Point2D(o.cx, o.cy)) for o in recording.observations),
		polygons,
		min_fraction=min_fraction,
	)
	if not excluded:
		return recording, 0
	kept = [o for o in recording.observations if o.track_id not in excluded]
	filtered = TrackRecording(metadata=recording.metadata, observations=kept)
	return filtered, len(excluded)


_GlobalLabeledPolygons = tuple[tuple[int, tuple[Point2D, ...]], ...]


def _assign_labels[Zones: (LinkZones, LaneZones)](
	recording: TrackRecording,
	zones_path: Path,
	transforms_path: Path,
	*,
	load_zones: Callable[[Path], Zones],
	to_global_polygons: Callable[[Zones, CoordinateTransform], _GlobalLabeledPolygons],
	label_for_point: Callable[[Point2D, _GlobalLabeledPolygons], int],
) -> dict[tuple[int, int], int]:
	"""Classify each surviving observation into a label (Link ID or Lane ID), keyed by
	``(track_id, frame_index)``.

	Shared by ``--link-zones`` and ``--lane-zones`` (src/tratrac/domain/ARCHITECTURE.md /
	docs/IMPLEMENTATION_PLAN.md Groups C1/C2): per-observation, not per-track, since a vehicle
	can cross links/lanes mid-track and SSAM's Link ID / Lane ID are per-VEHICLE-RECORD fields.
	Runs on image-space coordinates (before any world projection), the same stage
	exclusion filtering already runs at.
	"""
	try:
		zones = load_zones(zones_path)
		polygons = to_global_polygons(zones, _pose_for(transforms_path))
	except (ValueError, OSError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	return {
		(o.track_id, o.frame_index): label_for_point(Point2D(o.cx, o.cy), polygons)
		for o in recording.observations
	}


def _apply_link_ids(
	states_by_frame: dict[int, list[VehicleState]], link_ids: dict[tuple[int, int], int]
) -> dict[int, list[VehicleState]]:
	"""Stamp each smoothed state's ``link_id`` from the pre-smoothing per-observation assignment."""
	return {
		frame_index: [
			replace(state, link_id=link_ids.get((state.vehicle_id, frame_index), 0))
			for state in states
		]
		for frame_index, states in states_by_frame.items()
	}


def _apply_lane_ids(
	states_by_frame: dict[int, list[VehicleState]], lane_ids: dict[tuple[int, int], int]
) -> dict[int, list[VehicleState]]:
	"""Stamp each smoothed state's ``lane_id`` from the pre-smoothing per-observation assignment."""
	return {
		frame_index: [
			replace(state, lane_id=lane_ids.get((state.vehicle_id, frame_index), 0))
			for state in states
		]
		for frame_index, states in states_by_frame.items()
	}


def _pose_for(transforms_path: Path) -> CoordinateTransform:
	"""Resolve a zone's/correspondence's reference-frame pose (raw -> global) from a
	``tratrac-preprocess`` run's transforms file, as a ``CoordinateTransform`` callers apply
	with ``.apply(point, reference_frame)``.

	Each pose comes straight from the file's dense ego-motion-stage table — any
	``reference_frame`` within the clip resolves, not just an anchor frame; a
	``reference_frame`` outside the table (e.g. past the clip's end) raises
	``KeyError``, which callers here re-raise as a clean ``typer.BadParameter``. A
	malformed file raises ``ValueError``, caught by callers alongside their own
	file-loading errors. A static run's file simply has no ego-motion rows, so
	every pose is the identity (global == raw) automatically.
	"""
	return read_ego_motion(transforms_path)


def _project_to_world(
	recording: TrackRecording,
	projection: TransformTable,
	pos_noise: float,
	jerk: float,
) -> tuple[TrackRecording, float, float, CoordinateTransform]:
	"""Project the recording's image coordinates onto the metric world plane (MVP2).

	``projection`` is the already-fitted homography table ``tratrac-preprocess project``
	wrote (see ``application/coordinate_transforms.py``'s ``TransformTable``) -- this
	function only *applies* it, never fits anything. Rewrites every observation's
	centroid and bbox into world metres and stamps ``scale = 1.0`` so the downstream
	smoother/exporter emit world coordinates with ``DIMENSIONS.Scale = 1.0``.

	The projected coordinates are translated to a non-negative origin and the recording's
	``width``/``height`` are replaced with the **world** extent (metres) — not the pixel
	grid. This matters because the SSAM exporter writes those as the DIMENSIONS bounds and
	flips Y about ``height x scale``: leaving the pixel dimensions in place would make an
	external reader see metric coordinates against a pixel-sized canvas, flipped about the
	wrong axis (src/tratrac/infrastructure/export/SSAM_FORMAT.md + src/tratrac/application/WORLD_PROJECTION.md). The translation discards the operator's
	absolute world origin, which is arbitrary in Approach A and irrelevant to the
	translation-invariant conflict analytics. ``pos_noise``/``jerk`` are converted from
	pixels into world units by a representative local scale near the recording's own
	observations (there's no calibration-correspondence centroid to sample here anymore --
	fitting moved to ``tratrac-preprocess project`` -- so this samples the recording itself).

	The returned ``CoordinateTransform`` is the **shifted** one (homography + 0-origin
	translation composed via ``compose_invertible``/``TranslationTransform``) — the one
	whose ``.reverse()`` (when ``projection.is_invertible``; a multi-zone per-plane
	projection isn't, see ``TransformTable.is_invertible``) correctly undoes everything
	this function did to a point, not just the homography. Used by ``--smoothed-record``
	(``application/SMOOTHING.md``'s "Dual-space export" section), never by the forward
	path above, which already applied the unshifted ``projection`` + a separate
	observation-level shift and stays untouched.
	"""
	try:
		projected = [_project_observation(o, projection) for o in recording.observations]
	except KeyError as exc:
		raise typer.BadParameter(
			f"an observation's frame_index is not a frame in the transforms file: {exc}"
		) from exc
	world_recording, shift_x, shift_y = _normalize_world_recording(recording.metadata, projected)
	shift = TranslationTransform(shift_x, shift_y)
	shifted_projector: CoordinateTransform = (
		compose_invertible(projection, shift)
		if projection.is_invertible
		else compose(projection, shift)
	)
	scale = _representative_local_scale(recording, projection)
	return world_recording, pos_noise * scale, jerk * scale * scale, shifted_projector


def _representative_local_scale(
	recording: TrackRecording, projection: CoordinateTransform
) -> float:
	"""A single metres-per-pixel scale representative of the whole recording, for converting
	``--pos-noise``/``--jerk`` (tuned in pixels) into the projection's world units.

	Averages each observation's *own* local scale (sampled exactly at its own recorded
	position and frame) rather than averaging raw coordinates into one synthetic point
	first: for a multi-zone (per-plane) projection, a mean point computed across
	observations from disjoint zones can land in the gap between them, matching no zone
	at all -- every real observation, by construction, sits inside whichever zone it
	actually occupies. One global value still feeds the smoother for the whole recording
	(it isn't per-observation) — same averaging caveat ``tratrac-preprocess project``
	already accepts when a calibration spans multiple anchors/planes.
	"""
	observations = recording.observations
	if not observations:
		return 1.0
	scales = [local_scale_at(projection, Point2D(o.cx, o.cy), o.frame_index) for o in observations]
	return sum(scales) / len(scales)


def _normalize_world_recording(
	metadata: VideoMetadata, projected: list[TrackObservation]
) -> tuple[TrackRecording, float, float]:
	"""Shift projected observations to a 0-origin and size the metadata to the world extent.

	Pads the bounding box by the largest projected vehicle dimension so the front/rear
	bumpers the smoother derives from each centroid stay inside the DIMENSIONS bounds. An
	empty recording keeps the original (pixel) metadata — there are no coordinates to size.

	Also returns the ``(shift_x, shift_y)`` applied (``cx - min_x`` etc. above, so the shift
	itself is ``-min_x, -min_y``) — composed into an invertible projector by the caller
	(``_project_to_world``) so ``--smoothed-record`` can undo this translation along with the
	homography itself, not just the homography.
	"""
	if not projected:
		return TrackRecording(metadata=metadata, observations=projected), 0.0, 0.0
	pad = max(max(o.width, o.height) for o in projected)
	min_x = min(o.cx for o in projected) - pad
	min_y = min(o.cy for o in projected) - pad
	max_x = max(o.cx for o in projected) + pad
	max_y = max(o.cy for o in projected) + pad
	shifted = [replace(o, cx=o.cx - min_x, cy=o.cy - min_y) for o in projected]
	world_meta = replace(
		metadata,
		width=max(1, math.ceil(max_x - min_x)),
		height=max(1, math.ceil(max_y - min_y)),
	)
	return TrackRecording(metadata=world_meta, observations=shifted), -min_x, -min_y


def _project_observation(o: TrackObservation, projector: TransformTable) -> TrackObservation:
	"""Map one observation's centroid + bbox extent into world metres.

	Looks up the zone/function **once**, at the observation's own centroid, and applies
	that same function to all five points (centre + 4 extent corners) -- not one
	independent lookup per point. A corner near a zone boundary would otherwise risk
	picking a *different* zone than the box's own position (an inconsistent projection
	across one observation), or simply fall outside a narrow zone's polygon and raise
	"matches no zone" for a box that only pokes past the canvas edge, even though the
	observation itself is squarely inside it.

	``angle``/``obb_w``/``obb_h`` (Group A7/A8 OBB fields) survive only when the
	*function itself* declares it preserves shape (``TransformFunction.preserves_shape``
	— true for identity/scale, false for a general homography): a homography's
	perspective distortion can shear/rotate differently across the image, so carrying a
	raw image-space angle into world-space heading selection would be wrong rather than
	merely imprecise, and re-deriving a world-space OBB angle is out of scope here (see
	`src/tratrac/application/WORLD_PROJECTION.md`) — the smoother's low-speed fallback
	drops back to the bbox-major-axis heuristic for a projected run instead. A
	shape-preserving function is locally isotropic, so its own scale factor (the ratio
	between the projected and original bbox width) rescales the OBB dimensions exactly;
	the angle itself is unchanged.
	"""
	function = projector.function_at_point(Point2D(o.cx, o.cy), o.frame_index)
	center = function.apply(Point2D(o.cx, o.cy))
	left = function.apply(Point2D(o.cx - o.width / 2.0, o.cy))
	right = function.apply(Point2D(o.cx + o.width / 2.0, o.cy))
	top = function.apply(Point2D(o.cx, o.cy - o.height / 2.0))
	bottom = function.apply(Point2D(o.cx, o.cy + o.height / 2.0))
	width = math.hypot(right.x - left.x, right.y - left.y)
	height = math.hypot(bottom.x - top.x, bottom.y - top.y)
	angle, obb_w, obb_h = None, None, None
	if function.preserves_shape:
		# Angle and OBB dimensions are independent fields (a footprint-derived size can
		# arrive with no angle at all) -- carry each through on its own terms rather than
		# requiring both to be present.
		angle = o.angle
		if o.obb_w is not None or o.obb_h is not None:
			local_scale = width / o.width if o.width > 0.0 else height / o.height
			obb_w = o.obb_w * local_scale if o.obb_w is not None else None
			obb_h = o.obb_h * local_scale if o.obb_h is not None else None
	return replace(
		o,
		cx=center.x,
		cy=center.y,
		width=width,
		height=height,
		angle=angle,
		obb_w=obb_w,
		obb_h=obb_h,
	)


def _smooth_recording(
	recording: TrackRecording, scale: ScaleFunction, *, pos_noise: float, jerk: float
) -> dict[int, list[VehicleState]]:
	"""Group observations by track, smooth each, and regroup the states by frame index."""
	by_track: dict[int, list[TrackObservation]] = defaultdict(list)
	for observation in recording.observations:
		by_track[observation.track_id].append(observation)

	from tqdm import tqdm

	fps = recording.metadata.fps
	states_by_frame: dict[int, list[VehicleState]] = defaultdict(list)
	for track_id, observations in tqdm(by_track.items(), desc="Smoothing", unit="track"):
		observations.sort(key=lambda o: o.frame_index)
		samples = [
			TrackSample(
				frame_index=o.frame_index,
				timestamp_seconds=o.frame_index / fps,
				center=Point2D(o.cx, o.cy),
				width=o.width,
				height=o.height,
				angle=o.angle,
				oriented_size=(o.obb_w, o.obb_h)
				if o.obb_w is not None and o.obb_h is not None
				else None,
			)
			for o in observations
		]
		states = smooth_to_states(track_id, samples, scale, pos_noise=pos_noise, jerk=jerk)
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
	"""Write the smoothed .trj from the per-frame states, optionally decimating TIMESTEPs.

	``scale=1.0`` always: every observation was already projected into final output
	units by ``_project_to_world`` before smoothing, regardless of whether the
	transforms file held scale or homography rows.
	"""
	fps = recording.metadata.fps
	exporter: TrajectoryExporter = SsamTrjExporter(out, recording.metadata, scale=1.0)
	if timestep_precision > 0.0:
		# Thin the exported TIMESTEP stream; the smoothing still uses every observation.
		exporter = DecimatingTrajectoryExporter(
			exporter, min_interval_seconds=timestep_precision, fps=fps
		)
	with exporter:
		for frame_index in sorted(states_by_frame):
			exporter.emit_frame(frame_index / fps, states_by_frame[frame_index])


def _emit_smoothed_record(
	path: Path,
	metadata: VideoMetadata,
	states_by_frame: dict[int, list[VehicleState]],
	*,
	projector: CoordinateTransform,
) -> None:
	"""Write the smoothed record always in image-space pixels (see ``smoothed_parquet.py``).

	``projector`` is the shifted, invertible projector ``_project_to_world`` returns —
	never a non-invertible multi-zone one (rejected earlier, in ``postprocess``, when
	combined with ``--smoothed-record``, so the cast below is a validated internal
	invariant, not a user-facing error).
	"""
	with SmoothedTrackParquetSink(path, metadata) as sink:
		for frame_index in sorted(states_by_frame):
			for state in states_by_frame[frame_index]:
				invertible = cast(InvertibleCoordinateTransform, projector)
				centroid, angle, dimensions = invert_state_to_image(state, invertible, frame_index)
				sink.record(
					SmoothedObservation(
						frame_index=frame_index,
						track_id=state.vehicle_id,
						cx=centroid.x,
						cy=centroid.y,
						angle=angle,
						length=dimensions.length,
						width=dimensions.width,
						link_id=state.link_id,
						lane_id=state.lane_id,
					)
				)


if __name__ == "__main__":
	app()
