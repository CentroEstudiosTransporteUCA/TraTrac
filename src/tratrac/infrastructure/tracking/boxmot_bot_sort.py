"""boxmot BoT-SORT adapter (IoU-only for MVP1; ReID arrives in MVP5).

boxmot 19.x exposes tracker classes under ``boxmot.trackers``. The top-level
``Boxmot`` orchestrator is intentionally avoided — we want fine-grained control
over per-frame updates inside the application pipeline.

Why BoT-SORT, not plain SORT: ``docs/TECH_STACK.md``'s ideal stack names BoT-SORT
deliberately, and this choice was re-checked against current benchmarks (not just carried over
from the original stack pick) — it held up, unlike the detector and ReID picks. ``boxmot``
(already TraTrac's dependency) supports six trackers: BoTSORT, HybridSORT, StrongSORT,
DeepOCSORT, ByteTrack, OCSORT. BoT-SORT ranks highest of all of them on the MOT17 benchmark
(68.9, ahead of HybridSORT 68.2, StrongSORT 68.1, DeepOCSORT 67.8), and it's the only one of
the six with an appearance branch — what identity persistence through occlusion (MVP5) needs;
ByteTrack and OCSORT are motion-only and can't do that job regardless of association quality.
Plain SORT is motion-only too, with poor long-term stability — exactly the failure mode this
project needs to avoid once occlusion recovery matters. One nuance: BoT-SORT's built-in
camera-motion-compensation is a large part of why it's generally recommended for drone footage,
but TraTrac already handles ego-motion itself via the keyframe-anchored ORB stabilizer
(MVP1.9) and disables BoT-SORT's own CMC (``cmc_method=None``) when that's active — so the
value BoT-SORT adds to *this* pipeline specifically is the appearance branch, not CMC.

Current state: shipped IoU-only (``boxmot.trackers.BotSort`` constructed without a ReID model)
— the appearance/ReID half of BoT-SORT isn't wired in here, so today it behaves closer to SORT
in practice. ReID activation is MVP5's job (see ``application/REID_MERGE.md`` — TraTrac's
actual MVP5 design is an offline track-stitcher, not this adapter's live ReID slot, though the
slot exists as a documented fallback); the adapter and the ``Tracker`` port don't need to
change for that, only the model/config BoT-SORT is constructed with.

``boxmot`` is AGPL-3.0 — resolved, not an open risk: TraTrac is GPL-3.0 and GPLv3 §13
explicitly permits the combination (see ``CLAUDE.md`` Dependency Notes).

OBB support (Group A5, GitHub Issues): ``boxmot``'s ``BotSort`` natively tracks oriented boxes
given a 7-column ``(cx, cy, w, h, angle, conf, cls)`` det array instead of the 6-column AABB
one, confirmed against ``boxmot.trackers.detection_layout``. The tracker infers this from the
*first* non-empty det array's column count and locks it in for the run — an empty first frame
would silently lock in AABB mode even for an OBB-capable detector — so ``is_obb`` is an
explicit constructor flag (decided from the run's detector, not sniffed per frame) rather than
relying on that auto-detection. Output rows carry the angle back into
``Detection.angle``/``oriented_size``; ``Detection.bbox`` stays populated as the rotated box's
axis-aligned enclosing rectangle (``_obb_to_aabb``), since ORB masking and other AABB-only
consumers still need it.

ReID for MVP5, DINOv3 not FastReID: ``docs/TECH_STACK.md``'s original stack named FastReID.
That doesn't survive scrutiny here, for a reason more fundamental than "wrong model": the
nadir viewpoint itself discards most of what vehicle-ReID models are trained to discriminate
on — license plates, side profile, grille/tail-light shape are largely invisible from directly
overhead. The literature is explicit about this, and a real ten-UAV, twenty-intersection
drone-traffic deployment (Songdo, South Korea) found pure vision-only ReID insufficient from
the air, resorting to fusing it with a temporal travel-time model built on traffic-flow
shockwave theory. There's also no nadir-matched fine-tuning dataset the way UAV-OBB solved
that problem for detection: the best available aerial vehicle-ReID dataset, VRAI (137K
images), spans 15-80m altitude with hovering/cruising/rotating viewpoint changes — mixed-angle,
not nadir-specific. FastReID's checkpoints (and ``boxmot``'s own ``clip_vehicleid.pt``) are
trained on ground-level/oblique surveillance datasets (VehicleID, VeRi-style) — vehicle-domain,
which beats a generic person-ReID checkpoint, but carrying the same oblique-vs-nadir gap.

What actually generalizes better across this gap: DINOv3. Self-supervised foundation-model
embeddings — not trained on any single fixed viewpoint — transfer across viewpoint shift
measurably better than a supervised ReID checkpoint. A DINOv3-pretrained backbone reaches
88.19 mAP on VeRi-Wild vehicle ReID from visual cues alone (matching the strongest
metadata-dependent baselines), and DINO-family ViTs are the most geometry-aware
general-purpose vision transformers available, retaining mIoU 0.766 under 90° angular
separation — effectively the oblique-to-nadir gap this project faces. Still measurably weaker
under large viewpoint shift than small ones, not immune to the problem, but the strongest
available answer.

Recommended and landed as: DINOv3 embeddings as the appearance signal, combined with
motion-plausibility gating — using the Kalman state ``application/kalman.py`` already
maintains to reject re-identification candidates that aren't a physically plausible
reappearance (right place, right time, right velocity for the elapsed gap). This mirrors the
Songdo study's appearance+temporal fusion in principle, using motion infrastructure TraTrac
already has instead of a separate travel-time model. Cheap color/footprint-shape heuristics
remain a legitimate fallback given how little nadir-view visual signal survives at all. The
motion-plausibility gate half is landed (``application/reid_merge.py``,
``application/REID_MERGE.md``): ``KinematicKalmanFilter.from_state``/``.predict`` extrapolate a
track fragment's end state forward and gate a candidate reappearance by standard deviations of
the extrapolated position, exactly as described above. The DINOv3 appearance-embedding half is
not — the merge-decision logic takes an embedding as an opaque input and is exercised only
against synthetic vectors so far, pending a GPU + real footage to build and validate the embed
stage against.

Sources: BoxMOT tracker list/MOT17 rankings/AABB+OBB support
(https://github.com/mikel-brostrom/boxmot); tracker comparison, MOT benchmark results
(https://trackers.roboflow.com/latest/trackers/comparison/); DINOv3-based vehicle ReID, 88.19
mAP on VeRi-Wild (https://arxiv.org/html/2607.22068); DINOv3 cross-viewpoint robustness under
angular separation (https://www.alphaxiv.org/abs/2508.10104); VRAI dataset altitude/viewpoint
composition (https://arxiv.org/pdf/1904.01400); Songdo drone-traffic ReID + shockwave-theory
fusion (https://www.eurekalert.org/news-releases/1124023); oblique-vs-nadir
vehicle-classification visibility tradeoff (https://doi.org/10.3390/rs17152653).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from boxmot.trackers import BotSort

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox, oriented_box_to_aabb

_VEHICLE_CLASS_TO_COCO_ID: dict[VehicleClass, int] = {
	VehicleClass.CAR: 2,
	VehicleClass.MOTORCYCLE: 3,
	VehicleClass.BUS: 5,
	VehicleClass.TRUCK: 7,
}


class BoxmotBotSortTracker:
	"""Wraps ``boxmot.trackers.BotSort`` behind the ``Tracker`` port."""

	def __init__(
		self,
		metadata: VideoMetadata,
		*,
		det_thresh: float,
		compensate_camera_motion: bool = True,
		is_obb: bool = False,
	) -> None:
		# det_thresh below the detector's threshold so the detector is the sole
		# gatekeeper. Aerial-domain detections often peak in the 0.25-0.4 range,
		# and BotSort's stock 0.3 would suppress legitimate low-confidence cars.
		#
		# When coordinate stabilization is on (see ego_motion_orb.py's module docstring),
		# detections are already mapped into the stabilized frame before they reach the tracker, so
		# BoT-SORT must NOT also compensate camera motion — doing so would double-
		# correct. cmc_method=None disables its internal CMC. Left on for raw
		# (unstabilized) runs so a moving camera is still handled for association.
		cmc_method = "ecc" if compensate_camera_motion else None
		self._is_obb = is_obb
		self._tracker: Any = BotSort(
			reid_model=None,
			with_reid=False,
			frame_rate=round(metadata.fps),
			det_thresh=det_thresh,
			cmc_method=cmc_method,
			is_obb=is_obb,
		)

	def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedDetection]:
		dets_array = self._detections_to_array(detections)
		results = self._tracker.update(dets_array, frame.pixels)

		tracked: list[TrackedDetection] = []
		if results.size == 0:
			return tracked

		for row in results:
			if self._is_obb:
				# OBB layout: cx, cy, w, h, angle, id, conf, cls, det_ind.
				det_ind = int(row[8])
				track_id, score = int(row[5]), float(row[6])
			else:
				# AABB layout: x1, y1, x2, y2, id, conf, cls, det_ind.
				det_ind = int(row[7])
				track_id, score = int(row[4]), float(row[5])
			# det_ind == -1 means the track is being held by motion prediction
			# with no fresh detection this frame. MVP1 skips these — long-term
			# track survival is MVP5 work.
			if det_ind < 0 or det_ind >= len(detections):
				continue
			original = detections[det_ind]
			if self._is_obb:
				cx, cy, w, h, angle = (float(v) for v in row[:5])
				detection = Detection(
					bbox=oriented_box_to_aabb(cx, cy, w, h, angle),
					score=score,
					vehicle_class=original.vehicle_class,
					angle=angle,
					oriented_size=(w, h),
				)
			else:
				x1, y1, x2, y2 = (float(v) for v in row[:4])
				detection = Detection(
					bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
					score=score,
					vehicle_class=original.vehicle_class,
				)
			tracked.append(TrackedDetection(track_id=track_id, detection=detection))
		return tracked

	def _detections_to_array(self, detections: Sequence[Detection]) -> np.ndarray:
		if not detections:
			return np.empty((0, 7 if self._is_obb else 6), dtype=np.float32)
		if self._is_obb:
			return np.array(
				[
					[
						d.bbox.center.x,
						d.bbox.center.y,
						*(
							d.oriented_size
							if d.oriented_size is not None
							else (d.bbox.width, d.bbox.height)
						),
						d.angle if d.angle is not None else 0.0,
						d.score,
						_VEHICLE_CLASS_TO_COCO_ID[d.vehicle_class],
					]
					for d in detections
				],
				dtype=np.float32,
			)
		return np.array(
			[
				[
					d.bbox.x,
					d.bbox.y,
					d.bbox.x + d.bbox.width,
					d.bbox.y + d.bbox.height,
					d.score,
					_VEHICLE_CLASS_TO_COCO_ID[d.vehicle_class],
				]
				for d in detections
			],
			dtype=np.float32,
		)
