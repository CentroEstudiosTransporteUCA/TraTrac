# Implementation Plan: Full Feature List + Full Pipeline

## Context

TraTrac's tech stack was re-derived from a citation-backed research pass that overturned three
of its original picks (RT-DETR → YOLO-OBB, FastReID → DINOv3, SAM2 → SAM3) and confirmed others
(BoT-SORT, the Kalman/RTS smoother, single-homography geometry). That research is fully written
up in `docs/TECH_STACK.md` and cascaded into `DETECTOR_CHOICE.md`, `TRACKER_CHOICE.md`,
`SMOOTHING.md`, `WORLD_PROJECTION.md`, `BACKLOG.md`, and the `mvp4`/`mvp5` roadmap docs.
`docs/PIPELINE_STAGES.md` separately classified every pipeline stage as always-run,
optional-but-live, or optional-and-deferrable-to-post-processing — which pipeline stage a new
capability belongs in is already decided there, not open in this plan.

What didn't exist until now was a plan that takes `docs/PRODUCTION_MVP.md`'s value-ordered
feature checklist and resequences it by **dependency** — what actually blocks what, what's
independent and can run in parallel, and where shared infrastructure needs to be built once
instead of three times. That's what this document is.

This plan was built via a verified read-only pass over the actual current source (not just the
docs describing it) followed by a design pass — both are reflected below. Six groups (A–F),
ordered by dependency, not value. Groups B, C, and D are each internally parallelizable with
Group A; only specific tasks within a group are truly sequential.

---

## Group A — Detection quality & orientation (do first, sequential)

Everything else that wants a per-detection orientation angle depends on this landing.

| # | Task | Files |
|---|---|---|
| A0 | Restore `scripts/probe_detector.py` — referenced throughout `DETECTOR_CHOICE.md` and `PRODUCTION_MVP.md`'s "Known blocker," missing from the repo. Blocks any credible before/after detector comparison. | `scripts/probe_detector.py` (new) |
| A1 | Swap `[tool.uv.sources]` to a CUDA torch/torchvision index (same index for both — mismatched indices break `torchvision::nms` registration, per `CLAUDE.md`). | `pyproject.toml` |
| A2 | Acquire UAV-OBB (Mendeley Data), fine-tune `yolo11-obb`/`yolo26-obb`. No training harness exists yet — this is new design + code. Acceptance criterion (already stated in `DETECTOR_CHOICE.md`): must measurably outperform the YOLOv8 baseline via A0 + `scripts/validate_trj.py`. | `scripts/train_yolo_obb.py` (new) |
| A3 | Domain: add `Detection.angle: float \| None` (radians) and `Detection.oriented_size: tuple[float, float] \| None`. **Keep `bbox` as the AABB** — still needed for ORB's vehicle-masking and IoU-based tracking paths; don't conflate it with the rotated box. | `domain/detection.py` |
| A4 | New adapter wrapping `ultralytics`'s OBB task (`.obb`, not `.boxes`), mapping UAV-OBB's six classes into `VehicleClass`, populating `angle`/`oriented_size` alongside the AABB. | `infrastructure/detection/yolo_obb.py` (new) |
| A5 | **Real integration work, not optional:** `boxmot` has separate OBB support with a different det-array layout and result-row parsing than the current AABB path (`_detections_to_array`, `row[:4]`/`det_ind`). Needs a constructor flag or auto-detection based on whether `angle` is populated. | `infrastructure/tracking/boxmot_bot_sort.py` |
| A6 | Parquet schema: nullable `angle`, `obb_w`, `obb_h` columns. Recommend backward-compatible tolerate-missing-columns reads over a hard version bump — old records shouldn't become unreadable. | `infrastructure/tracks/parquet.py` |
| A7 | `TrackSample` gains `angle`/`oriented_size`; `build_state` gets the new heading-selection logic (below); use `oriented_size` for `Dimensions` when present — this is the real accuracy payoff OBB buys for vehicle sizing, not just heading. | `application/track_smoothing.py` |
| A8 | Thread the new fields through `TrackObservation → TrackSample` construction. | `cli_postprocess.py` |
| A9 | Wire `yolo_obb` into `DetectorChoice`; decide the new default; decide `DetectorConfig.filename`'s fate (already flagged open in `DETECTOR_CHOICE.md`). | `application/config.py`, `cli.py` |
| A10 | Remove the YOLOv8 emergency adapter — mechanical, **last**, only once A2–A9 are proven: delete file, drop enum value + CLI branch, `uv remove dill`, update tests/`CLAUDE.md`/`tratrac.example.toml`. | `infrastructure/detection/yolov8_visdrone.py` (delete), `application/config.py`, `cli.py` |

**Licensing checkpoint (not a code task):** staying on `ultralytics`/`boxmot` for OBB deepens the AGPL-3.0 exposure `CLAUDE.md` already flags — needs an explicit business decision before external distribution, not resolved by this plan.

### Orientation integration mechanism

Current `build_state` derives heading only from smoothed Kalman velocity, falling back to a
180°-ambiguous bbox-major-axis guess at low speed — exactly what the validator catches flipping.
Replace **only the low-speed fallback branch**, not the primary path:

```
if speed >= _VELOCITY_EPSILON:
    heading = velocity.normalized()          # unchanged — trust the RTS-smoothed signal
elif sample.angle is not None:
    candidate = Heading.from_angle(sample.angle)
    heading = candidate if last_heading is None or dot(candidate, last_heading) >= 0 else candidate.reversed()
else:
    heading = last_heading or _major_axis_heading(width, height)   # unchanged, last resort
```

Why not always blend: the RTS-smoothed velocity direction is already the product of the two-pass
Kalman design — trusting a raw, per-frame, unsmoothed OBB angle over it while moving risks
reintroducing the jitter the smoother exists to remove. The fallback branch is exactly where the
OBB angle earns its keep, replacing a 180°-ambiguous guess with a real detected orientation,
disambiguated against track continuity.

### Tests
No existing pattern for either detector adapter or the tracker adapter — `test_yolo_obb.py` and
`test_boxmot_bot_sort.py` are new files establishing the pattern. `test_track_smoothing.py` and
`test_tracks.py` extend cleanly.

---

## Group B — Validation & trigger-gated swaps (parallel with A)

| # | Task |
|---|---|
| B1 | Real-footage validation pass #1, current YOLOv8 baseline — satisfies `PRODUCTION_MVP.md`'s "validated against real congress-site footage" line now. |
| B2 | Repeat after A2–A9 land — this *is* A2's stated acceptance criterion, not a separate deliverable. |
| B3 | SuperPoint+LightGlue ego-motion adapter (`BACKLOG.md` #1). **Trigger-gated, not fired.** If triggered: extract `_AnchorChain` out of `ego_motion_orb.py` into a shared `infrastructure/video/anchor_chain.py` first (it's pure, ORB-agnostic), then a new adapter implementing all three ports `OrbEgoMotionEstimator` implements today (`EgoMotionEstimator`, `DetectionObserver`, `StabilizationTransformSource`), reusing the vehicle-masking contract. `EgoMotionConfig` needs restructuring (currently ORB-parameter-shaped only). |
| B4 | TorchCodec(+NVDEC) decode adapter (`BACKLOG.md` #3). **Trigger-gated, not fired.** If triggered: must replicate **both** `DecimationGrid` call sites in `OpenCvVideoSource` — the streaming skip in `frames()` and the separate `_processed_count` precomputation for `VideoMetadata.total_frames` — not just one. |

**Recommendation:** hold B3/B4 until B1/B2's real-footage validation actually shows ORB or decode is the bottleneck, rather than building opportunistically against an untriggered gate.

---

## Group C — Geometry (parallel with A/B; three independent tasks, not a chain)

**C1 — Link ID assignment.** Cheapest task in the plan; no dependency on C3/C4/C5.
- `domain/road_graph.py` (new): `LinkZone(link_id, reference_frame, polygon)`, mirrors `ExclusionZone`'s shape but carries a label.
- `application/road_graph.py` (new): `to_global_link_polygons` (mirrors `to_global_polygons`) + `assign_link_ids`. **Cannot reuse `excluded_track_ids` directly** — it does a track-level majority-vote drop/keep; this needs classification into one of N labeled zones, recommended **per-frame** (a vehicle can cross links mid-track; SSAM's Link ID is a per-VEHICLE-record field, which supports this).
- `infrastructure/road_graph/json.py` (new): sidecar loader, mirrors `infrastructure/exclusion/json.py`.
- `cli_postprocess.py`: new `--link-zones` option; thread the assigned id into `VehicleState.link_id` (currently always 0 — `build_state` doesn't touch it today).
- Test: `test_road_graph.py` (new), modeled on `test_exclusion.py` + `test_geometry.py`.

**C2 — Lane ID assignment (MVP6).** Hard-depends on C1 only. Same module, lane-strip polygons `{link_id, lane_id, polygon}`, populates `VehicleState.lane_id`.

**C3 — Multi-anchor world projection (MVP2 remainder).** Independent of C1/C2/C4/C5.
- `application/world_projection.py`: new `PerAnchorWorldProjector` implementing `WorldProjector.to_world`, using `frame_index` (already threaded through the port for exactly this) to pick the right anchor's homography.
- Fitter groups correspondences by anchor, fits one `H` per anchor. `Correspondence` already carries `reference_frame` — no domain change needed.
- Open question: nearest-anchor switch vs. interpolating between neighboring anchors' homographies — undecided in `WORLD_PROJECTION.md`, resolve when this task starts.

**C4 — Automatic road-geometry calibration.** Independent of everything else in this group; a workflow upgrade to Approach A's fitting, not a prerequisite for C3/C5.
- New module producing **proposed** correspondences from visible lane markings/road borders into the existing `calibration.json` schema — not a fully automatic replacement (manual validation still outperforms automatic, per the cited source).
- Open question: does TraTrac ship only the correspondence-proposal capability, with interactive confirm/adjust living in URBAn (`URBAn/docs/PRODUCTION_MVP.md` already flags this UX gap there)? Needs an explicit repo-boundary decision before this task starts.

**C5 — Multi-homography + plane assignment (MVP3).** The largest item in this group.
- Plane polygons live in the **same** `application/road_graph.py` module as Link/Lane (`mvp3.md`: they share infrastructure but read different polygon sets).
- New `MultiHomographyWorldProjector` (one `H` per plane, selected by per-observation plane assignment, not by `frame_index`).
- Plane assignment must run before projection in `cli_postprocess.py` — see the integration-order note below.

---

## Group D — Perception enrichment (parallel with C)

**Shared infra, build once before D1/D2 need it:** extract the windowed-reopen + frame-bucketing
pattern already proven in `cli_render.py`/`overlay_video.py` into `infrastructure/replay/track_frames.py`
(`iter_track_frames(video_path, recording, padding_seconds=...) -> Iterator[(Frame, observations)]`),
so segmentation and ReID-embed call it instead of each reimplementing the windowed
`OpenCvVideoSource` open + `round(timestamp*fps)` bucketing + transform remap. Test:
`test_replay.py`, modeled on `test_overlay_video.py` + `tests/integration/test_render.py`.

**D1 — Segmentation (SAM 3, MVP4 remainder).**
- **Scope gate, not a code blocker:** MVP4's narrowed "footprint only" scope is only valid once Group A's OBB adapter has landed. Start in earnest after A2–A4.
- New `cli_segment.py` (`tratrac-segment`), shaped like `cli_render.py`: re-opens via the shared replay helper, runs SAM 3 per track-frame crop (box- or text-prompted from the OBB), extracts the occupancy contour, writes `infrastructure/tracks/footprint_parquet.py` sidecar.
- `cli_postprocess.py` gains optional `--footprint sidecar.parquet`, replacing bbox/OBB-derived dimensions with mask-derived ones before smoothing.
- Test: `test_footprint.py` (new), SAM 3 kept behind an injected seam so unit tests never import it — same discipline `overlay_video.py` already uses for `draw`/`open_writer`.

**D2 — ReID (DINOv3, MVP5).** The largest net-new algorithm in the whole plan. Three stages, split expensive/cacheable from cheap/re-tunable — the same split `SMOOTHING.md`'s two-pass design already uses:

1. **Embed** (`cli_embed.py`, `tratrac-embed`) — the only stage that reopens the video. Via the shared replay helper, samples representative frames per track fragment (proposed default: first/middle/last visible frame — validate against real footage before trusting it), crops (using the OBB angle for an upright crop when available — soft payoff of Group A, not a hard dependency), runs DINOv3, writes `embeddings.parquet`. GPU-bound, cacheable, never rerun just to retune a threshold.
2. **Merge decision** (`application/reid_merge.py`) — pure, no video/GPU, fully re-tunable offline like `--pos-noise`/`--jerk`. For fragment pairs with a plausible temporal gap: gate with `application.kalman.KinematicKalmanFilter` (already kept in the codebase for exactly this — "the primitive for a future streaming/RT path") extrapolating fragment *i*'s end state to fragment *j*'s start time; among gated candidates, score by DINOv3 cosine similarity, resolve 1:1 within each time window. Output `reid_merge.json` (`{old_track_id: canonical_track_id}` + audit scores).
3. **Apply** (`cli_postprocess.py`, new `--reid-merge`) — remap `TrackObservation.track_id` on the recording. This is the **entire integration point**: `_smooth_recording` already groups purely by `track_id`, so merged fragments are automatically smoothed as one continuous track by the existing forward+RTS pass — no new smoothing code needed. Apply **before** exclusion filtering (so track-aware exclusion's majority vote considers the whole reidentified vehicle's lifetime) and before world projection/smoothing.

**Documented fallback, not the primary route** (per `PIPELINE_STAGES.md`'s already-settled post-hoc reasoning — lower effort but forecloses re-tunability and non-causal matching): `boxmot.trackers.BotSort` genuinely accepts `reid_model=<prebuilt DINOv3>, with_reid=True` — confirmed available in `boxmot_bot_sort.py` if the offline stitcher proves harder than expected.

Tests: `test_reid_merge.py` (pure motion-gate + merge-resolution logic, no video/GPU) + an embed-CLI test mirroring `test_render.py`'s seam-injection approach.

---

## Group E — Opportunistic research (KalmanNet family)

No file-level plan possible yet — `SMOOTHING.md` is explicit this needs its own design pass, no
dataset plan exists. Concrete next step, when picked up: a research/design spike producing
`application/KALMANNET_DESIGN.md` before any code. No phase gate; doesn't block or get blocked by
anything above.

## Group F — Platform (FiftyOne, async, Docker+CUDA)

Not explored at the same depth as A–E in this pass — flagging honestly rather than inventing
file-level detail for infrastructure that's naturally last and decoupled from everything above.
Needs its own exploration/design pass before it's execution-ready at the same level of detail.

---

## Composition-root integration order (matters once multiple groups land)

Groups A (via richer `TrackSample`), C (plane/link/lane assignment, multi-anchor projection), and
D (ReID-merge) all add steps to `cli_postprocess.postprocess`. Target final order so parallel work
doesn't collide:

```
read record
  → apply --reid-merge (D2)         [track_id remap — before anything track-lifetime-aware]
  → filter --exclusion-zones        [already shipped]
  → assign plane / link / lane      (C1 / C5)   [needed before projection knows which H to use]
  → project --calibration           [already shipped; C3/C5 extend the projector itself]
  → smooth (Kalman/RTS)             [already shipped, unchanged]
  → export .trj
```

---

## Open questions (resolve at the point each task starts, not blocking this plan)

1. Does the fine-tuned YOLO-OBB checkpoint report a full 0–360° heading or only 0–180°? Unknown until trained (A2).
2. Should per-frame OBB angle be smoothed (circular EMA) across a track? Ship raw first, add smoothing only if `validate_trj.py` shows jitter on real footage (A7).
3. Parquet schema evolution: confirmed recommendation is backward-compatible tolerate-missing-columns, not a hard version bump (A6) — needs sign-off since it's a compatibility policy, not a technicality.
4. `DetectorConfig.filename`: drop or repurpose for the OBB checkpoint path (A9) — already flagged unresolved in `DETECTOR_CHOICE.md`.
5. `rt_detr.py`: keep dormant behind `Detector`, or delete as dead code (A10/A11) — already flagged unresolved in `DETECTOR_CHOICE.md`.
6. B3/B4: build opportunistically or strictly hold for their measured triggers? Recommendation is hold.
7. C3: nearest-anchor switch vs. interpolation between neighboring anchors' homographies — undecided in `WORLD_PROJECTION.md`.
8. C4: TraTrac ships only correspondence-proposal, interactive confirm/adjust lives in URBAn? Needs an explicit repo-boundary decision.
9. D2: DINOv3 crop-sampling policy and exact merge-scoring function — proposed defaults given, both need validation against real occlusion footage before the thresholds mean anything.
10. C1: confirm per-frame Link ID classification (proposed, new precedent) over per-track majority vote (exclusion's existing pattern).

---

## Test-file map

| Area | Extends existing pattern | New file (no pattern exists yet) |
|---|---|---|
| YOLO-OBB adapter | — | `test_yolo_obb.py` |
| Tracker OBB path | — | `test_boxmot_bot_sort.py` |
| Orientation/dims in smoother | `test_track_smoothing.py` | — |
| Parquet schema | `test_tracks.py` | — |
| SuperPoint+LightGlue | `test_ego_motion_orb.py` (close model) | `test_ego_motion_superpoint.py` |
| TorchCodec decode | `test_cadence.py`, `test_video_window.py` | `test_video_torchcodec.py` |
| Link/Lane ID | `test_exclusion.py`, `test_geometry.py` (close model) | `test_road_graph.py` |
| Multi-anchor/homography projection | `test_world_projection.py`, `test_world_calibration.py` | — |
| Replay helper (Group D shared infra) | `test_overlay_video.py`, `test_render.py` (close model) | `test_replay.py` |
| Footprint sidecar | `test_tracks.py` (close model) | `test_footprint.py` |
| ReID merge logic | — | `test_reid_merge.py` |
| `cli_postprocess.py` new sidecars | `test_cli_postprocess.py` | — |

---

## Verification

Every task inherits the repo's existing gates — no new verification philosophy needed:
- `uv run ruff format . && uv run ruff check . && uv run mypy` before any change is considered done (per `CLAUDE.md`).
- `uv run pytest` (fast) for the unit-test areas in the map above; `uv run pytest -m slow` for anything touching the detector checkpoint download path (Group A).
- `scripts/validate_trj.py` is the semantic acceptance gate for anything affecting trajectory quality (A2's baseline-beating criterion, D2's occlusion-recovery validation, C5's plane-assignment sanity check) — compare compliance percentages before/after, not just "it runs."
- `scripts/probe_detector.py` (once A0 restores it) is the detector-quality acceptance gate specifically.
- Real-footage validation (B1/B2) is the end-to-end check that ties everything together — run the full `tratrac` → `tratrac-postprocess` → `tratrac-render` chain on actual site footage after each group lands, not just synthetic/test clips.

## Critical files (most referenced across groups)

- `src/tratrac/domain/detection.py` — orientation fields, root of Group A's schema propagation
- `src/tratrac/application/track_smoothing.py` — heading-selection mechanism, the real orientation payoff
- `src/tratrac/infrastructure/tracks/parquet.py` — schema evolution decision point
- `src/tratrac/cli_postprocess.py` — shared composition root every group but A eventually touches
- `src/tratrac/infrastructure/tracking/boxmot_bot_sort.py` — undocumented OBB-awareness work Group A requires
- `src/tratrac/application/exclusion.py` / `src/tratrac/domain/exclusion.py` — the pattern Group C's Link/Lane ID mirrors but cannot directly reuse
- `src/tratrac/infrastructure/export/overlay_video.py` / `src/tratrac/cli_render.py` — the pattern Group D's shared replay helper is extracted from
