# Pipeline Stage Composition: Mandatory vs. Optional, Main-Line vs. Post

This is the single source of truth for **which pipeline stages always run, which are toggles,
and — for the toggles — whether they have to ride along inside `tratrac` itself or can be
deferred to a separately-invoked post-processing tool.** No other doc in the repo attempts this
full classification; individual design docs (linked throughout) cover *why* a specific stage
works the way it does, but this is the only place the whole pipeline's composition is laid out
together. If a new stage is added anywhere in the pipeline, it belongs in this doc too.

## The operative rule

"Optional" here means **not inside `tratrac` (the run) itself**. A stage that's unconditional
*once you're inside* `tratrac-postprocess` or `tratrac-render` is still optional in this
classification, because invoking those tools at all is a separate, optional choice — `tratrac`
alone already produces a complete, valid track record. This is a direct consequence of the
project's B-first / post-hoc design bias (see `src/tratrac/domain/ARCHITECTURE.md`): world
projection, exclusion zones, and rendering were each deliberately pulled *out* of the live loop
over this project's history specifically because they didn't need to be there.

---

## 1. Always run — the mandatory spine

Nothing skips these. They execute unconditionally the moment `tratrac` runs.

| Stage | Why it can't be optional |
| --- | --- |
| Video decode | Nothing downstream exists without frames |
| Detection | Can't track or do anything without finding vehicles first |
| Tracking (BoT-SORT) | Without it there are no trajectories, only per-frame detections |
| Track recording (Parquet) | This *is* `tratrac`'s entire output — the perception result |

---

## 2. Optional — main-line

Toggleable, but when turned on they execute **inside** `tratrac`'s live per-frame loop, not
after. Two different reasons force this, worth telling apart:

**Forced by a correctness/data dependency** — a later live step needs this stage's output:

| Stage | Config | Why it's forced live | Detail |
| --- | --- | --- | --- |
| Ego-motion stabilization | `ego_motion.enabled` | Tracking associates on stabilized coordinates — stabilization has to precede it in the same pass | `src/tratrac/infrastructure/video/EGO_MOTION.md` |
| Decode-time decimation | `input.process_fps` | Frames are skipped *before decode* (`cv2.grab()` with no decode) — a skipped frame's data never exists to defer | `src/tratrac/infrastructure/TIMESTEP_PRECISION.md` |

**Forced by a measurement dependency** — the thing being captured only exists at the moment it happens, nothing to reconstruct:

| Stage | Config | Why it's forced live | Detail |
| --- | --- | --- | --- |
| Step timing profiling | `run.timing_csv` | Per-frame wall-clock latency can't be retroactively known — it's observability, not correctness, but still only capturable live | `src/tratrac/infrastructure/timing/STEP_TIMING.md` |
| Transform sidecar recording | `export.transform_csv` | Captures the ego-motion transform at the instant it's computed; reconstructing it after the fact means re-running the estimator, not post-processing a record | `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md` |
| Anchor export | `export.anchors_dir` | The ORB keyframe anchors are chosen live during stabilization | `src/tratrac/application/EXCLUSION_ZONES.md` (consumes the anchors this produces) |

---

## 3. Optional — post

Nothing here runs unless `tratrac-postprocess` or `tratrac-render` is separately invoked.

**Genuinely optional** — a complete, valid `.trj` exists with or without these:

| Stage | Why it's safely post-hoc | Detail |
| --- | --- | --- |
| Exclusion zone filtering | Pure point-in-polygon over recorded centroids | `src/tratrac/application/EXCLUSION_ZONES.md` |
| World projection (single-homography today; multi-homography is a drop-in extension of the same seam) | A pure coordinate map over already-recorded measurements, needs no pixels | `src/tratrac/application/WORLD_PROJECTION.md`, `docs/roadmap/mvp3.md` |
| Export-time (TIMESTEP) decimation | Thins an otherwise-complete `.trj` | `src/tratrac/infrastructure/TIMESTEP_PRECISION.md` |
| Rendering (+ `--violations`, `--transforms`) | Fully derivable from an already-finished `.trj` and a re-opened video file | `src/tratrac/infrastructure/export/VIDEO_EXPORT.md` |
| Link ID / Lane ID assignment (not yet built) | Point-in-polygon over recorded centroids, same pattern as exclusion zones; a `.trj` without them just carries `0` | `docs/roadmap/road_topology.md`, `docs/roadmap/mvp3.md`, `docs/roadmap/mvp6.md` |

**Recommended placement for two not-yet-built stages, not the currently-drafted one:**

| Stage | Recommended placement | Why |
| --- | --- | --- |
| Segmentation (SAM 3, MVP4) | Post — re-open the video against recorded box/frame-index data, the same way rendering already does | The current draft pipeline diagram in `docs/roadmap/mvp4.md` shows it live before tracking; that's not settled, and every other stage this shape (world projection, exclusion, rendering) ended up post-hoc once someone checked whether it needed to be live. Nothing forces this one to be live either. |
| ReID (DINOv3, MVP5) | Post — an offline track-stitcher: re-open the video, embed each track fragment, merge fragments using appearance + a motion-plausibility gate against the Kalman state | The assumed path (plug into BoT-SORT's live appearance slot) works and is lower-effort, but forecloses re-tunability without re-detection and non-causal matching (using both sides of an occlusion gap, the way RTS smoothing already does for kinematics) — advantages a live slot can't offer. See `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md` and `docs/roadmap/mvp5.md`. |

**Mandatory, but not inside `tratrac` — a distinct case from everything else above:**

| Stage | Why it's here, not in Group 1 | Why it's *not* like the rest of Group 3 | Detail |
| --- | --- | --- | --- |
| Smoothing (Kalman/RTS) | Lives entirely in `tratrac-postprocess`, a separately-invoked tool | Unlike exclusion zones, world projection, or rendering, there's no flag to skip it and still get a `.trj` — it's the mechanism that reconstructs kinematics from raw positions | `src/tratrac/application/SMOOTHING.md` |
| `.trj` export (SSAM serialization) | Same — lives in `tratrac-postprocess` | Always runs immediately after smoothing, on every invocation; no path smooths without exporting or exports without smoothing having already run | `src/tratrac/infrastructure/export/SSAM_FORMAT.md` |

These two are the mandatory second and third stages of a two-stage-mandatory pipeline
(`tratrac` → `tratrac-postprocess`), not optional add-ons that happen to live in post-processing.
"Not inside the run" and "skippable" are independent properties — these satisfy the first and
not the second.
