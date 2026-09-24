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
| Ego-motion stabilization | `input.transforms_in` (whether the file has any similarity rows — no separate enable toggle) | Tracking associates on stabilized coordinates — stabilization has to precede it in the same pass | `src/tratrac/infrastructure/video/EGO_MOTION.md` |
| Decode-time decimation | `input.process_fps` | Frames are skipped *before decode* (`cv2.grab()` with no decode) — a skipped frame's data never exists to defer | `src/tratrac/infrastructure/TIMESTEP_PRECISION.md` |

**Forced by a measurement dependency** — the thing being captured only exists at the moment it happens, nothing to reconstruct:

| Stage | Config | Why it's forced live | Detail |
| --- | --- | --- | --- |
| Step timing profiling | `run.timing_csv` | Per-frame wall-clock latency can't be retroactively known — it's observability, not correctness, but still only capturable live | `src/tratrac/infrastructure/timing/STEP_TIMING.md` |

Transform sidecar recording and anchor export used to be optional stages here, but they no
longer exist inside `tratrac` at all: `tratrac` never estimates ego-motion (or scale, or a
world-projection homography) itself, only reads an already-built transforms file
(`input.transforms_in`, an always-required config key — there is no `[ego_motion]` section or
`enabled` toggle for it to gate on). The estimation, transform recording, and anchor export
instead run *mandatorily* (not a toggle) inside the separate `tratrac-preprocess estimate`
subcommand's own live single pass, before `tratrac` runs at all — same "measurement dependency"
reasoning (the transform and the anchor frame's pixels only exist at the instant
`tratrac-preprocess` walks that frame), just relocated to a different tool's mandatory spine.
`tratrac-preprocess` itself is now mandatory for every run, even a fully static camera, since
it's the only place the GSD scale row gets resolved. See
`src/tratrac/infrastructure/video/EGO_MOTION.md` and
`src/tratrac/infrastructure/transform/TRANSFORM_SINK.md`.

---

## 3. Optional — post

Nothing here runs unless `tratrac-postprocess` or `tratrac-render` is separately invoked.

**Genuinely optional** — a complete, valid `.trj` exists with or without these:

| Stage | Why it's safely post-hoc | Detail |
| --- | --- | --- |
| Exclusion zone filtering | Pure point-in-polygon over recorded centroids | `src/tratrac/application/EXCLUSION_ZONES.md` |
| World projection (single- and multi-homography both land as the same `homography` row kind, fitted by `tratrac-preprocess project`, applied by `tratrac-postprocess`) | A pure coordinate map over already-recorded measurements, needs no pixels | `src/tratrac/application/WORLD_PROJECTION.md` |
| Export-time (TIMESTEP) decimation | Thins an otherwise-complete `.trj` | `src/tratrac/infrastructure/TIMESTEP_PRECISION.md` |
| Rendering (+ `--violations`, `--transforms`) | Fully derivable from an already-finished `.trj` and a re-opened video file | `src/tratrac/infrastructure/export/VIDEO_EXPORT.md` |
| Link ID / Lane ID assignment (shipped, `--link-zones`/`--lane-zones`) | Point-in-polygon over recorded centroids, same pattern as exclusion zones; a `.trj` without either flag just carries `0` | `src/tratrac/application/ROAD_GRAPH.md` |

**Placement followed as recommended, for two mostly-built stages:**

| Stage | Placement | Why |
| --- | --- | --- |
| Segmentation (SAM 3, MVP4) | Post — re-opens the video against recorded box/frame-index data, the same way rendering already does. Shipped: the footprint sidecar storage format and `--footprint` dimension-override consumption in `tratrac-postprocess`. Still blocked: the segmentation stage itself (`cli_segment.py`, actually running SAM 3), which needs a GPU. | Every other stage this shape (world projection, exclusion, rendering) ended up post-hoc once someone checked whether it needed to be live; nothing forced this one to be live either. See `src/tratrac/application/FOOTPRINT.md`. |
| ReID (DINOv3, MVP5) | Post — an offline track-stitcher: re-open the video, embed each track fragment, merge fragments using appearance + a motion-plausibility gate against the Kalman state. Shipped and validated against real footage: the merge-decision and apply stages (`application/reid_merge.py`, `--reid-merge`). Still blocked: the DINOv3 embedding stage itself (`cli_embed.py`), which needs a GPU. | The alternative (plug into BoT-SORT's live appearance slot) works and is lower-effort, but forecloses re-tunability without re-detection and non-causal matching (using both sides of an occlusion gap, the way RTS smoothing already does for kinematics) — advantages a live slot can't offer; kept as a documented fallback, not the primary route. See `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md` and `src/tratrac/application/REID_MERGE.md`. |

**Mandatory, but not inside `tratrac` — a distinct case from everything else above:**

| Stage | Why it's here, not in Group 1 | Why it's *not* like the rest of Group 3 | Detail |
| --- | --- | --- | --- |
| Smoothing (Kalman/RTS) | Lives entirely in `tratrac-postprocess`, a separately-invoked tool | Unlike exclusion zones, world projection, or rendering, there's no flag to skip it and still get a `.trj` — it's the mechanism that reconstructs kinematics from raw positions | `src/tratrac/application/SMOOTHING.md` |
| `.trj` export (SSAM serialization) | Same — lives in `tratrac-postprocess` | Always runs immediately after smoothing, on every invocation; no path smooths without exporting or exports without smoothing having already run | `src/tratrac/infrastructure/export/SSAM_FORMAT.md` |

These two are the mandatory second and third stages of a two-stage-mandatory pipeline
(`tratrac` → `tratrac-postprocess`), not optional add-ons that happen to live in post-processing.
"Not inside the run" and "skippable" are independent properties — these satisfy the first and
not the second.
