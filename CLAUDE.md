# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

**MVP1.75 landed** — metric sizes/speeds from drone-metadata GSD calibration (see `src/tratrac/calibration/GSD_CALIBRATION.md`): `Length`/`Width`/`Speed`/`Acceleration` and `DIMENSIONS.Scale` are now real metric values. **MVP1.9 landed** — ORB + RANSAC similarity ego-motion as an optional step (`ego_motion.enabled`, off by default; see `src/tratrac/infrastructure/video/EGO_MOTION.md`): when on, detection/tracking run on the **raw, full-resolution frame** and the keyframe-anchored ego-motion transform is applied to the **detections** (coordinates, not pixels) before tracking, so trajectories are ego-motion-free and nothing is ever cropped to black. The keyframe anchor re-sets when too little of it stays in view (`ego_motion.min_anchor_overlap`). It is an intermediate "keep if good enough" shortcut before MVP2's learned stabilizer + world projection. **MVP1.5 (aerial-robust detector fine-tune — now a YOLO-OBB plan, not the originally-planned RT-DETR; see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`) was leapfrogged, not yet done** — 1.75/1.9 are independent shortcuts that slot between MVP1 and MVP2. So detection + tracking are still MVP1-grade (YOLOv8-VisDrone emergency detector + IoU-only BoT-SORT, no ReID), and SSAM **positions are image-space pixels by default** — **MVP2 world projection partially landed**: `tratrac-postprocess --calibration` now optionally projects trajectories onto a metric world plane via a **post-hoc single homography** (Approach A, see `src/tratrac/application/WORLD_PROJECTION.md`) fitted from operator `image↔world` correspondences, so SSAM can carry metric world coords with `DIMENSIONS.Scale=1.0`; deferred are the SuperPoint+LightGlue stabilization upgrade (ORB still does ego-motion) and the multi-anchor projector. **Export inverted (perception-only pipeline)** — `tratrac` now writes only the **track record** (raw tracked measurements, `export.out`, the canonical internal "export B"); it no longer computes kinematics or writes a `.trj`. The SSAM `.trj` is produced **post-hoc** by `tratrac-postprocess` (the only `.trj` path; runs the Kalman/RTS smoother and the orientation/kinematics). The orientation subsystem and the live `.trj` exporter were removed from the run. The record is an **Apache Parquet** file (`ParquetTrackSink` / `read_tracks` in `infrastructure/tracks/parquet.py`; run metadata + scale live in the Parquet schema metadata) — this pulls the MVP7 Parquet-storage choice forward for the canonical record. The repo contains:

- `docs/` — the documentation index (`docs/README.md`) plus cross-cutting docs that don't belong to one module: `ROADMAP.md` (system objective + MVP status reconciliation), `ARCHITECTURE.md` (end-state pipeline diagram), `TECH_STACK.md` (stack choices not yet adopted), `BACKLOG.md` ("ship cheaper now, upgrade later" tracker), and `docs/roadmap/` (not-yet-started MVP3–7 + road topology). Everything else lives next to the code it describes — see `docs/README.md` for the full map.
- `pyproject.toml`, `uv.lock`, `.python-version` (3.12) — uv-managed project.
- `src/tratrac/` — the implementation, organized in onion layers:
  - `domain/` — pure value objects (geometry — including `Transform2D`, the affine/similarity transform used for stabilization — frame, detection, vehicle, progress events, step-timing records, per-frame ego-motion transform records in `stabilization.py` — `FrameTransform`) and Protocol ports (`VideoSource`, `EgoMotionEstimator`, `Detector`, `DetectionObserver`, `DetectionStabilizer`, `Tracker`, `TrackSink`, `TrajectoryExporter`, `ProgressReporter`, `TimingSink`, `TransformSink`, `AnchorSink`).
  - `application/` — `TrajectoryPipeline` (perception-only per-frame orchestrator that emits a progress event stream and records each frame's tracked detections to a `TrackSink` — its primary output; when ego-motion is enabled it stabilizes detections via the `DetectionStabilizer` port before tracking — MVP1.9), `track_smoothing.py` (`smooth_to_states`/`build_state` — reconstruct `VehicleState`s with kinematics from raw track samples; the offline `tratrac-postprocess` core, src/tratrac/application/SMOOTHING.md), `stabilization.py` (`apply_transform` — the pure detection→stabilized-frame coordinate map; plus `EgoMotionStabilizer`/`NullDetectionStabilizer`, the `DetectionStabilizer` port impls so the stabilize step is timeable, see `src/tratrac/infrastructure/video/EGO_MOTION.md`+`src/tratrac/infrastructure/timing/STEP_TIMING.md`), `NullProgressReporter` (silent default), and `config.py` (the `RunConfig` value object + `RunConfig.resolve` merge/validate, `DetectorChoice`, `ConfigError`; the pure zero-defaults run spec, see `src/tratrac/application/CONFIG_DESIGN.md`).
  - `infrastructure/` — adapters: `video/opencv.py` (+ `video/window.py`, the pure `FrameWindow` seconds→frame-range math for `--start`/`--end` trimming, see `src/tratrac/infrastructure/video/TIME_WINDOW.md`), `video/ego_motion_orb.py` (`OrbEgoMotionEstimator`, the keyframe-anchored ORB+RANSAC `EgoMotionEstimator` adapter — matches each frame to a keyframe anchor, re-anchors on low overlap, returns the current-frame→global transform; MVP1.9, see `src/tratrac/infrastructure/video/EGO_MOTION.md`), `detection/rt_detr.py` (HF transformers), `detection/yolov8_visdrone.py` (**MVP1 emergency default**, see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`), `tracking/boxmot_bot_sort.py`, `export/ssam_trj.py` (binary v1.04 writer + a **viz-only** `read_trj` reader that reconstructs `VehicleState`s for the renderer — not for analytics, see `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`), `progress/console.py` (throttled stderr progress reporter), `timing/decorators.py` (per-port `Timed*` step-timing decorators) + `timing/csv.py` (wide-row CSV timing sink), `transform/recording.py` (`RecordingEgoMotionEstimator` — the `EgoMotionEstimator` decorator that tees each frame's transform to a `TransformSink`, leaving the pipeline untouched) + `transform/csv.py` (`CsvTransformSink` — per-frame `frame,a,b,c,d,tx,ty` rows so an offline tool can map stabilized coords back to raw; see `src/tratrac/infrastructure/video/EGO_MOTION.md`), `export/decimating.py` (`TrajectoryExporter` decorator thinning the TIMESTEP stream for `--timestep-precision`, see `src/tratrac/infrastructure/TIMESTEP_PRECISION.md`), `export/overlay_video.py` (`OverlayVideoExporter` — a **standalone** renderer, *not* a `TrajectoryExporter`, that writes a video of each **raw** frame with bumpers/IDs/trails drawn, mapping stabilized coordinates back onto the raw frame via an injected `transform_source`; cv2 behind injected seams. Driven by the post-hoc `tratrac-render`, not the pipeline — its `emit_frame` takes the `Frame`, unlike the frameless data port. See `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`), `config/toml.py` (`load_toml` via stdlib `tomllib` — the TOML config reader, see `src/tratrac/application/CONFIG_DESIGN.md`).
  - `calibration/` — drone-metadata GSD calibration (MVP1.75, see `src/tratrac/calibration/GSD_CALIBRATION.md`): `gsd.py` (`ground_sample_distance` from sensor + focal + altitude), `drone_specs.py` (`known_models`/`lookup` sensor+focal registry), `srt_parser.py` (`mean_altitude` from a DJI `.SRT` sidecar). Resolves the metres-per-pixel scale stamped into the track record (the offline smoother reads it to produce metric output).
  - `cli.py` — single-command Typer entry point (invoked `tratrac --config …`, **no `process` subcommand**). **Zero hardcoded defaults** and **config-only**: every value comes from the `--config` TOML, else the run fails listing every missing key (see `src/tratrac/application/CONFIG_DESIGN.md`). There are **no per-key override flags** — the run is driven entirely by the config. The only operational flag is `--force`/`--no-force` (overwrite control, default off); it is **not** a config key — overwrite policy never affects the trajectories, so it is not part of the reproducible run spec — and a non-TTY overwrite without force errors (exit 2). A `--check` flag (with `--json`) validates the config and exits without running — parse + `RunConfig.resolve` + the static path guards, aggregated; it never opens the video or downloads a checkpoint (for the Tauri config client / CI; see `src/tratrac/application/CONFIG_DESIGN.md` and `src/tratrac/CHECK_COMMAND.md`). (Earlier revisions mirrored every config key as a `--flag` override + a positional `VIDEO`; those were removed — see `src/tratrac/application/CONFIG_DESIGN.md` "Design history".) `tratrac.example.toml` (repo root) is a copyable template. The run is **perception only**: `export.out` is the track record; it produces no `.trj` (run `tratrac-postprocess` on the record) and no overlay video (run `tratrac-render`). `export.anchors_dir` (with `ego_motion.enabled`) exports the ORB keyframe anchors (PNGs + manifest) an operator draws exclusion zones on (src/tratrac/application/EXCLUSION_ZONES.md).
  - `cli_postprocess.py` / `cli_render.py` — sibling Typer apps for the other installed scripts: `tratrac-postprocess` (offline: **filter** exclusion zones track-aware, then optionally **project to metric world coords** via a post-hoc single homography, then Kalman/RTS de-jitter → smoothed `.trj`; `--exclusion-zones`/`--anchors`/`--exclusion-min-fraction`/`--calibration`, see `src/tratrac/application/EXCLUSION_ZONES.md`+`src/tratrac/application/SMOOTHING.md`+`src/tratrac/application/WORLD_PROJECTION.md`. The world-projection seam: `domain/world.py` value objects, the `WorldProjector` port + `application/world_projection.py` impls, `infrastructure/world/calibration.py` loader+cv2 fit — MVP2 Approach A), and `tratrac-render` (post-hoc overlay video ± violation marks: draws a `.trj`'s trajectories over its source clip via `OverlayVideoExporter`, reading the `.trj` with `read_trj`; see `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`). Installed scripts: `tratrac`, `tratrac-postprocess`, `tratrac-render`.
- `tests/` — `unit/` (fully isolated from heavy deps; includes `test_config.py` for run-config resolution) and `integration/` (1 e2e smoke test, marked `@pytest.mark.slow` because it downloads the detector checkpoint on first run).
- `scripts/` — standalone diagnostic tools (pure stdlib + cv2/numpy/matplotlib where possible, do **not** depend on the package internals so they work even when something is broken — the lone exception is `visualize_stabilization.py`, which *intentionally* imports the package to exercise the real stabilizer code path):
  - `plot_run.py` — per-run diagnostic figures from an outputs folder (stdlib + numpy + matplotlib, no package imports). Walks the directory, finds each baseline `.trj` (and its `_smooth.trj` / `.parquet` record siblings) and writes **one PNG per graph** into a per-run folder: speed/accel/jerk/trajectories (the baseline-vs-RTS-smoothed de-jitter story) plus track lifespans/length/active-counts/birth-death/detection-confidence (tracking & continuity). `--video CLIP_OR_FOLDER` draws one still behind the spatial panels (static camera); `--scout-dir DIR` draws a swept-area mosaic of the run's anchor frames (point it at `--anchors-dir`) warped by the run's transforms (moving drone).
  - `visualize_stabilization.py` — opens a `[ ORIGINAL | WARPED-into-reference ]` side-by-side window with a HUD of the live cumulative transform (translation/rotation/scale), so you can eyeball ORB ego-motion drift before deciding how to fix it. Imports the real `OrbEgoMotionEstimator` + warp (the exact pipeline code path). Default mode is pure ORB (fast, no model download, unmasked); `--mask` runs the real detector per frame and feeds detections back so vehicles are masked out (faithful to the pipeline, slower). `--no-window --save out.mp4` writes a headless side-by-side video instead.
  - `validate_trj.py` — semantic e2e validator for a `.trj` (see `scripts/validate_trj.md`): reports per-check compliance (continuity — no interior appear/disappear; orientation smoothness — no sudden front/rear switches; speed/accel physical plausibility against unit-aware bounds). Pure stdlib. `--violations-csv PATH` writes every non-compliant instance with its frame/timestamp/vehicle-id/reason and image-space position to locate it in the video; `--fail-under PCT` is a CI gate.

## Commands

All commands run from the repo root. `uv` manages the venv (`.venv/`) and resolves dependencies from `pyproject.toml` + `uv.lock`.

| Purpose | Command |
| --- | --- |
| Install / sync deps | `uv sync` |
| Add a runtime dep | `uv add <pkg>` |
| Add a dev dep | `uv add --dev <pkg>` |
| Run an arbitrary command in the env | `uv run <cmd>` |
| Format (tabs, double quotes) | `uv run ruff format .` |
| Lint | `uv run ruff check .` (add `--fix` to auto-fix) |
| Typecheck (strict mypy) | `uv run mypy` |
| Run all tests | `uv run pytest` |
| Run only unit tests (fast) | `uv run pytest tests/unit` |
| Run only the slow e2e test | `uv run pytest -m slow` |
| Run the CLI (replay a saved config) | `uv run tratrac --config run.toml` (writes the track record; every value is mandatory via the config — there are no per-key override flags) |
| Run the CLI (overwrite existing outputs) | `uv run tratrac --config run.toml --force` (`--force`/`--no-force` is the only flag) |
| Post-process the record into a `.trj` (filter + project + smooth) | `uv run tratrac-postprocess RECORD.parquet --out OUT.trj [--exclusion-zones Z.json --anchors M.json] [--exclusion-min-fraction 0.5] [--calibration CAL.json] [--pos-noise PX] [--jerk Q] [--timestep-precision S]` (the only `.trj` path; `--calibration` projects to metric world coords — MVP2 Approach A; see `src/tratrac/application/WORLD_PROJECTION.md`+`src/tratrac/application/EXCLUSION_ZONES.md`+`src/tratrac/application/SMOOTHING.md`) |
| Plot per-run diagnostics | `uv run python scripts/plot_run.py OUTPUTS_DIR [--out DIR] [--accel-bound 8.0] [--video CLIP_OR_FOLDER] [--scout-dir SCOUT_OR_PARENT]` |
| Visualize ORB stabilization | `uv run python scripts/visualize_stabilization.py VIDEO [--mask] [--no-window --save OUT.mp4 --max-frames N]` |
| Render the overlay video (trajectories + optional violations) | `uv run tratrac-render VIDEO --trj RUN.trj --out OUT.mp4 [--transforms T.csv] [--violations V.csv [--checks ...]] [--trail N]` (post-hoc; see `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`) |
| Semantically validate a `.trj` (e2e) | `uv run python scripts/validate_trj.py PATH [--violations-csv OUT.csv] [--fail-under PCT]` |

## Dependency Notes

- `torch` and `torchvision` are pinned to the **CPU index** (`https://download.pytorch.org/whl/cpu`) in `[tool.uv.sources]`. Swap to a CUDA index when a GPU is available — both packages must come from the same index or `torchvision::nms` won't register.
- `boxmot==19.x` reorganized its API; the tracker class is at `boxmot.trackers.BotSort`, not the top-level `boxmot`. **`boxmot` is AGPL-3.0** — relevant if TraTrac will be distributed.
- `ultralytics` is the **YOLOv8-VisDrone** runtime — also **AGPL-3.0**, also distribution-relevant. The dep is scoped to MVP1's emergency detector adapter today (see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`), but **stays** post-MVP1.5: the replanned MVP1.5 target is a YOLO-OBB fine-tune, which still needs `ultralytics`. Only the `yolov8_visdrone.py` adapter file, its CLI enum value, and `dill` (pickle dependency of that specific checkpoint) get removed — the AGPL trade-off itself is not resolved by MVP1.5 anymore.
- `dill` is pulled in because the `Mahadih534/YoloV8-VisDrone` checkpoint was pickled with it. Pinned explicitly so the ultralytics auto-installer doesn't re-trigger on every run.
- `pyarrow` is the Parquet engine for the track record (`infrastructure/tracks/parquet.py`). Pulled forward from the MVP7 storage plan.
- `av` (PyAV) encodes the `tratrac-render` overlay video (`_pyav_open_writer` in `infrastructure/export/overlay_video.py`, libx264) — `cv2.VideoWriter`'s bundled FFmpeg has no software H.264 encoder here and was silently falling back to the much less efficient `mp4v`, producing 3-5x oversized overlays; see `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`. This is also the first adoption of PyAV, named as the target video I/O library in `docs/TECH_STACK.md`; decode stays on `cv2` for now (see `docs/BACKLOG.md` item 3).
- `transformers`, `cv2`, `boxmot`, `ultralytics`, `huggingface_hub`, `pyarrow`, `av` are configured with `follow_imports = "skip"` in mypy, so they're treated as `Any` at the third-party seam. Everything else is fully typed under strict mypy.

## Code Style

- **Indentation: tabs.** Ruff's formatter is configured with `indent-style = "tab"`. `W191` (tab indentation) is ignored. Do not introduce spaces for indentation.
- **Line length: 100.** Enforced by the formatter; lint `E501` is off because formatter owns it.
- **Typing: strict.** `mypy strict = true` plus `warn_unreachable`, `warn_redundant_casts`, `warn_unused_ignores`. Public domain APIs must be fully annotated. No bare `Any` without a comment justifying it.
- **Lint rules selected:** `E,W,F,I,N,B,C4,UP,SIM,PTH,TID,RET,ARG,PT,RUF`. Notable: `PTH` (prefer `pathlib`), `TID` (no relative parent imports), `UP` (modern syntax for the target version).

## Source Of Truth: `docs/README.md`

Design docs are no longer a centralized vault — they live **next to the code they describe**
(a `DESIGN.md`/`ARCHITECTURE.md`/topic-named `.md` inside the relevant `src/tratrac/...`,
`scripts/`, or repo-root directory), plus a slim set of genuinely cross-cutting docs
(`docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/TECH_STACK.md`, `docs/BACKLOG.md`,
`docs/roadmap/` for not-yet-started MVPs) at the repo root. **`docs/README.md` is the map of
all of it** — read it before proposing or writing code; it links every doc by the module it
sits beside. Do not re-list individual docs here — that duplicates `docs/README.md` and the two
will drift. If a doc and the code disagree, surface the conflict and ask which is authoritative
before editing.

## What TraTrac Is

A vehicle tracking and trajectory-export system for **cenital/nadir aerial video**. The pipeline detects vehicles, tracks identities across occlusions and re-entries, projects to world-space metric coordinates (with multi-homography for bridges/overpasses), and exports SSAM-compatible `.trj` files for traffic safety analytics.

## Load-Bearing Architectural Invariants

These are project-defining decisions recorded across the repo's design docs (see `docs/README.md`). Do not violate them without an explicit conversation with the user.

- **SSAM `.trj` is an export format, never the internal representation.** The canonical in-memory type is `VehicleState` (see `src/tratrac/domain/ARCHITECTURE.md`), which carries polygons, embeddings, plane/lane metadata, and uncertainty that SSAM cannot represent. Using SSAM internally would cripple future analytics.
- **Dual export architecture, B-first.** (B) the extended internal record (raw tracked measurements now; masks/embeddings/topology/uncertainty later) is the **pipeline's primary output**. (A) the SSAM `.trj` is a **derived, post-hoc** product built from B by `tratrac-postprocess`. New analytics or debug data goes into (B), never into (A). The pipeline never re-ingests A.
- **Every MVP must be able to emit syntactically valid SSAM `.trj`, starting from MVP1.** MVPs differ in trajectory *quality*, not in whether trajectories exist. Since the export inversion this is satisfied in **two steps** (`tratrac` → record, then `tratrac-postprocess` → `.trj`), not one command. Orientation, front/rear point, and dimensions are estimated in the smoother (from trajectory direction + bbox aspect ratio).
- **Coordinate semantics by MVP.** MVP1 may emit image-space coordinates (syntactically valid, not physically meaningful). MVP2+ SSAM exports **must** be world-space metric — image-space coordinates in SSAM make analytics scientifically invalid even though the file parses.
- **Multi-homography, not full 3D.** Roads are treated as piecewise planar (MVP3+). Full 3D reconstruction is explicitly rejected as unnecessary and operationally expensive.

## MVP Roadmap

Each MVP delivers an end-to-end runnable system that improves trajectory quality. **The MVP
number is a capability ID (a dependency ladder, `1 → 1.5 → 1.75 → 1.9 → 2 → 3 → 4 → 5 → 6 → 7`),
not execution order** — work has crossed MVP boundaries (shortcuts inserted, later foundations
pulled forward, one milestone skipped). **The single, authoritative status table lives in
`docs/ROADMAP.md`** ("Roadmap: Capability IDs vs Execution Order") — read it there, not here,
so this file and that one don't drift out of sync. Each MVP doc (now scattered across the
code-adjacent design docs and `docs/roadmap/` — see `docs/README.md`) carries its own Status
banner too.

A **supporting layer** (the cross-cutting docs indexed in `docs/README.md`: progress, timing,
validation, time-window, timestep, config, render, exclusion, smoothing) was built **outside**
this numbering. When proposing changes, identify which MVP (capability) the work belongs to;
pulling a capability forward is allowed but say so and record it in the `docs/ROADMAP.md`
reconciliation.

## Target Tech Stack

Per `docs/TECH_STACK.md`: PyTorch runtime; NVDEC+TorchCodec decoding, PyAV encoding; SuperPoint+LightGlue stabilization; **YOLO-OBB** detection (deliberately *not* RT-DETR — RT-DETR has no OBB support and its dense-oblique-scene advantage doesn't fit nadir-only footage); **SAM 3** segmentation, scope narrowed now that OBB detection already reports orientation (deliberately *not* Mask R-CNN, and not SAM2 which is superseded); **BoT-SORT** tracking (deliberately *not* plain SORT) with **DINOv3** ReID embeddings + motion-plausibility gating (*not* FastReID — nadir view discards most of what vehicle-ReID models are trained to see); constant-acceleration Kalman/RTS motion modeling (shipped), KalmanNet-family adaptive noise as the frontier upgrade; OpenCV multi-homography, with road-geometry-based auto-calibration as a frontier upgrade to manual correspondence-authoring; Parquet + FiftyOne + Docker/CUDA for MVP7. `docs/TECH_STACK.md` explains *why* each was chosen and what was rejected, with citations — preserve that reasoning when picking libraries.

Python 3.12 is the implementation language, managed with `uv`. None of the runtime ML dependencies are pinned yet — add them with `uv add` as each MVP component lands.

## Working In This Repo

- Per the user's global instructions, follow the docs-first workflow: inspect `docs/README.md` and the code-adjacent design docs it indexes, identify conflicts, understand the goal, propose approaches with trade-offs, and ask before implementing.
- Before reporting any change as complete, run `uv run ruff format .`, `uv run ruff check .`, and `uv run mypy`. Strict mypy will reject untyped functions — annotate as you write, not after.
- When new architectural facts emerge during implementation, update the relevant design doc (next to the code it describes, or in `docs/` for cross-cutting ones — see `docs/README.md`) rather than scattering decisions across code comments.
