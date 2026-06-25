# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

**MVP1.75 landed** — metric sizes/speeds from drone-metadata GSD calibration (see `vault/05_5_mvp1_75.md`): `Length`/`Width`/`Speed`/`Acceleration` and `DIMENSIONS.Scale` are now real metric values. **MVP1.9 landed** — ORB + RANSAC similarity ego-motion as an optional step (`ego_motion.enabled`, off by default; see `vault/05_75_mvp1_9.md`): when on, detection/tracking run on the **raw, full-resolution frame** and the keyframe-anchored ego-motion transform is applied to the **detections** (coordinates, not pixels) before tracking, so trajectories are ego-motion-free and nothing is ever cropped to black. The keyframe anchor re-sets when too little of it stays in view (`ego_motion.min_anchor_overlap`). It is an intermediate "keep if good enough" shortcut before MVP2's learned stabilizer + world projection. **MVP1.5 (RT-DETR fine-tune) was leapfrogged, not yet done** — 1.75/1.9 are independent shortcuts that slot between MVP1 and MVP2. So detection + tracking are still MVP1-grade (YOLOv8-VisDrone emergency detector + IoU-only BoT-SORT, no ReID), and SSAM **positions remain image-space pixels** (MVP2 homography/world projection not started). **Export inverted (perception-only pipeline)** — `tratrac` now writes only the **track record** (raw tracked measurements, `export.out`, the canonical internal "export B"); it no longer computes kinematics or writes a `.trj`. The SSAM `.trj` is produced **post-hoc** by `tratrac-postprocess` (the only `.trj` path; runs the Kalman/RTS smoother and the orientation/kinematics). The orientation subsystem and the live `.trj` exporter were removed from the run. The record is an **Apache Parquet** file (`ParquetTrackSink` / `read_tracks` in `infrastructure/tracks/parquet.py`; run metadata + scale live in the Parquet schema metadata) — this pulls the MVP7 Parquet-storage choice forward for the canonical record. The repo contains:

- `vault/` — authoritative design documents (system overview, architecture principles, coordinate systems, tech stack, MVP roadmap, SSAM format spec). The two SSAM PDFs (v1.04 + v3.0) live here as the ground truth for the binary format.
- `pyproject.toml`, `uv.lock`, `.python-version` (3.12) — uv-managed project.
- `src/tratrac/` — the implementation, organized in onion layers:
  - `domain/` — pure value objects (geometry — including `Transform2D`, the affine/similarity transform used for stabilization — frame, detection, vehicle, progress events, step-timing records, per-frame ego-motion transform records in `stabilization.py` — `FrameTransform`) and Protocol ports (`VideoSource`, `EgoMotionEstimator`, `Detector`, `DetectionObserver`, `DetectionStabilizer`, `Tracker`, `TrackSink`, `TrajectoryExporter`, `ProgressReporter`, `TimingSink`, `TransformSink`, `AnchorSink`).
  - `application/` — `TrajectoryPipeline` (perception-only per-frame orchestrator that emits a progress event stream and records each frame's tracked detections to a `TrackSink` — its primary output; when ego-motion is enabled it stabilizes detections via the `DetectionStabilizer` port before tracking — MVP1.9), `track_smoothing.py` (`smooth_to_states`/`build_state` — reconstruct `VehicleState`s with kinematics from raw track samples; the offline `tratrac-postprocess` core, vault/22), `stabilization.py` (`apply_transform` — the pure detection→stabilized-frame coordinate map; plus `EgoMotionStabilizer`/`NullDetectionStabilizer`, the `DetectionStabilizer` port impls so the stabilize step is timeable, see `vault/05_75_mvp1_9.md`+`vault/15_step_timing.md`), `NullProgressReporter` (silent default), and `config.py` (the `RunConfig` value object + `RunConfig.resolve` merge/validate, `DetectorChoice`, `ConfigError`; the pure zero-defaults run spec, see `vault/19_config_file.md`).
  - `infrastructure/` — adapters: `video/opencv.py` (+ `video/window.py`, the pure `FrameWindow` seconds→frame-range math for `--start`/`--end` trimming, see `vault/17_time_window.md`), `video/ego_motion_orb.py` (`OrbEgoMotionEstimator`, the keyframe-anchored ORB+RANSAC `EgoMotionEstimator` adapter — matches each frame to a keyframe anchor, re-anchors on low overlap, returns the current-frame→global transform; MVP1.9, see `vault/05_75_mvp1_9.md`), `detection/rt_detr.py` (HF transformers), `detection/yolov8_visdrone.py` (**MVP1 emergency default**, see `vault/05_mvp1.md`), `tracking/boxmot_bot_sort.py`, `export/ssam_trj.py` (binary v1.04 writer + a **viz-only** `read_trj` reader that reconstructs `VehicleState`s for the renderer — not for analytics, see `vault/20_video_export.md`), `progress/console.py` (throttled stderr progress reporter), `timing/decorators.py` (per-port `Timed*` step-timing decorators) + `timing/csv.py` (wide-row CSV timing sink), `transform/recording.py` (`RecordingEgoMotionEstimator` — the `EgoMotionEstimator` decorator that tees each frame's transform to a `TransformSink`, leaving the pipeline untouched) + `transform/csv.py` (`CsvTransformSink` — per-frame `frame,a,b,c,d,tx,ty` rows so an offline tool can map stabilized coords back to raw; see `vault/05_75_mvp1_9.md`), `export/decimating.py` (`TrajectoryExporter` decorator thinning the TIMESTEP stream for `--timestep-precision`, see `vault/18_timestep_precision.md`), `export/overlay_video.py` (`OverlayVideoExporter` — a **standalone** renderer, *not* a `TrajectoryExporter`, that writes a video of each **raw** frame with bumpers/IDs/trails drawn, mapping stabilized coordinates back onto the raw frame via an injected `transform_source`; cv2 behind injected seams. Driven by the post-hoc `tratrac-render`, not the pipeline — its `emit_frame` takes the `Frame`, unlike the frameless data port. See `vault/20_video_export.md`), `config/toml.py` (`load_toml` via stdlib `tomllib` — the TOML config reader, see `vault/19_config_file.md`).
  - `calibration/` — drone-metadata GSD calibration (MVP1.75, see `vault/05_5_mvp1_75.md`): `gsd.py` (`ground_sample_distance` from sensor + focal + altitude), `drone_specs.py` (`known_models`/`lookup` sensor+focal registry), `srt_parser.py` (`mean_altitude` from a DJI `.SRT` sidecar). Resolves the metres-per-pixel scale stamped into the track record (the offline smoother reads it to produce metric output).
  - `cli.py` — single-command Typer entry point (invoked `tratrac VIDEO …` / `tratrac --config …`, **no `process` subcommand**). **Zero hardcoded defaults**: every value comes from the `--config` TOML or a flag, else the run fails listing every missing key (see `vault/19_config_file.md`). Flags (`--detector`, `--conf`, `--checkpoint`, `--checkpoint-file`, `--device`, `--meters-per-pixel`/`--drone-model`/`--altitude`/`--srt`, `--stabilize`/`--no-stabilize` + `--orb-features`/`--orb-match-ratio`/`--orb-min-matches`/`--orb-ransac-threshold`, `--det-thresh`, `--process-fps`, `--transform-csv`, `--anchors-dir`, `--start`/`--end`, `--timing-csv`, `--out`) override the config key-for-key; the positional `VIDEO` overrides `input.video`. `--force`/`--no-force` controls overwrite; a non-TTY overwrite without force errors (exit 2). `tratrac.example.toml` (repo root) is a copyable template. The run is **perception only**: `--out` is the track record; it produces no `.trj` (run `tratrac-postprocess` on the record) and no overlay video (run `tratrac-render`). `--anchors-dir` (with `--stabilize`) exports the ORB keyframe anchors (PNGs + manifest) an operator draws exclusion zones on (vault/21).
  - `cli_postprocess.py` / `cli_render.py` — sibling Typer apps for the other installed scripts: `tratrac-postprocess` (offline: **filter** exclusion zones track-aware, then Kalman/RTS de-jitter → smoothed `.trj`; `--exclusion-zones`/`--anchors`/`--exclusion-min-fraction`, see `vault/21`+`vault/22`), and `tratrac-render` (post-hoc overlay video ± violation marks: draws a `.trj`'s trajectories over its source clip via `OverlayVideoExporter`, reading the `.trj` with `read_trj`; see `vault/20_video_export.md`). Installed scripts: `tratrac`, `tratrac-postprocess`, `tratrac-render`.
- `tests/` — `unit/` (fully isolated from heavy deps; includes `test_config.py` for run-config resolution) and `integration/` (1 e2e smoke test, marked `@pytest.mark.slow` because it downloads the detector checkpoint on first run).
- `scripts/` — standalone diagnostic tools (pure stdlib + cv2/numpy/matplotlib where possible, do **not** depend on the package internals so they work even when something is broken — the lone exception is `visualize_stabilization.py`, which *intentionally* imports the package to exercise the real stabilizer code path):
  - `plot_run.py` — per-run diagnostic figures from an outputs folder (stdlib + numpy + matplotlib, no package imports). Walks the directory, finds each baseline `.trj` (and its `_smooth.trj` / `.parquet` record siblings) and writes **one PNG per graph** into a per-run folder: speed/accel/jerk/trajectories (the baseline-vs-RTS-smoothed de-jitter story) plus track lifespans/length/active-counts/birth-death/detection-confidence (tracking & continuity). `--video CLIP_OR_FOLDER` draws one still behind the spatial panels (static camera); `--scout-dir DIR` draws a swept-area mosaic of the run's anchor frames (point it at `--anchors-dir`) warped by the run's transforms (moving drone).
  - `visualize_stabilization.py` — opens a `[ ORIGINAL | WARPED-into-reference ]` side-by-side window with a HUD of the live cumulative transform (translation/rotation/scale), so you can eyeball ORB ego-motion drift before deciding how to fix it. Imports the real `OrbEgoMotionEstimator` + warp (the exact pipeline code path). Default mode is pure ORB (fast, no model download, unmasked); `--mask` runs the real detector per frame and feeds detections back so vehicles are masked out (faithful to the pipeline, slower). `--no-window --save out.mp4` writes a headless side-by-side video instead.
  - `validate_trj.py` — semantic e2e validator for a `.trj` (see `vault/16_trj_validation.md`): reports per-check compliance (continuity — no interior appear/disappear; orientation smoothness — no sudden front/rear switches; speed/accel physical plausibility against unit-aware bounds). Pure stdlib. `--violations-csv PATH` writes every non-compliant instance with its frame/timestamp/vehicle-id/reason and image-space position to locate it in the video; `--fail-under PCT` is a CI gate.

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
| Run the CLI (replay a saved config) | `uv run tratrac --config run.toml` |
| Run the CLI (config + overrides) | `uv run tratrac VIDEO --config run.toml [--out RECORD] [--conf 0.4] …` (writes the track record; every value is mandatory via config or flag) |
| Post-process the record into a `.trj` (filter + smooth) | `uv run tratrac-postprocess RECORD.parquet --out OUT.trj [--exclusion-zones Z.json --anchors M.json] [--exclusion-min-fraction 0.5] [--pos-noise PX] [--jerk Q] [--timestep-precision S]` (the only `.trj` path; see `vault/21`+`vault/22`) |
| Plot per-run diagnostics | `uv run python scripts/plot_run.py OUTPUTS_DIR [--out DIR] [--accel-bound 8.0] [--video CLIP_OR_FOLDER] [--scout-dir SCOUT_OR_PARENT]` |
| Visualize ORB stabilization | `uv run python scripts/visualize_stabilization.py VIDEO [--mask] [--no-window --save OUT.mp4 --max-frames N]` |
| Render the overlay video (trajectories + optional violations) | `uv run tratrac-render VIDEO --trj RUN.trj --out OUT.mp4 [--transforms T.csv] [--violations V.csv [--checks ...]] [--trail N]` (post-hoc; see `vault/20_video_export.md`) |
| Semantically validate a `.trj` (e2e) | `uv run python scripts/validate_trj.py PATH [--violations-csv OUT.csv] [--fail-under PCT]` |

## Dependency Notes

- `torch` and `torchvision` are pinned to the **CPU index** (`https://download.pytorch.org/whl/cpu`) in `[tool.uv.sources]`. Swap to a CUDA index when a GPU is available — both packages must come from the same index or `torchvision::nms` won't register.
- `boxmot==19.x` reorganized its API; the tracker class is at `boxmot.trackers.BotSort`, not the top-level `boxmot`. **`boxmot` is AGPL-3.0** — relevant if TraTrac will be distributed.
- `ultralytics` is the **YOLOv8-VisDrone** runtime — also **AGPL-3.0**, also distribution-relevant. The dep is scoped to MVP1's emergency detector adapter (see `vault/05_mvp1.md`); when RT-DETR fine-tuning lands in MVP1.5, this dep + the `yolov8_visdrone.py` adapter file + the CLI enum value get removed in one cleanup.
- `dill` is pulled in because the `Mahadih534/YoloV8-VisDrone` checkpoint was pickled with it. Pinned explicitly so the ultralytics auto-installer doesn't re-trigger on every run.
- `pyarrow` is the Parquet engine for the track record (`infrastructure/tracks/parquet.py`). Pulled forward from the MVP7 storage plan.
- `transformers`, `cv2`, `boxmot`, `ultralytics`, `huggingface_hub`, `pyarrow` are configured with `follow_imports = "skip"` in mypy, so they're treated as `Any` at the third-party seam. Everything else is fully typed under strict mypy.

## Code Style

- **Indentation: tabs.** Ruff's formatter is configured with `indent-style = "tab"`. `W191` (tab indentation) is ignored. Do not introduce spaces for indentation.
- **Line length: 100.** Enforced by the formatter; lint `E501` is off because formatter owns it.
- **Typing: strict.** `mypy strict = true` plus `warn_unreachable`, `warn_redundant_casts`, `warn_unused_ignores`. Public domain APIs must be fully annotated. No bare `Any` without a comment justifying it.
- **Lint rules selected:** `E,W,F,I,N,B,C4,UP,SIM,PTH,TID,RET,ARG,PT,RUF`. Notable: `PTH` (prefer `pathlib`), `TID` (no relative parent imports), `UP` (modern syntax for the target version).

## Source Of Truth: `vault/`

`vault/` is the project's canonical design knowledge. **Read it before proposing or writing code.** The intended reading order is the numeric prefix:

- `00_system_overview.md` — system objective, core philosophy, final vision.
- `01_architecture_principles.md` — internal vs external representation, `VehicleState`, mapping to SSAM.
- `02_coordinate_systems.md` — image-space vs world-space, multi-homography rationale.
- `03_tech_stack.md` — ideal final stack and the *why* behind each choice (and what was rejected).
- `04_ssam_format.md` — SSAM `.trj` shape and MVP1 orientation approximation.
- `05_mvp1.md` … `11_mvp7.md` — staged MVPs; each adds one capability. `05_25_mvp1_5.md` (RT-DETR fine-tune + YOLOv8-adapter removal; open, leapfrogged by 1.75), `05_5_mvp1_75.md` (drone-metadata calibration shortcut, shipped), and `05_75_mvp1_9.md` (ORB video stabilization, shipped) slot between MVP1 and MVP2.
- `12_final_architecture.md` — end-state pipeline diagram.
- `13_road_topology.md` — how SSAM `Link ID` / `Lane ID` get sourced across MVPs.
- `14_progress_reporting.md` — the `ProgressReporter` output port and the `ProgressEvent` family (console now, any UI client later).
- `15_step_timing.md` — the `Timed*` decorators and the `TimingSink` port (CSV now, telemetry later). **Every per-frame step is a port and is timed**: `detect,observe,ego_motion,stabilize,track,record` (the wide CSV columns); the three stabilization-only steps are blank on a non-stabilized run. Stabilization was promoted from an inline `apply_transform` to the `DetectionStabilizer` port precisely so it's decoratable; `TimedEgoMotion` wraps the ORB innermost so its measurement excludes the transform/anchor recording I/O.
- `16_trj_validation.md` — the semantic e2e `.trj` validator: the three checks (continuity, orientation smoothness, kinematic plausibility), why no-ground-truth bounds what can be validated, and why a kinematic *consistency* check was rejected as tautological.
- `17_time_window.md` — `--start`/`--end` analysis-window trimming: why it lives in the video adapter (seek), the pure `FrameWindow` math, and why trimmed clips keep absolute TIMESTEPs.
- `18_timestep_precision.md` — `--timestep-precision` output decimation (export seam) **and** `--process-fps`/`input.process_fps` decode-time decimation (video adapter): both reuse the shared `DecimationGrid` (`infrastructure/cadence.py`); the export knob thins only the `.trj` (no compute saving), the processing knob skips frames with `cv2.grab()` for a real speedup at the cost of more BoT-SORT ID switches. They stack (process first, export thins further); the anchored emission-grid math and the SSAM coarseness caveat live here.
- `19_config_file.md` — the persisted run config: zero hardcoded defaults (every value from the `--config` TOML or a flag, else fail), `RunConfig.resolve` precedence/aggregation, the "every key present, off is explicit" rule, calibration one-of, and the removal of the library pixel fallback.
- `20_video_export.md` — the **post-hoc** overlay video (`tratrac-render`): why trajectory rendering was pulled out of the live pipeline (per-frame encode cost; fully derivable from the `.trj`), the viz-only `read_trj` reader, the no-y-flip image-space drawing mapped back onto the raw frame, and why the `TrajectoryExporter` port is frameless (the renderer is a standalone non-port class).
- `21_exclusion_zones.md` — image-space "do-not-analyze" pixel polygons, applied **post-hoc and track-aware** by `tratrac-postprocess --exclusion-zones` (off by default): a whole **track** is dropped when the majority of its observations' centroids fall inside the polygons' union (`--exclusion-min-fraction`, default 0.5; `domain/geometry.point_in_polygon` + `application/exclusion.excluded_track_ids`). The perception run no longer masks. Zones carry a `reference_frame` (0 = static); for a moving drone they are drawn on the **anchor frames the run emits** (`--anchors-dir` → PNGs + `manifest.json` via the ORB `anchor_observer` seam), and post-process reads each anchor's pose from the manifest (`--anchors`) to map zones into the global frame. Scout, replay, and the in-pipeline `DetectionMask` were removed.
- `22_smoothing.md` — constant-acceleration **Kalman/RTS trajectory de-jittering**, and the **only** `.trj` path: pass 1 is the perception run, which writes the track record (`export.out`, the canonical raw measurements via the pipeline's `TrackSink`); pass 2 (`tratrac-postprocess`) runs a forward Kalman + RTS backward pass per track (`application/kalman.py`, hand-rolled numpy, no filterpy), reconstructs `VehicleState`s (kinematics) via `build_state`, and writes the smoothed `.trj` (`--timestep-precision` thins its TIMESTEPs). Smooth position, read velocity/accel out of the filter state — the root-cause fix for the accel-noise issue. Re-tunable offline (no re-detection).
- `final_polish.md` — backlog of deliberate "ship cheaper now, upgrade later" decisions behind stable ports (not numbered: it tracks quality upgrades to existing capabilities, not new MVP capabilities). First entry: replace the MVP2 OpenCV-ECC ego-motion adapter with SuperPoint+LightGlue via the `EgoMotionEstimator` port.

If the vault and any future code disagree, surface the conflict and ask which is authoritative before editing.

## What TraTrac Is

A vehicle tracking and trajectory-export system for **cenital/nadir aerial video**. The pipeline detects vehicles, tracks identities across occlusions and re-entries, projects to world-space metric coordinates (with multi-homography for bridges/overpasses), and exports SSAM-compatible `.trj` files for traffic safety analytics.

## Load-Bearing Architectural Invariants

These are project-defining decisions from `vault/`. Do not violate them without an explicit conversation with the user.

- **SSAM `.trj` is an export format, never the internal representation.** The canonical in-memory type is `VehicleState` (see `vault/01_architecture_principles.md`), which carries polygons, embeddings, plane/lane metadata, and uncertainty that SSAM cannot represent. Using SSAM internally would cripple future analytics.
- **Dual export architecture, B-first.** (B) the extended internal record (raw tracked measurements now; masks/embeddings/topology/uncertainty later) is the **pipeline's primary output**. (A) the SSAM `.trj` is a **derived, post-hoc** product built from B by `tratrac-postprocess`. New analytics or debug data goes into (B), never into (A). The pipeline never re-ingests A.
- **Every MVP must be able to emit syntactically valid SSAM `.trj`, starting from MVP1.** MVPs differ in trajectory *quality*, not in whether trajectories exist. Since the export inversion this is satisfied in **two steps** (`tratrac` → record, then `tratrac-postprocess` → `.trj`), not one command. Orientation, front/rear point, and dimensions are estimated in the smoother (from trajectory direction + bbox aspect ratio).
- **Coordinate semantics by MVP.** MVP1 may emit image-space coordinates (syntactically valid, not physically meaningful). MVP2+ SSAM exports **must** be world-space metric — image-space coordinates in SSAM make analytics scientifically invalid even though the file parses.
- **Multi-homography, not full 3D.** Roads are treated as piecewise planar (MVP3+). Full 3D reconstruction is explicitly rejected as unnecessary and operationally expensive.

## MVP Roadmap

Work is staged so each MVP delivers an end-to-end runnable system that improves trajectory quality:

| MVP | Adds |
| --- | --- |
| 1 | RT-DETR + BoT-SORT, approximate orientation, image-space SSAM `.trj`. **Shipped with a temporary YOLOv8-VisDrone detector adapter** (see `vault/05_mvp1.md`) because COCO-RT-DETR doesn't see aerial cars and fine-tuning was out of timebox. |
| 1.5 | Fine-tune RT-DETR on VisDrone/UAVDT; remove the YOLOv8 adapter, restore RT-DETR as default. |
| 1.75 | **Metric sizes and speeds from drone metadata.** GSD calibration from sensor + focal + altitude; `Length` / `Width` / `Speed` / `Acceleration` in real metres / m·s⁻¹ / m·s⁻²; `DIMENSIONS.Scale` populated. No homography. See `vault/05_5_mvp1_75.md`. |
| 1.9 | **ORB ego-motion (intermediate).** Keyframe-anchored ORB + RANSAC 4-DOF similarity behind the `EgoMotionEstimator` port. Detection/tracking run on the **raw frame**; the transform is applied to **detections** (coordinates, not pixels) before tracking, so nothing is cropped. Still image-space. Optional + off by default; "keep if good enough" before MVP2's learned stabilizer. See `vault/05_75_mvp1_9.md`. |
| 2 | SuperPoint+LightGlue stabilization, single-homography world projection (handles moving drones, non-nadir gimbals, fixed cameras without telemetry) |
| 3 | Multi-homography + polygon-based plane assignment + **Link ID assignment from hand-drawn polygons** (see `vault/13_road_topology.md`) |
| 4 | SAM2 segmentation, mask-based orientation, dual export begins |
| 5 | FastReID + embedding memory for long-term identity persistence |
| 6 | Lane-graph topology constraints + **Lane ID assignment from hand-drawn lane polygons** |
| 7 | Apache Parquet storage, FiftyOne visualization, async/Docker deployment |

When proposing changes, identify which MVP the work belongs to and avoid pulling capabilities forward without justification.

## Target Tech Stack

Per `vault/03_tech_stack.md`: PyTorch runtime; NVDEC+PyAV decoding; SuperPoint+LightGlue stabilization; **RT-DETR** detection (deliberately *not* YOLO — aerial robustness over speed); **SAM2** segmentation (deliberately *not* Mask R-CNN); **BoT-SORT** tracking (deliberately *not* plain SORT); FastReID; EKF motion; OpenCV multi-homography; Parquet + FiftyOne + Docker/CUDA for MVP7. The vault explains *why* each was chosen and what was rejected — preserve that reasoning when picking libraries.

Python 3.12 is the implementation language, managed with `uv`. None of the runtime ML dependencies are pinned yet — add them with `uv add` as each MVP component lands.

## Working In This Repo

- Per the user's global instructions, follow the `vault`-first workflow: inspect vault, identify conflicts, understand the goal, propose approaches with trade-offs, and ask before implementing.
- Before reporting any change as complete, run `uv run ruff format .`, `uv run ruff check .`, and `uv run mypy`. Strict mypy will reject untyped functions — annotate as you write, not after.
- When new architectural facts emerge during implementation, update the relevant `vault/*.md` file rather than scattering decisions across code comments.
