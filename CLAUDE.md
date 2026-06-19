# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

**MVP1.75 landed** — metric sizes/speeds from drone-metadata GSD calibration (see `vault/05_5_mvp1_75.md`): `Length`/`Width`/`Speed`/`Acceleration` and `DIMENSIONS.Scale` are now real metric values. **MVP1.9 landed** — ORB + RANSAC similarity ego-motion as an optional step (`ego_motion.enabled`, off by default; see `vault/05_75_mvp1_9.md`): when on, detection/tracking run on the **raw, full-resolution frame** and the keyframe-anchored ego-motion transform is applied to the **detections** (coordinates, not pixels) before tracking, so trajectories are ego-motion-free and nothing is ever cropped to black. The keyframe anchor re-sets when too little of it stays in view (`ego_motion.min_anchor_overlap`). It is an intermediate "keep if good enough" shortcut before MVP2's learned stabilizer + world projection. **MVP1.5 (RT-DETR fine-tune) was leapfrogged, not yet done** — 1.75/1.9 are independent shortcuts that slot between MVP1 and MVP2. So detection + tracking are still MVP1-grade (YOLOv8-VisDrone emergency detector + IoU-only BoT-SORT, no ReID), and SSAM **positions remain image-space pixels** (MVP2 homography/world projection not started). The repo contains:

- `vault/` — authoritative design documents (system overview, architecture principles, coordinate systems, tech stack, MVP roadmap, SSAM format spec). The two SSAM PDFs (v1.04 + v3.0) live here as the ground truth for the binary format.
- `pyproject.toml`, `uv.lock`, `.python-version` (3.12) — uv-managed project.
- `src/tratrac/` — the implementation, organized in onion layers:
  - `domain/` — pure value objects (geometry — including `Transform2D`, the affine/similarity transform used for stabilization — frame, detection, vehicle, progress events, step-timing records, per-frame ego-motion transform records in `stabilization.py` — `FrameTransform`) and Protocol ports (`VideoSource`, `EgoMotionEstimator`, `Detector`, `Tracker`, `OrientationEstimator`, `TrajectoryExporter`, `ProgressReporter`, `TimingSink`, `TransformSink`).
  - `application/` — `EmaOrientationEstimator` (batch `OrientationEstimator` port impl: per-track kinematics with EMA-smoothed heading), `TrajectoryPipeline` (per-frame orchestrator that emits a progress event stream; when ego-motion is enabled it applies the transform to detections before tracking — MVP1.9), `stabilization.py` (`apply_transform` — the pure detection→stabilized-frame coordinate map, see `vault/05_75_mvp1_9.md`), `NullProgressReporter` (silent default), and `config.py` (the `RunConfig` value object + `RunConfig.resolve` merge/validate, `DetectorChoice`, `ConfigError`; the pure zero-defaults run spec, see `vault/19_config_file.md`).
  - `infrastructure/` — adapters: `video/opencv.py` (+ `video/window.py`, the pure `FrameWindow` seconds→frame-range math for `--start`/`--end` trimming, see `vault/17_time_window.md`), `video/ego_motion_orb.py` (`OrbEgoMotionEstimator`, the keyframe-anchored ORB+RANSAC `EgoMotionEstimator` adapter — matches each frame to a keyframe anchor, re-anchors on low overlap, returns the current-frame→global transform; MVP1.9, see `vault/05_75_mvp1_9.md`), `detection/rt_detr.py` (HF transformers), `detection/yolov8_visdrone.py` (**MVP1 emergency default**, see `vault/05_mvp1.md`), `tracking/boxmot_bot_sort.py`, `export/ssam_trj.py` (binary v1.04 writer), `progress/console.py` (throttled stderr progress reporter), `timing/decorators.py` (per-port `Timed*` step-timing decorators) + `timing/csv.py` (wide-row CSV timing sink), `transform/recording.py` (`RecordingEgoMotionEstimator` — the `EgoMotionEstimator` decorator that tees each frame's transform to a `TransformSink`, leaving the pipeline untouched) + `transform/csv.py` (`CsvTransformSink` — per-frame `frame,a,b,c,d,tx,ty` rows so an offline tool can map stabilized coords back to raw; see `vault/05_75_mvp1_9.md`), `export/decimating.py` (`TrajectoryExporter` decorator thinning the TIMESTEP stream for `--timestep-precision`, see `vault/18_timestep_precision.md`), `export/overlay_video.py` (`OverlayVideoExporter` — a `TrajectoryExporter` that writes a video of each **raw** frame with bumpers/IDs/trails drawn, mapping stabilized coordinates back onto the raw frame via an injected `transform_source`; cv2 behind injected seams, see `vault/20_video_export.md`) + `export/composite.py` (`CompositeTrajectoryExporter` — GoF Composite fanning the export stream to several exporters, e.g. `.trj` + overlay video), `config/toml.py` (`load_toml` via stdlib `tomllib` — the TOML config reader, see `vault/19_config_file.md`).
  - `calibration/` — drone-metadata GSD calibration (MVP1.75, see `vault/05_5_mvp1_75.md`): `gsd.py` (`ground_sample_distance` from sensor + focal + altitude), `drone_specs.py` (`known_models`/`lookup` sensor+focal registry), `srt_parser.py` (`mean_altitude` from a DJI `.SRT` sidecar). Resolves the metres-per-pixel scale that feeds the exporter and orientation estimator.
  - `cli.py` — single-command Typer entry point (invoked `tratrac VIDEO …` / `tratrac --config …`, **no `process` subcommand**). **Zero hardcoded defaults**: every value comes from the `--config` TOML or a flag, else the run fails listing every missing key (see `vault/19_config_file.md`). Flags (`--detector`, `--conf`, `--checkpoint`, `--checkpoint-file`, `--device`, `--meters-per-pixel`/`--drone-model`/`--altitude`/`--srt`, `--stabilize`/`--no-stabilize` + `--orb-features`/`--orb-match-ratio`/`--orb-min-matches`/`--orb-ransac-threshold`, `--det-thresh`, `--smoothing-window`, `--timestep-precision`, `--process-fps`, `--video-out`/`--video-trail`, `--transform-csv`, `--tracks`, `--start`/`--end`, `--timing-csv`, `--out`) override the config key-for-key; the positional `VIDEO` overrides `input.video`. `--force`/`--no-force` controls overwrite; a non-TTY overwrite without force errors (exit 2). `tratrac.example.toml` (repo root) is a copyable template. Installed script: `tratrac`.
- `tests/` — `unit/` (fully isolated from heavy deps; includes `test_config.py` for run-config resolution) and `integration/` (1 e2e smoke test, marked `@pytest.mark.slow` because it downloads the detector checkpoint on first run).
- `scripts/` — standalone diagnostic tools (pure stdlib + cv2 where possible, do **not** depend on the package internals so they work even when something is broken):
  - `dump_trj.py` — human-readable dump of a binary `.trj` (FORMAT/DIMENSIONS/TIMESTEP/VEHICLE records with totals).
  - `probe_detector.py` — runs a detector with all filters off on a single chosen video frame, prints every detection by class+score and writes an annotated PNG. Use this to diagnose why detection looks bad before changing anything.
  - `render_violations.py` — overlays a `validate_trj.py` violations CSV (`--violations-csv`, required; filterable with `--checks`) onto a video, marking each non-compliant instance in red on the frame it occurs, aligned by `round(timestamp_s * fps)`. Trajectory drawing (bumpers/IDs/trails) now lives in the pipeline's `OverlayVideoExporter` (`export.video_out`, see `vault/20_video_export.md`); point this script at that overlay `.mp4` to get warped frame + trajectories + violations. No longer reads the `.trj` (positions come from the CSV). For ego-motion runs the CSV positions are in the global stabilization frame, so pass `--transforms-csv` (the run's `export.transform_csv`) to map each mark back onto the raw video; omit it for non-stabilized runs (see `vault/05_75_mvp1_9.md`).
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
| Run the CLI (config + overrides) | `uv run tratrac VIDEO --config run.toml [--out PATH] [--conf 0.4] …` (see `tratrac.example.toml`; every value is mandatory via config or flag) |
| Smooth a tracks sidecar into a de-jittered `.trj` | `uv run tratrac-smooth TRACKS.csv --out OUT.trj [--pos-noise PX] [--jerk Q]` (needs a run with `export.tracks`; see `vault/22_smoothing.md`) |
| Dump a `.trj` | `uv run python scripts/dump_trj.py PATH [--summary] [--max-frames N]` |
| Probe the detector on a frame | `uv run python scripts/probe_detector.py VIDEO --frame N` |
| Render the overlay video (frame + trajectories) | set `export.video_out` / `--video-out OUT.mp4` on a run (see `vault/20_video_export.md`) |
| Overlay violations on a video | `uv run python scripts/render_violations.py VIDEO --violations-csv OUT.csv [--transforms-csv T.csv] [--out OUT.mp4] [--checks appearance,...]` |
| Semantically validate a `.trj` (e2e) | `uv run python scripts/validate_trj.py PATH [--violations-csv OUT.csv] [--fail-under PCT]` |

## Dependency Notes

- `torch` and `torchvision` are pinned to the **CPU index** (`https://download.pytorch.org/whl/cpu`) in `[tool.uv.sources]`. Swap to a CUDA index when a GPU is available — both packages must come from the same index or `torchvision::nms` won't register.
- `boxmot==19.x` reorganized its API; the tracker class is at `boxmot.trackers.BotSort`, not the top-level `boxmot`. **`boxmot` is AGPL-3.0** — relevant if TraTrac will be distributed.
- `ultralytics` is the **YOLOv8-VisDrone** runtime — also **AGPL-3.0**, also distribution-relevant. The dep is scoped to MVP1's emergency detector adapter (see `vault/05_mvp1.md`); when RT-DETR fine-tuning lands in MVP1.5, this dep + the `yolov8_visdrone.py` adapter file + the CLI enum value get removed in one cleanup.
- `dill` is pulled in because the `Mahadih534/YoloV8-VisDrone` checkpoint was pickled with it. Pinned explicitly so the ultralytics auto-installer doesn't re-trigger on every run.
- `transformers`, `cv2`, `boxmot`, `ultralytics`, `huggingface_hub` are configured with `follow_imports = "skip"` in mypy, so they're treated as `Any` at the third-party seam. Everything else is fully typed under strict mypy.

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
- `15_step_timing.md` — the homogeneous-port migration (batch `OrientationEstimator`), the `Timed*` decorators, and the `TimingSink` port (CSV now, telemetry later).
- `16_trj_validation.md` — the semantic e2e `.trj` validator: the three checks (continuity, orientation smoothness, kinematic plausibility), why no-ground-truth bounds what can be validated, and why a kinematic *consistency* check was rejected as tautological.
- `17_time_window.md` — `--start`/`--end` analysis-window trimming: why it lives in the video adapter (seek), the pure `FrameWindow` math, and why trimmed clips keep absolute TIMESTEPs.
- `18_timestep_precision.md` — `--timestep-precision` output decimation (export seam) **and** `--process-fps`/`input.process_fps` decode-time decimation (video adapter): both reuse the shared `DecimationGrid` (`infrastructure/cadence.py`); the export knob thins only the `.trj` (no compute saving), the processing knob skips frames with `cv2.grab()` for a real speedup at the cost of more BoT-SORT ID switches. They stack (process first, export thins further); the anchored emission-grid math and the SSAM coarseness caveat live here.
- `19_config_file.md` — the persisted run config: zero hardcoded defaults (every value from the `--config` TOML or a flag, else fail), `RunConfig.resolve` precedence/aggregation, the "every key present, off is explicit" rule, calibration one-of, and the removal of the library pixel fallback.
- `20_video_export.md` — the overlay-video output + composite exporter: why the `TrajectoryExporter` port widened to carry the `Frame`, the no-y-flip image-space drawing, cv2 behind injected seams, and composing the (decimated) `.trj` leg with a full-framerate video.
- `21_exclusion_zones.md` — image-space "do-not-analyze" pixel polygons (`analysis.exclusion_zones` sidecar JSON, off by default): a detection whose bbox is >50% covered by the polygons' **rasterized union** (`cv2.fillPoly`, so concave/overlapping zones are handled) is dropped inside the pipeline via the `DetectionMask` port, applied **after** ego-motion estimation so each zone is mapped into the current frame and **tracks the scene** under a moving drone. Zones carry a `reference_frame` (0 = static); for a moving drone they are authored on the ORB-anchor frames discovered by the headless `tratrac-scout` pass and the real run **replays** the scout's transform schedule (`ReplayEgoMotionEstimator`).
- `22_smoothing.md` — constant-acceleration **Kalman/RTS trajectory de-jittering**, two-pass: pass 1 writes a track-observation sidecar (`export.tracks`, the dual-export "B" format — raw measurements via `RecordingTracker`→`CsvTrackSink`, pipeline untouched); pass 2 (`tratrac-smooth`) runs a forward Kalman + RTS backward pass per track (`application/kalman.py`, hand-rolled numpy, no filterpy) and writes a smoothed `.trj`. Smooth position, read velocity/accel out of the filter state — the root-cause fix for the accel-noise issue. Re-tunable offline (no re-detection). The inline forward `KalmanOrientationEstimator` (RT path) is a deferred `final_polish` item.
- `final_polish.md` — backlog of deliberate "ship cheaper now, upgrade later" decisions behind stable ports (not numbered: it tracks quality upgrades to existing capabilities, not new MVP capabilities). First entry: replace the MVP2 OpenCV-ECC ego-motion adapter with SuperPoint+LightGlue via the `EgoMotionEstimator` port.

If the vault and any future code disagree, surface the conflict and ask which is authoritative before editing.

## What TraTrac Is

A vehicle tracking and trajectory-export system for **cenital/nadir aerial video**. The pipeline detects vehicles, tracks identities across occlusions and re-entries, projects to world-space metric coordinates (with multi-homography for bridges/overpasses), and exports SSAM-compatible `.trj` files for traffic safety analytics.

## Load-Bearing Architectural Invariants

These are project-defining decisions from `vault/`. Do not violate them without an explicit conversation with the user.

- **SSAM `.trj` is an export format, never the internal representation.** The canonical in-memory type is `VehicleState` (see `vault/01_architecture_principles.md`), which carries polygons, embeddings, plane/lane metadata, and uncertainty that SSAM cannot represent. Using SSAM internally would cripple future analytics.
- **Dual export architecture.** Two exporters: (A) SSAM `.trj`, (B) an extended internal format with masks/embeddings/topology/uncertainty. New analytics or debug data goes into (B), never into (A).
- **Every MVP must emit syntactically valid SSAM `.trj`, starting from MVP1.** MVPs differ in trajectory *quality*, not in whether trajectories exist. This forces orientation, front/rear point, and dimensions to be estimated very early (in MVP1, from trajectory direction + bounding-box aspect ratio).
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
