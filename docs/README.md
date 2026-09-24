# Documentation Map

Design docs used to live in one centralized `vault/` folder. They now live **next to the code
they describe** — this file is the index into all of them, plus the handful of docs that are
genuinely cross-cutting (project philosophy, an end-state diagram, the full tech stack) and have
no single code location to sit beside.

Feature tracking and status live in
[GitHub Issues](https://github.com/CentroEstudiosTransporteUCA/TraTrac/issues) and the org's
[Traffic Analysis Pipeline project board](https://github.com/orgs/CentroEstudiosTransporteUCA/projects/1),
not in a checklist doc.

Read `docs/ROADMAP.md` first for the project's status; everything else is detail behind it.

## Cross-cutting (this folder)

| Doc | What it covers |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md) | System objective and core philosophy — the project's north star, not a task list. Start here. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The end-state pipeline diagram (final architecture vision). |
| [`PIPELINE_STAGES.md`](PIPELINE_STAGES.md) | Which stages always run vs. are optional, and for the optional ones, which must stay in the live `tratrac` pass vs. can be deferred to post-processing — the single source of truth for this, not duplicated elsewhere. |
| [`TECH_STACK.md`](TECH_STACK.md) | Full stack, layer by layer, with research-backed comparisons and citations for each pick — what's confirmed (BoT-SORT, SuperPoint+LightGlue, multi-homography), what changed after review (detection: YOLO-OBB not RT-DETR; ReID: DINOv3 not FastReID; segmentation: SAM 3 not SAM2; decode: TorchCodec not PyAV), and what's not yet adopted (SAM 3, DINOv3 ReID, Lane Graph, KalmanNet, CVAT). Already-shipped choices are documented next to their adapters instead — see `DETECTOR_CHOICE.md` and `TRACKER_CHOICE.md` in the table below. |

Feature/task status (what's built, what's blocked, what's next) lives in
[GitHub Issues](https://github.com/CentroEstudiosTransporteUCA/TraTrac/issues) and the org's
[Traffic Analysis Pipeline project board](https://github.com/orgs/CentroEstudiosTransporteUCA/projects/1).
Advisory "should we do this" content (build-vs-buy audits, alternative approaches not currently
planned) lives in [GitHub Discussions](https://github.com/CentroEstudiosTransporteUCA/TraTrac/discussions).
Neither lives in a repo file.

## Repo-root

| Doc | What it covers |
| --- | --- |
| [`../README.md`](../README.md) | Install, usage, CLI options — the user-facing entry point. |
| [`../CLAUDE.md`](../CLAUDE.md) | Conventions for working in this repo with an AI coding agent. |
| [`../DEPLOYMENT.md`](../DEPLOYMENT.md) | Docker multi-stage build for deployment | `../Dockerfile` |

## Design docs next to code (`src/tratrac/`)

| Doc | Covers | Code it sits beside |
| --- | --- | --- |
| [`domain/ARCHITECTURE.md`](../src/tratrac/domain/ARCHITECTURE.md) | Why SSAM is never the internal representation; the canonical `VehicleState`; dual-export strategy | `domain/vehicle.py`, `domain/ports.py` |
| [`application/WORLD_PROJECTION.md`](../src/tratrac/application/WORLD_PROJECTION.md) | Coordinate systems, multi-homography rationale, and the shipped MVP2 Approach A post-hoc single-homography projector | `application/coordinate_transforms.py`, `domain/world.py`, `infrastructure/world/calibration.py` |
| [`application/PROGRESS_REPORTING.md`](../src/tratrac/application/PROGRESS_REPORTING.md) | The `ProgressReporter` output port and `ProgressEvent` family | `application/progress.py`, `domain/progress.py`, `infrastructure/progress/` |
| [`application/CONFIG_DESIGN.md`](../src/tratrac/application/CONFIG_DESIGN.md) | Why the run config has zero hardcoded defaults; resolution model; `--check` | `application/config.py` |
| [`application/EXCLUSION_ZONES.md`](../src/tratrac/application/EXCLUSION_ZONES.md) | Post-hoc, track-aware "do-not-analyze" polygons | `application/exclusion.py`, `domain/exclusion.py`, `infrastructure/exclusion/` |
| [`infrastructure/background/BACKGROUND_ZONES.md`](../src/tratrac/infrastructure/background/BACKGROUND_ZONES.md) | Operator-authored ORB masking (`tratrac-preprocess`'s detector-free pre-pass), no live detections needed | `domain/background.py`, `infrastructure/background/`, `infrastructure/zones.py`, `cli_preprocess.py` |
| [`application/ROAD_GRAPH.md`](../src/tratrac/application/ROAD_GRAPH.md) | Post-hoc, per-observation Link ID / Lane ID classification from hand-drawn polygons | `application/road_graph.py`, `domain/road_graph.py`, `infrastructure/road_graph/` |
| [`application/AUTO_CALIBRATION.md`](../src/tratrac/application/AUTO_CALIBRATION.md) | Automatic calibration-point proposal from road markings (Canny+Hough, proposal-only — a human confirms and supplies world coords) | `scripts/propose_calibration.py` |
| [`application/SMOOTHING.md`](../src/tratrac/application/SMOOTHING.md) | Constant-acceleration Kalman/RTS trajectory de-jittering (the only `.trj` path) | `application/kalman.py`, `application/track_smoothing.py` |
| [`application/KALMANNET_DESIGN.md`](../src/tratrac/application/KALMANNET_DESIGN.md) | Research/design spike (Group E, no code yet): learned-Kalman-gain adaptive noise, reconsidering MAML-KalmanNet vs. plain unsupervised KalmanNet trained on TraTrac's own recorded tracks | (design-only; targets `application/kalman.py`) |
| [`application/REID_MERGE.md`](../src/tratrac/application/REID_MERGE.md) | Offline ReID fragment-merge decision: motion-plausibility gating (Kalman extrapolation) + appearance scoring; embed stage (DINOv3) not yet built | `application/reid_merge.py`, `application/kalman.py` (`from_state`/`predict`), `infrastructure/reid/` |
| [`application/FOOTPRINT.md`](../src/tratrac/application/FOOTPRINT.md) | Footprint sidecar: mask-derived vehicle dimensions, reusing Group A's OBB slot; segmentation stage (SAM 3) not yet built | `infrastructure/tracks/footprint_parquet.py`, `domain/geometry.py` (`oriented_extent`) |
| [`application/RESEARCH_NOTES.md`](../src/tratrac/application/RESEARCH_NOTES.md) | External literature review backing the smoothing + stabilization design | (supports `application/kalman.py` + ego-motion masking) |
| [`infrastructure/detection/DETECTOR_CHOICE.md`](../src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md) | MVP1 YOLOv8-VisDrone emergency adapter + the open MVP1.5 YOLO-OBB fine-tune plan (replanned from an earlier RT-DETR target) | `infrastructure/detection/rt_detr.py`, `yolov8_visdrone.py` |
| [`infrastructure/tracking/TRACKER_CHOICE.md`](../src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md) | Why BoT-SORT over plain SORT; current IoU-only state pending MVP5 ReID | `infrastructure/tracking/boxmot_bot_sort.py` |
| [`infrastructure/video/EGO_MOTION.md`](../src/tratrac/infrastructure/video/EGO_MOTION.md) | MVP1.9 keyframe-anchored ORB ego-motion estimator design | `infrastructure/video/ego_motion_orb.py` |
| [`infrastructure/video/TIME_WINDOW.md`](../src/tratrac/infrastructure/video/TIME_WINDOW.md) | `--start`/`--end` analysis-window trimming | `infrastructure/video/window.py`, `opencv.py` |
| [`infrastructure/transform/TRANSFORM_SINK.md`](../src/tratrac/infrastructure/transform/TRANSFORM_SINK.md) | The unified transforms file: one JSON-Lines row model (`frame_index`, `zone`, `function`) covering ego-motion, GSD scale, and world-projection homography, owned entirely by `tratrac-preprocess` | `infrastructure/transform/recording.py`, `sink.py`, `records.py` |
| [`infrastructure/timing/STEP_TIMING.md`](../src/tratrac/infrastructure/timing/STEP_TIMING.md) | Per-step latency profiling decorators | `infrastructure/timing/decorators.py`, `csv.py` |
| [`infrastructure/TIMESTEP_PRECISION.md`](../src/tratrac/infrastructure/TIMESTEP_PRECISION.md) | Export-cadence vs processing-cadence decimation | `infrastructure/cadence.py`, `export/decimating.py` |
| [`infrastructure/export/SSAM_FORMAT.md`](../src/tratrac/infrastructure/export/SSAM_FORMAT.md) | SSAM `.trj` byte-level spec (+ the two authoritative PDFs alongside it) | `infrastructure/export/ssam_trj.py` |
| [`infrastructure/export/VIDEO_EXPORT.md`](../src/tratrac/infrastructure/export/VIDEO_EXPORT.md) | Post-hoc trajectory overlay rendering (`tratrac-render`) | `infrastructure/export/overlay_video.py` |
| [`POSTPROCESS_ORDER.md`](../src/tratrac/POSTPROCESS_ORDER.md) | The fixed stage order inside `cli_postprocess.postprocess` (footprint → ReID-merge → exclusion → plane/link/lane → projection → smoothing → export) and why it can't be reshuffled | `cli_postprocess.py` |
| [`FIFTYONE.md`](../src/tratrac/FIFTYONE.md) | Building a FiftyOne dataset from existing outputs for interactive visual QA, the MongoDB-on-Linux workaround it needed | `cli_fiftyone.py` |
| [`calibration/GSD_CALIBRATION.md`](../src/tratrac/calibration/GSD_CALIBRATION.md) | MVP1.75 ground-sample-distance metric calibration from drone metadata | `calibration/gsd.py`, `drone_specs.py`, `srt_parser.py` (called from `cli_preprocess.py`'s `estimate` subcommand — no dedicated scale sidecar anymore, see `TRANSFORM_SINK.md`) |
| [`CHECK_COMMAND.md`](../src/tratrac/CHECK_COMMAND.md) | Scope of `tratrac --check` (validate a config without running) | `cli.py` |

## Design docs next to code (outside `src/`)

| Doc | Covers | Code it sits beside |
| --- | --- | --- |
| [`../scripts/validate_trj.md`](../scripts/validate_trj.md) | The semantic e2e `.trj` validator: the three checks, why there's no ground truth | `scripts/validate_trj.py` |

If you add a new design doc, put it next to the code it describes and add a row here.
