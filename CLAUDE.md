# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

MVP1 landed. The repo contains:

- `vault/` — authoritative design documents (system overview, architecture principles, coordinate systems, tech stack, MVP roadmap, SSAM format spec). The two SSAM PDFs (v1.04 + v3.0) live here as the ground truth for the binary format.
- `pyproject.toml`, `uv.lock`, `.python-version` (3.12) — uv-managed project.
- `src/tratrac/` — the implementation, organized in onion layers:
  - `domain/` — pure value objects (geometry, frame, detection, vehicle) and Protocol ports (`VideoSource`, `Detector`, `Tracker`, `TrajectoryExporter`).
  - `application/` — `OrientationEstimator` (per-track kinematics with EMA-smoothed heading) and `TrajectoryPipeline` (per-frame orchestrator).
  - `infrastructure/` — adapters: `video/opencv.py`, `detection/rt_detr.py` (HF transformers), `detection/yolov8_visdrone.py` (**MVP1 emergency default**, see `vault/05_mvp1.md`), `tracking/boxmot_bot_sort.py`, `export/ssam_trj.py` (binary v1.04 writer).
  - `cli.py` — Typer CLI entry point with `--detector {yolov8_visdrone,rt_detr}` flag. Installed script: `tratrac`.
- `tests/` — `unit/` (44 tests, fully isolated from heavy deps) and `integration/` (1 e2e smoke test, marked `@pytest.mark.slow` because it downloads the detector checkpoint on first run).
- `scripts/` — standalone diagnostic tools (pure stdlib + cv2 where possible, do **not** depend on the package internals so they work even when something is broken):
  - `dump_trj.py` — human-readable dump of a binary `.trj` (FORMAT/DIMENSIONS/TIMESTEP/VEHICLE records with totals).
  - `probe_detector.py` — runs a detector with all filters off on a single chosen video frame, prints every detection by class+score and writes an annotated PNG. Use this to diagnose why detection looks bad before changing anything.
  - `render_trajectories.py` — overlays a `.trj` on its source video, drawing bumpers + IDs + trails colored per track.

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
| Run the CLI | `uv run tratrac VIDEO --out PATH [--detector yolov8_visdrone\|rt_detr] [--conf 0.25]` |
| Dump a `.trj` | `uv run python scripts/dump_trj.py PATH [--summary] [--max-frames N]` |
| Probe the detector on a frame | `uv run python scripts/probe_detector.py VIDEO --frame N` |
| Overlay trajectories on the video | `uv run python scripts/render_trajectories.py VIDEO TRJ --out OUT.mp4` |

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
- `05_mvp1.md` … `11_mvp7.md` — staged MVPs; each adds one capability.
- `12_final_architecture.md` — end-state pipeline diagram.

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
| 2 | SuperPoint+LightGlue stabilization, single-homography world projection |
| 3 | Multi-homography + polygon-based plane assignment |
| 4 | SAM2 segmentation, mask-based orientation, dual export begins |
| 5 | FastReID + embedding memory for long-term identity persistence |
| 6 | Lane-graph topology constraints |
| 7 | Apache Parquet storage, FiftyOne visualization, async/Docker deployment |

When proposing changes, identify which MVP the work belongs to and avoid pulling capabilities forward without justification.

## Target Tech Stack

Per `vault/03_tech_stack.md`: PyTorch runtime; NVDEC+PyAV decoding; SuperPoint+LightGlue stabilization; **RT-DETR** detection (deliberately *not* YOLO — aerial robustness over speed); **SAM2** segmentation (deliberately *not* Mask R-CNN); **BoT-SORT** tracking (deliberately *not* plain SORT); FastReID; EKF motion; OpenCV multi-homography; Parquet + FiftyOne + Docker/CUDA for MVP7. The vault explains *why* each was chosen and what was rejected — preserve that reasoning when picking libraries.

Python 3.12 is the implementation language, managed with `uv`. None of the runtime ML dependencies are pinned yet — add them with `uv add` as each MVP component lands.

## Working In This Repo

- Per the user's global instructions, follow the `vault`-first workflow: inspect vault, identify conflicts, understand the goal, propose approaches with trade-offs, and ask before implementing.
- Before reporting any change as complete, run `uv run ruff format .`, `uv run ruff check .`, and `uv run mypy`. Strict mypy will reject untyped functions — annotate as you write, not after.
- When new architectural facts emerge during implementation, update the relevant `vault/*.md` file rather than scattering decisions across code comments.
