# TraTrac

Vehicle tracking and trajectory export for cenital / nadir aerial video. The pipeline detects vehicles, maintains identities across frames, estimates orientation, and writes [SSAM](https://highways.dot.gov/safety/rsa/ssam/surrogate-safety-assessment-model-ssam) `.trj` files ready for traffic safety analytics.

## Status

**MVP1 — basic SSAM trajectory generation.** The first runnable end-to-end pipeline:

- **YOLOv8-VisDrone** detection (community checkpoint from HuggingFace, fine-tuned on aerial drone imagery) + **BoT-SORT** tracking (IoU-only).
- Approximate orientation: motion-magnitude-weighted EMA of velocity direction with cached last-good heading, falling back to bbox major axis for freshly-seen stationary tracks.
- Binary **SSAM `.trj` v1.04** output (FORMAT → DIMENSIONS → TIMESTEP → VEHICLE records).
- Image-space coordinates — syntactically valid SSAM, **not yet physically meaningful**. Real metric coordinates land in MVP2 (single homography); multi-plane geometry in MVP3.

### A note on the detector

The vault (`vault/03_tech_stack.md`) selects RT-DETR over YOLO for long-term aerial robustness. At MVP1 ship the COCO-pretrained RT-DETR was unable to detect aerial cars (it labelled them as `bird` and `traffic light` — verifiable with `scripts/probe_detector.py`), and there was no GPU available in the timebox to fine-tune. YOLOv8-VisDrone is wired in as a separate `Detector` adapter behind the same port; RT-DETR coexists unchanged. The YOLO override is a single file + one CLI enum value + two extra dependencies, scheduled for removal in MVP1.5 once a fine-tuned RT-DETR checkpoint exists.

See `vault/05_mvp1.md` through `vault/11_mvp7.md` for the full staged roadmap.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`torch` and `torchvision` are pinned to the CPU index in `pyproject.toml`. To use CUDA, change `[tool.uv.sources]` to point both packages at the matching CUDA index (e.g. `https://download.pytorch.org/whl/cu124`) and re-sync.

## Usage

```bash
uv run tratrac path/to/video.mp4 --out out/trajectory.trj
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--out / -o` | required | Output `.trj` path. |
| `--detector` | `yolov8_visdrone` | `yolov8_visdrone` (default, aerial-trained) or `rt_detr` (HF transformers, COCO weights — currently weak on aerial). |
| `--conf` | `0.25` | Detection confidence threshold (0–1). |
| `--checkpoint` | depends on detector | YOLO: HF repo id (default `Mahadih534/YoloV8-VisDrone`). RT-DETR: HF checkpoint name (default `PekingU/rtdetr_r18vd`). |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps`. |

The first run downloads the chosen detector's checkpoint into the HuggingFace cache (YOLOv8-VisDrone ≈ 20 MB; RT-DETR-R18 ≈ 80 MB).

## Diagnostic scripts

Standalone tools that don't depend on the package internals — they keep working even when something else is broken.

```bash
# Dump a binary .trj into human-readable text.
uv run python scripts/dump_trj.py path/to/file.trj                  # full dump
uv run python scripts/dump_trj.py path/to/file.trj --summary        # totals only
uv run python scripts/dump_trj.py path/to/file.trj --max-frames 50  # first 50 timesteps

# Probe a detector on one chosen video frame: every detection, every class, every
# score, no filtering. Also writes an annotated PNG alongside the video.
uv run python scripts/probe_detector.py path/to/video.mp4 --frame 1000

# Overlay a .trj on its source video — bumpers, IDs, trails coloured per track.
uv run python scripts/render_trajectories.py path/to/video.mp4 path/to/file.trj \
    --out path/to/overlay.mp4 [--start SEC --end SEC --trail FRAMES]
```

## Architecture

Onion layers under `src/tratrac/`:

- `domain/` — Frozen value objects (`Point2D`, `Vector2D`, `Heading`, `Dimensions`, `BoundingBox`, `Frame`, `VehicleState`, …) and Protocol ports (`VideoSource`, `Detector`, `Tracker`, `TrajectoryExporter`). Pure Python — no framework imports.
- `application/` — `OrientationEstimator` (motion-weighted EMA on heading, last-good cache, bbox fallback) and `TrajectoryPipeline` (per-frame orchestrator). Depends only on domain types.
- `infrastructure/` — adapters behind the ports:
  - `video/opencv.py` — `cv2.VideoCapture` reader.
  - `detection/yolov8_visdrone.py` — **MVP1 emergency default.** Wraps `Mahadih534/YoloV8-VisDrone`.
  - `detection/rt_detr.py` — HuggingFace `transformers` RT-DETR. Coexists; selectable via `--detector rt_detr`.
  - `tracking/boxmot_bot_sort.py` — `boxmot.trackers.BotSort` (IoU-only; ReID arrives in MVP5).
  - `export/ssam_trj.py` — binary SSAM `.trj` v1.04 writer (`struct`-based, no external deps).
- `cli.py` — Typer CLI installed as the `tratrac` script.

### Load-bearing invariants

- **SSAM is an export format, never the internal representation.** The canonical in-memory type is `VehicleState`, which carries fields SSAM cannot represent (segmentation polygons, embeddings, plane metadata in later MVPs).
- **Dual export.** Two exporter ports — SSAM (`.trj`) plus a richer internal format that arrives with segmentation in MVP4.
- **Every MVP emits valid SSAM from MVP1.** MVPs differ in trajectory *quality*, not whether trajectories exist.

The full design rationale lives in `vault/00_system_overview.md` through `vault/13_road_topology.md`. The SSAM `.trj` byte-level spec is in `vault/04_ssam_format.md`, derived from the two PDFs alongside it. Where SSAM's `Link ID` and `Lane ID` come from at each MVP is in `vault/13_road_topology.md`.

## Development

```bash
uv run ruff format .      # format (tabs, double quotes, line 100)
uv run ruff check .       # lint
uv run mypy               # type-check (strict)
uv run pytest             # all tests
uv run pytest tests/unit  # unit only, fast
uv run pytest -m slow     # end-to-end smoke (downloads a detector checkpoint)
```

All checked-in code passes ruff + strict mypy. Indentation is tabs.

## Known limitations of MVP1

- Trajectory numbers (length, speed, accel) are in pixel units labelled as meters — physically meaningless. Fixed by MVP2's homography.
- Stationary vehicles can still show some residual heading flicker when bbox jitter is comparable to the EMA's full-trust speed threshold. Worse on motorcycles than cars.
- Object shadows on the ground are occasionally detected as separate vehicles — a YOLOv8-VisDrone weakness, not a pipeline bug.
- No occlusion bridging: BoT-SORT is configured IoU-only with prediction-only tracks dropped from the output. Identity persistence arrives in MVP5 with FastReID.

## Roadmap

| MVP | Adds |
| --- | --- |
| 1 (this) | YOLOv8-VisDrone (emergency override) + BoT-SORT, EMA orientation, image-space SSAM `.trj` |
| 1.5 | Fine-tune RT-DETR on VisDrone / UAVDT, restore RT-DETR as default, drop ultralytics |
| **1.75** | **Metric sizes and speeds from drone metadata.** GSD calibration from sensor + focal + altitude; populates `DIMENSIONS.Scale` and writes `Length` / `Width` / `Speed` / `Acceleration` in real units. No homography needed for hovering, nadir drone footage. See `vault/05_5_mvp1_75.md`. |
| 2 | SuperPoint + LightGlue stabilization, single-homography world projection (handles moving drones, non-nadir gimbals, fixed cameras without telemetry) |
| 3 | Multi-homography + polygon-based plane assignment (bridges / overpasses) + Link ID assignment from hand-drawn polygons (see `vault/13_road_topology.md`) |
| 4 | SAM2 segmentation, mask-based orientation, dual export |
| 5 | FastReID + embedding memory for long-term identity persistence |
| 6 | Lane-graph topology constraints + Lane ID assignment from hand-drawn lane polygons |
| 7 | Apache Parquet storage, FiftyOne visualization, async / Docker deployment |

## License

GPL-3.0 (see `LICENSE`). Both `boxmot` (tracker) and `ultralytics` (YOLOv8 runtime) are **AGPL-3.0**, which propagates to any distribution of the combined work. The `ultralytics` dependency is bounded to MVP1's emergency detector and is scheduled for removal in MVP1.5.

## Further reading

- `vault/` — full design knowledge, MVP roadmap, and the authoritative SSAM PDFs.
- `CLAUDE.md` — conventions for working in this repo with Claude Code.
