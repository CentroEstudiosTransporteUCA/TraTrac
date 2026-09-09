# Detector Choice: YOLOv8-VisDrone (MVP1 emergency) vs RT-DETR (MVP1.5, target)

> **Status — MVP1 ✅ Shipped, MVP1.5 ❌ Open.** The MVP number is a capability ID, not execution
> order — see the roadmap reconciliation in `docs/ROADMAP.md`. Detection shipped with the
> YOLOv8-VisDrone *emergency* adapter, not the planned RT-DETR (that swap is MVP1.5, still open,
> detailed in "MVP1.5 — RT-DETR fine-tune, remove the YOLOv8 emergency adapter" below).

---

## Goal

Generate:

- syntactically valid SSAM `.trj`
- basic vehicle IDs
- image-space trajectories
- approximate orientation

This is the FIRST usable system.

---

## Technologies

| Component | Technology |
| --- | --- |
| Runtime | PyTorch (CPU build for dev) |
| Detection | **YOLOv8-VisDrone** (MVP1 emergency override) |
| Tracking | BoT-SORT (boxmot, IoU-only) |
| Visualization | OpenCV |

### Detection override

`docs/TECH_STACK.md` selects RT-DETR for long-term aerial robustness. At MVP1 ship,
COCO-pretrained RT-DETR-R18 misclassified aerial cars as `bird` and `traffic light`
(confirmed via `scripts/probe_detector.py`). No GPU was available within the MVP1
timebox to fine-tune RT-DETR on aerial data, so a community YOLOv8 checkpoint
trained on VisDrone (`Mahadih534/YoloV8-VisDrone` on HuggingFace) was wired in as
a parallel `Detector` adapter, defaulted in the CLI. RT-DETR adapter coexists
unchanged. The YOLO override is contained in a single file and one CLI enum
value, so removal in MVP1.5 is mechanical:

1. delete `src/tratrac/infrastructure/detection/yolov8_visdrone.py`
2. drop the `yolov8_visdrone` value from `DetectorChoice` in `cli.py`
3. `uv remove ultralytics dill`

---

## Pipeline

```text
Video
    ↓
YOLOv8-VisDrone (or RT-DETR via --detector rt_detr)
    ↓
BoT-SORT (IoU-only)
    ↓
Orientation Approximation (EMA-smoothed heading)
    ↓
Front/Rear Estimation
    ↓
SSAM .trj Export
```

---

## Output Quality

### Available

- Vehicle IDs
- Frame-by-frame positions
- Approximate front/rear points
- Approximate dimensions
- Syntactically valid SSAM

---

### Missing

- Physically meaningful coordinates
- Multi-plane geometry
- Precise occupancy
- Long-term identity persistence

---

## Output Format

MVP1 emits binary SSAM `.trj` v1.04
(see `src/tratrac/infrastructure/export/SSAM_FORMAT.md` for the byte-level spec). Concrete MVP1 conventions:

- Endianness `L`, Units Metric, Scale 1.0.
- DIMENSIONS bounds = (0, 0, image_width, image_height).
- Y-axis flipped from image space (Cartesian Y-up).
- Link ID = 0, Lane ID = 0.
- Length / Width = bounding-box dimensions in pixels (treated as
meters — physically meaningless).
- Speed / Acceleration = pixel-displacement per second (treated as m/s, m/s² — meaningless).
- Timestep = frame_index / fps.

These numbers will parse cleanly in any SSAM 2.x / 3.0 reader.
They will not produce meaningful analytics until MVP2 lands
the homography and real metric coordinates.

> **Note (MVP1.75+ / zero-defaults):** the Scale 1.0 pixels-as-metres mode above
> is no longer a default *anywhere* — the zero-defaults refactor
> (`src/tratrac/application/CONFIG_DESIGN.md`) removed the `scale=1.0` / `meters_per_pixel=1.0`
> library fallback. The `tratrac` CLI requires explicit calibration
> (`meters_per_pixel`, or `drone_model` + `altitude_m`/`srt`) and errors out
> otherwise. To deliberately produce pixel-space output, construct the
> estimator/exporter directly and pass `scale=1.0` / `meters_per_pixel=1.0`
> yourself.

---

## Link / Lane IDs

See `docs/roadmap/road_topology.md` for the full sourcing plan.

- **Link ID** — hardcoded `0`. No road network metadata exists at MVP1.
- **Lane ID** — hardcoded `0`. No lane assignment.

`VehicleState.link_id` and `VehicleState.lane_id` exist on the domain type
with default `0`; the exporter reads them directly. Future MVPs populate
them — the exporter does not change.

---

# MVP 1.5 — RT-DETR fine-tune, remove the YOLOv8 emergency adapter

> **Status — ❌ Skipped (open).** The MVP number is a capability ID, not execution order — see
> the roadmap reconciliation in `docs/ROADMAP.md`. Leapfrogged by 1.75 + 1.9; this is the
> open detection-quality upgrade. (Details below.)

## Status

**Not started.** MVP1.75 (metric calibration, `src/tratrac/calibration/GSD_CALIBRATION.md`) was
shipped first as an independent shortcut, leapfrogging this milestone. MVP1.5
remains the open detection-quality upgrade. It is numbered before 1.75 because
that is its place in the *quality* roadmap, not the order it was delivered.

## Goal

Replace the MVP1 emergency detector with the detector the tech stack actually
selected:

- **Fine-tune RT-DETR** on aerial data (VisDrone and/or UAVDT) so it detects
  aerial vehicles instead of misclassifying them.
- **Restore RT-DETR as the default detector.**
- **Remove the YOLOv8-VisDrone emergency adapter** and its dependencies.

No change to coordinate semantics, calibration, tracking, or export. MVP1.5
improves *detection quality only*; everything downstream of the `Detector` port
is untouched.

## Why this MVP exists

`docs/TECH_STACK.md` chose RT-DETR over YOLO deliberately — aerial
robustness over raw speed. But at MVP1 ship, COCO-pretrained RT-DETR-R18
misclassified aerial cars as `bird` and `traffic light` (confirmed via
`scripts/probe_detector.py`), and no GPU was available within the MVP1 timebox
to fine-tune. A community YOLOv8 checkpoint trained on VisDrone
(`Mahadih534/YoloV8-VisDrone`) was wired in as a parallel `Detector` adapter and
defaulted in the CLI (see "MVP 1" above).

MVP1.5 is defined by removing that constraint. Its core deliverable is the
fine-tuned checkpoint that the MVP1 emergency override was a placeholder for. The
YOLOv8 path is AGPL-3.0 (`ultralytics`), so its removal also clears a
distribution-licensing concern (see the dependency notes in `CLAUDE.md`).

## Technologies

| Component | Technology |
| --- | --- |
| Detection | **RT-DETR** (fine-tuned), via HuggingFace `transformers` |
| Training data | VisDrone and/or UAVDT |
| Training compute | **GPU required** (the project torch pin is CPU-only today) |
| Eval | `scripts/probe_detector.py`, `scripts/validate_trj.py` |

The RT-DETR adapter (`src/tratrac/infrastructure/detection/rt_detr.py`) already
exists and coexists with the YOLOv8 one — MVP1.5 makes it the only adapter, not
a new one.

## Pipeline

```text
Video
    ↓
RT-DETR (fine-tuned on aerial data)   ← was: YOLOv8-VisDrone (emergency)
    ↓
BoT-SORT (IoU-only)
    ↓
Orientation Estimator (unit-aware, from MVP1.75)
    ↓
SSAM .trj Export
```

The only pipeline change from MVP1.75 is the detector box.

## Work breakdown

The milestone decomposes into three parts of very different character. **They
must land in this order** — the YOLOv8 scaffolding is the only working detector
until A+B are proven, so removing it first would leave no detector.

### Part A — Train the model (the real deliverable; currently undocumented)

This is where MVP1.5 actually lives, and the repo provides no harness for it.

1. **Secure a GPU.** `pyproject.toml` `[tool.uv.sources]` pins `torch` /
   `torchvision` to the CPU wheel index; fine-tuning on CPU is impractical.
   Switching to a CUDA index is a prerequisite (both packages must come from the
   same index or `torchvision::nms` won't register — see `CLAUDE.md`).
2. **Acquire and prepare the dataset** — VisDrone and/or UAVDT, converted to the
   format the RT-DETR training loop expects.
3. **Fine-tune RT-DETR** on aerial vehicle classes, producing a checkpoint.
4. **Validate it beats the YOLOv8 baseline** on representative aerial footage,
   using `scripts/probe_detector.py` (the same tool that confirmed COCO RT-DETR
   was broken) and end-to-end `.trj` quality via `scripts/validate_trj.py`.
5. **Publish/store the checkpoint** where the adapter can load it — an HF repo
   id (`--checkpoint`) or a local weights file.

> The fine-tuning + eval workflow is not yet designed. Treat its design as a
> sub-task of this MVP; do not assume a training script exists.

### Part B — Adapt the RT-DETR adapter to the new classes (required, not yet noted elsewhere)

`rt_detr.py` is currently **hardwired to COCO label strings**: it maps
detections through `_COCO_LABEL_TO_VEHICLE_CLASS` keyed on `"car"`,
`"motorcycle"`, `"bus"`, `"truck"`, read from `model.config.id2label`.

A VisDrone/UAVDT-fine-tuned model emits **different class indices and label
strings**. So MVP1.5 requires:

- Rewriting the label → `VehicleClass` mapping to match the fine-tuned model's
  `id2label`.
- Deciding how aerial-dataset classes that have no COCO equivalent collapse into
  the `VehicleClass` enum. VisDrone, for example, distinguishes `car`, `van`,
  `truck`, `bus`, plus non-vehicle and `tricycle`/`awning-tricycle` classes.

This is the seam where Part A meets the codebase. Neither "MVP 1" above nor
the removal note below currently mention it.

### Part C — Remove the YOLOv8 emergency scaffolding (mechanical; do last)

"MVP 1" above lists three steps; the code has **more sites** because that
note predates the zero-defaults config refactor (`src/tratrac/application/CONFIG_DESIGN.md`). Full
set:

- **Delete** `src/tratrac/infrastructure/detection/yolov8_visdrone.py`.
- **`uv remove ultralytics dill`** (`dill` exists only because the YOLOv8
  checkpoint was pickled with it).
- `application/config.py` — drop `YOLOV8_VISDRONE` from `DetectorChoice` and
  update its docstring.
- `application/config.py` — `_resolve_detector_name` currently **defaults to
  `YOLOV8_VISDRONE`** when unset. Post-removal this must default to `RT_DETR` —
  or, better, be reconsidered, since a silent detector default sits oddly against
  the project's zero-defaults stance (`src/tratrac/application/CONFIG_DESIGN.md`). Decide
  explicitly.
- `application/config.py` — the `DetectorConfig.filename` field exists only for
  the YOLOv8 adapter (`detector.filename`, "yolov8_visdrone only"). Decide
  whether to drop it.
- `cli.py` — remove the `YoloV8VisDroneDetector` import, its `DetectorChoice`
  construction branch, and (pending the decision above) the `--checkpoint-file`
  flag and `detector.filename` config key.
- `tests/` — remove or update any test referencing the YOLOv8 adapter or the
  `yolov8_visdrone` choice.
- Docs — update `CLAUDE.md` (repository status, roadmap, dependency notes) and
  this file's "MVP 1" section; check `tratrac.example.toml` for yolov8 references.

## Output Quality

### Added

- Aerial-robust detection from the architecturally-chosen detector.
- Removal of the AGPL `ultralytics` runtime dependency.

### Unchanged from MVP1.75

- Tracking (still IoU-only BoT-SORT, no ReID — that is MVP5).
- Coordinate semantics and metric calibration (MVP1.75).
- SSAM `.trj` structure and the export contract.

### Still missing (later MVPs)

- World-space coordinates / stabilisation (MVP2).
- Long-term identity persistence / ReID (MVP5).

## Acceptance criteria

This MVP is done when:

- A fine-tuned RT-DETR checkpoint detects aerial vehicles on representative
  footage and **measurably outperforms** the YOLOv8-VisDrone baseline (via
  `scripts/probe_detector.py` per-frame detections and `scripts/validate_trj.py`
  end-to-end compliance).
- RT-DETR is the default detector, and its adapter maps the fine-tuned model's
  classes correctly to `VehicleClass`.
- The YOLOv8 adapter, the `yolov8_visdrone` enum value, and the `ultralytics` /
  `dill` dependencies are removed, with the full cleanup-site list above
  addressed.
- `uv run ruff format .`, `uv run ruff check .`, `uv run mypy`, and
  `uv run pytest` all pass.
- `CLAUDE.md` and this file reflect that the emergency override is gone.

## Open questions (resolve before starting)

- **Is a GPU available now?** It gates Part A entirely.
- **VisDrone, UAVDT, or both?** Affects class taxonomy and the Part B mapping.
- **What is the design of the fine-tuning + eval workflow?** No harness exists;
  this needs its own design pass.
