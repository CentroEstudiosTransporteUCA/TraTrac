# Detector Choice: YOLOv8-VisDrone AABB (MVP1 emergency) vs YOLO-OBB (MVP1.5, target)

> **Status — MVP1 ✅ Shipped, MVP1.5 ❌ Open, replanned.** The MVP number is a capability ID, not
> execution order — see the roadmap reconciliation in `docs/ROADMAP.md`. Detection shipped with
> the YOLOv8-VisDrone *emergency* axis-aligned-box adapter. **The original MVP1.5 plan (fine-tune
> RT-DETR) was superseded** after research found RT-DETR doesn't fit this project's nadir-only use
> case and doesn't support oriented bounding boxes (OBB) — see `docs/TECH_STACK.md`'s Detection
> section for the full comparison and sources. The replanned MVP1.5, detailed below, fine-tunes a
> **YOLO-OBB** model instead — same detector *family* as the MVP1 emergency adapter, upgraded to
> report each vehicle's orientation directly instead of via the downstream EMA-heading hack.

---

## MVP 1

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

At the time, `docs/TECH_STACK.md` selected RT-DETR for long-term aerial robustness. At MVP1 ship,
COCO-pretrained RT-DETR-R18 misclassified aerial cars as `bird` and `traffic light`
(confirmed via `scripts/probe_detector.py`). No GPU was available within the MVP1
timebox to fine-tune RT-DETR on aerial data, so a community YOLOv8 checkpoint
trained on VisDrone (`Mahadih534/YoloV8-VisDrone` on HuggingFace) was wired in as
a parallel `Detector` adapter, defaulted in the CLI. RT-DETR adapter coexists
unchanged. **That RT-DETR plan has since been superseded** (see the status banner
above and `docs/TECH_STACK.md`); MVP1.5's replacement is a YOLO-OBB fine-tune, not
a restoration of RT-DETR. The YOLOv8-VisDrone axis-aligned adapter is still
contained in a single file and one CLI enum value, so its removal in MVP1.5
remains mechanical once the OBB adapter is proven:

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

# MVP 1.5 — YOLO-OBB fine-tune, oriented detection replaces the axis-aligned emergency adapter

> **Status — ❌ Skipped (open), replanned.** The MVP number is a capability ID, not execution
> order — see the roadmap reconciliation in `docs/ROADMAP.md`. Leapfrogged by 1.75 + 1.9; this is
> the open detection-quality upgrade. **This plan replaces an earlier one that targeted a from-scratch
> RT-DETR fine-tune** — see "Why this replan happened" below and `docs/TECH_STACK.md`'s Detection
> section for the full comparison and sources.

## Status

**Not started.** MVP1.75 (metric calibration, `src/tratrac/calibration/GSD_CALIBRATION.md`) was
shipped first as an independent shortcut, leapfrogging this milestone. MVP1.5
remains the open detection-quality upgrade. It is numbered before 1.75 because
that is its place in the *quality* roadmap, not the order it was delivered.

## Goal

Replace the MVP1 emergency detector with an **oriented-box (OBB) detector**,
fine-tuned for TraTrac's actual footage regime:

- **Fine-tune a YOLO-OBB model** (`yolo11-obb` or `yolo26-obb`) on drone-altitude,
  nadir-matched data so it both detects aerial vehicles reliably *and* reports
  each vehicle's heading directly, rather than needing it inferred downstream.
- **Make YOLO-OBB the default detector.**
- **Remove the YOLOv8-VisDrone axis-aligned emergency adapter** once the OBB
  adapter is proven.

Unlike the original plan, this one **does** change something downstream of
detection: the orientation pipeline needs to consume the detected angle. See
"Downstream integration" below — this is not detection-quality-only work.

## Why this replan happened

The original MVP1.5 chose RT-DETR because `docs/TECH_STACK.md` picked it over
YOLO for "aerial robustness" in general — dense, oblique, cluttered scenes.
Re-examined against TraTrac's *actual* use case (consistently nadir/cenital
street video, not arbitrary oblique aerial imagery), that rationale turned out
to be weaker than assumed, and a bigger problem surfaced: **RT-DETR does not
support oriented bounding boxes** (confirmed against the Hugging Face
`transformers` implementation), so it can't solve TraTrac's actual persistent
weak point — orientation estimation currently relies on an EMA-smoothed-heading
hack (`OrientationEstimator`) that the validator specifically catches producing
sudden front/rear flips. An OBB detector reports orientation at detection time,
solving it at the source. `ultralytics` YOLO has first-class OBB support and is
already TraTrac's dependency (no new training framework needed), which also
makes this a materially cheaper integration than the from-scratch HuggingFace
RT-DETR harness the original plan required. Full comparison, alternatives
considered, and sources: `docs/TECH_STACK.md`.

**This reopens the licensing trade-off, it doesn't resolve it.** The original
plan's other selling point was clearing the AGPL-3.0 `ultralytics` dependency
(see `CLAUDE.md` Dependency Notes). Staying on `ultralytics` for OBB does not
clear that — it deepens it. Needs an explicit decision before distribution, not
a silent carry-over.

## Technologies

| Component | Technology |
| --- | --- |
| Detection | **YOLO-OBB** (`yolo11-obb` or `yolo26-obb`, fine-tuned), via `ultralytics` |
| Training data | **UAV-OBB** (nadir, 75–108m altitude, already YOLO-OBB label format) — **DroneVehicle** (28K pairs, more robust) if UAV-OBB alone underperforms |
| Training compute | **GPU required** (the project torch pin is CPU-only today) |
| Eval | `scripts/probe_detector.py` (**currently missing from the repo — must be restored or replaced first**, see below), `scripts/validate_trj.py` |

The `ultralytics` package is already a TraTrac dependency (the MVP1 emergency
adapter uses it) — this is a task/checkpoint swap within the same library, not
a new one.

## Pipeline

```text
Video
    ↓
YOLO-OBB (fine-tuned, oriented boxes)   ← was: YOLOv8-VisDrone axis-aligned (emergency)
    ↓
BoT-SORT (OBB-aware tracking — boxmot supports this natively, see TRACKER_CHOICE.md)
    ↓
Orientation from detected angle (replaces/supplements the EMA heading estimator)
    ↓
SSAM .trj Export
```

## Downstream integration this requires (not detection-quality-only)

Unlike the original RT-DETR plan, this one has real reach beyond the `Detector`
port:

- `VehicleState`/the orientation pipeline needs to accept a per-detection angle
  instead of, or blended with, the EMA-smoothed heading estimate — this is
  where the actual payoff is, so it should be designed properly rather than
  bolted on.
- Tracking: `boxmot` natively supports OBB tracking (confirmed, not assumed —
  see `TRACKER_CHOICE.md`), so the oriented box can survive into the track
  rather than being collapsed to an axis-aligned box first.

## Work breakdown

The milestone decomposes into parts of very different character. **They must
land in this order** — the YOLOv8 scaffolding is the only working detector
until the OBB adapter is proven, so removing it first would leave no detector.

### Part 0 — Restore the eval tool (blocks everything else)

`scripts/probe_detector.py` is referenced throughout this doc and the original
plan as the tool that validates detector quality before/after a swap, but it
does not currently exist in the repo. Credible before/after comparison for this
MVP depends on it (or an equivalent) existing first.

### Part A — Train the model (the real deliverable; currently undocumented)

This is where MVP1.5 actually lives, and the repo provides no harness for it.

1. **Secure a GPU.** `pyproject.toml` `[tool.uv.sources]` pins `torch` /
   `torchvision` to the CPU wheel index; fine-tuning on CPU is impractical.
   Switching to a CUDA index is a prerequisite (both packages must come from the
   same index or `torchvision::nms` won't register — see `CLAUDE.md`).
2. **Acquire the UAV-OBB dataset** (Mendeley Data) — already in YOLO-OBB label
   format, so no conversion step is needed, unlike the original VisDrone-for-RT-DETR
   plan. Layer in DroneVehicle if more data/robustness is needed.
3. **Fine-tune** `yolo11-obb`/`yolo26-obb` (Ultralytics-pretrained on DOTA as a
   starting point — note DOTA is satellite/very-high-altitude imagery, a real
   domain gap from drone altitude, so treat it as initialization, not a
   substitute for fine-tuning on UAV-OBB/DroneVehicle) on the vehicle classes.
4. **Validate it beats the YOLOv8-VisDrone axis-aligned baseline** on
   representative real footage, using `scripts/probe_detector.py` (Part 0) and
   end-to-end `.trj` quality via `scripts/validate_trj.py`.
5. **Publish/store the checkpoint** where the adapter can load it.

> The fine-tuning + eval workflow is not yet designed. Treat its design as a
> sub-task of this MVP; do not assume a training script exists.

### Part B — Build the YOLO-OBB adapter and class mapping (required, not yet noted elsewhere)

No OBB adapter exists yet behind the `Detector` port — this is new code, not a
modification of `rt_detr.py` (which stays as an unused axis-aligned option, see
"Open questions"). It needs to:

- Wrap `ultralytics`'s OBB task (distinct from its detection task) and surface
  the per-detection orientation angle, not just a box.
- Map UAV-OBB's six classes (`bike`, `bus`, `car`, `other_vehicle`, `taxi`,
  `truck`) — or DroneVehicle's five (`car`, `truck`, `bus`, `van`,
  `freight-car`) if that dataset is used — into TraTrac's `VehicleClass` enum,
  the same *kind* of mapping work the original RT-DETR plan required, against a
  better-matched taxonomy this time.

### Part C — Remove the YOLOv8 emergency scaffolding (mechanical; do last)

"MVP 1" above lists three steps; the code has **more sites** because that
note predates the zero-defaults config refactor (`src/tratrac/application/CONFIG_DESIGN.md`). Full
set:

- **Delete** `src/tratrac/infrastructure/detection/yolov8_visdrone.py`.
- **`uv remove dill`** (existed only because the YOLOv8-VisDrone checkpoint was
  pickled with it). **`ultralytics` itself is NOT removed** — the OBB adapter
  still needs it, unlike the original plan where RT-DETR would have let it go.
- `application/config.py` — drop `YOLOV8_VISDRONE` from `DetectorChoice` and
  update its docstring; add the new OBB choice.
- `application/config.py` — `_resolve_detector_name` currently **defaults to
  `YOLOV8_VISDRONE`** when unset. Post-removal this must default to the new
  OBB choice — or, better, be reconsidered, since a silent detector default
  sits oddly against the project's zero-defaults stance
  (`src/tratrac/application/CONFIG_DESIGN.md`). Decide explicitly.
- `application/config.py` — the `DetectorConfig.filename` field exists only for
  the YOLOv8 adapter (`detector.filename`, "yolov8_visdrone only"). Decide
  whether to drop it or repurpose it for the OBB checkpoint.
- `cli.py` — remove the `YoloV8VisDroneDetector` import and its `DetectorChoice`
  construction branch.
- `tests/` — remove or update any test referencing the YOLOv8 adapter or the
  `yolov8_visdrone` choice.
- Docs — update `CLAUDE.md` (repository status, roadmap, dependency notes) and
  this file's "MVP 1" section; check `tratrac.example.toml` for yolov8 references.

## Output Quality

### Added

- Nadir-matched, aerial-robust detection.
- Per-vehicle orientation from the detector itself, not an EMA heuristic.

### Unchanged

- Tracking algorithm (still BoT-SORT, IoU-only appearance branch — ReID is MVP5).
- Coordinate semantics and metric calibration (MVP1.75).
- SSAM `.trj` structure and the export contract.
- The AGPL `ultralytics` dependency stays (see "Why this replan happened" above).

### Still missing (later MVPs)

- World-space coordinates / stabilisation (MVP2, partially shipped — see `application/WORLD_PROJECTION.md`).
- Long-term identity persistence / ReID (MVP5).

## Acceptance criteria

This MVP is done when:

- `scripts/probe_detector.py` exists again and confirms the fine-tuned
  YOLO-OBB checkpoint **measurably outperforms** the YOLOv8-VisDrone axis-aligned
  baseline, corroborated by `scripts/validate_trj.py` end-to-end compliance.
- YOLO-OBB is the default detector, its adapter maps the fine-tuned model's
  classes correctly to `VehicleClass`, and its orientation angle flows into
  `VehicleState` instead of (or blended with) the EMA heading estimate.
- The YOLOv8-VisDrone axis-aligned adapter, the `yolov8_visdrone` enum value,
  and the `dill` dependency are removed, with the full cleanup-site list above
  addressed. `ultralytics` itself stays.
- `uv run ruff format .`, `uv run ruff check .`, `uv run mypy`, and
  `uv run pytest` all pass.
- `CLAUDE.md` and this file reflect that the emergency override is gone.

## Open questions (resolve before starting)

- **What happens to the unused `rt_detr.py` adapter?** It's not part of this
  plan anymore. Keep it as a dormant alternative behind the `Detector` port, or
  remove it as dead code — a call for whoever picks this up, not decided here.

- **UAV-OBB alone, or layer in DroneVehicle too?** Affects class taxonomy (six
  classes vs. five) and training time — start with UAV-OBB (small, fast, exact
  altitude match) and only add DroneVehicle if it underperforms.
- **What is the design of the fine-tuning + eval workflow?** No harness exists;
  this needs its own design pass.
