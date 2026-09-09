# TraTrac

Vehicle tracking and trajectory export for cenital / nadir aerial video. The pipeline detects vehicles, tracks identities across frames, and — via an offline pass — reconstructs kinematics and writes [SSAM](https://highways.dot.gov/safety/rsa/ssam/surrogate-safety-assessment-model-ssam) `.trj` files ready for traffic safety analytics.

## Status

**MVP1.75 and MVP1.9 shipped; MVP2 partially shipped (post-hoc world projection).** Three tools, run in sequence:

1. **`tratrac`** — perception only. Detects (**YOLOv8-VisDrone** by default, aerial-trained; a fine-tuned **YOLO-OBB** oriented detector is the MVP1.5 target, still open — an unused `rt_detr` adapter also exists, no longer the planned upgrade), tracks (**BoT-SORT**, IoU-only), optionally removes drone ego-motion (MVP1.9, off by default), and writes the raw **track record** — a Parquet file, the pipeline's only output.
2. **`tratrac-postprocess`** — offline. Optionally filters exclusion zones and optionally projects onto metric world coordinates (MVP2 Approach A: a post-hoc single homography), then runs a Kalman/RTS smoother to reconstruct kinematics and writes the binary **SSAM `.trj`**. This is the only path that produces a `.trj`.
3. **`tratrac-render`** — optional. Draws the trajectories (and, optionally, validator violations) back onto the source video.

Metric calibration is mandatory, not a default: every run requires either a direct `meters_per_pixel` value or drone-model + altitude metadata (MVP1.75 GSD calibration), so a `.trj` is never silently pixel-space passed off as metres.

### A note on the detector

At MVP1 ship, `docs/TECH_STACK.md` selected RT-DETR over YOLO for long-term aerial robustness, but COCO-pretrained RT-DETR was unable to detect aerial cars (it labelled them as `bird` and `traffic light`), and there was no GPU available in the timebox to fine-tune anything. YOLOv8-VisDrone was wired in as a separate `Detector` adapter behind the same port as an emergency measure; an unused `rt_detr` adapter still exists (selectable via `detector.name = "rt_detr"`) but is no longer the planned upgrade path — research since found RT-DETR doesn't fit this project's nadir-only footage and doesn't support oriented bounding boxes. The YOLO emergency adapter is scheduled for removal in MVP1.5 once a fine-tuned **YOLO-OBB** checkpoint exists instead — see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` for the full, current plan.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full staged roadmap and what's actually shipped vs. planned.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`torch` and `torchvision` are pinned to the CPU index in `pyproject.toml`. To use CUDA, change `[tool.uv.sources]` to point both packages at the matching CUDA index (e.g. `https://download.pytorch.org/whl/cu124`) and re-sync.

## Usage

There are **no CLI flags for run parameters** — every value comes from a TOML config, so a
`.trj` is always reconstructable from the file that produced it. Copy the template, edit it,
then run the three tools in sequence:

```bash
cp tratrac.example.toml my_run.toml   # edit: input.video, export.out, [calibration], ...
uv run tratrac --config my_run.toml                            # → track record (Parquet)
uv run tratrac-postprocess my_run.parquet --out my_run.trj     # → SSAM .trj
uv run tratrac-render my_run.mp4 --trj my_run.trj --out my_run_overlay.mp4  # optional
```

Validate a config without running anything: `uv run tratrac --config my_run.toml --check`.
Overwrite existing outputs: add `--force` to any of the three commands.

See [`CONFIG.md`](CONFIG.md) for the full config schema, `tratrac.example.toml` for a
documented template, and `CLAUDE.md`'s Commands table for every flag on all three tools
(exclusion zones, world projection, smoother tuning, violation overlays, ...).

The first run downloads the chosen detector's checkpoint into the HuggingFace cache
(YOLOv8-VisDrone ≈ 20 MB; RT-DETR-R18 ≈ 80 MB).

## Diagnostic scripts

Standalone tools that don't depend on the package internals — they keep working even when something else is broken.

```bash
# Semantically validate a .trj: continuity, orientation smoothness, kinematic plausibility.
uv run python scripts/validate_trj.py my_run.trj [--violations-csv out.csv] [--fail-under PCT]

# Per-run diagnostic figures (speed/accel/jerk, track lifespans, ...) from an outputs folder.
uv run python scripts/plot_run.py OUTPUTS_DIR [--out DIR] [--video CLIP_OR_FOLDER]

# Eyeball ORB ego-motion drift before deciding whether stabilization is worth enabling.
uv run python scripts/visualize_stabilization.py my_run.mp4 [--mask] [--no-window --save out.mp4]
```

## Architecture

Onion layers under `src/tratrac/` (`domain/` → `application/` → `infrastructure/`), plus three
Typer CLI entry points — `cli.py` (`tratrac`, perception), `cli_postprocess.py`
(`tratrac-postprocess`, smoothing/export), `cli_render.py` (`tratrac-render`, visualization).
The perception run is a **streaming per-frame pipeline** (`TrajectoryPipeline`); it computes no
kinematics and writes no `.trj` — it records raw tracked observations to a `TrackSink`
(Parquet). Kinematics (orientation, speed, acceleration) are reconstructed entirely offline by
`tratrac-postprocess`'s Kalman/RTS smoother.

For the full directory-by-directory breakdown (every adapter, every port, every design doc),
see `CLAUDE.md`'s Repository Status section and [`docs/README.md`](docs/README.md) — this
README stays intentionally brief so it doesn't drift out of sync with those.

### Load-bearing invariants

- **SSAM is an export format, never the internal representation.** The canonical in-memory type is `VehicleState`, which carries fields SSAM cannot represent (segmentation polygons, embeddings, plane metadata in later MVPs).
- **Dual export, B-first.** The track record (raw measurements now; richer fields later) is the pipeline's primary output; the SSAM `.trj` is a derived, post-hoc product built from it.
- **Every MVP emits valid SSAM from MVP1.** MVPs differ in trajectory *quality*, not whether trajectories exist — since the export inversion this takes two steps (`tratrac` → record, `tratrac-postprocess` → `.trj`) instead of one.

See `src/tratrac/domain/ARCHITECTURE.md` for the full rationale behind these invariants. The
SSAM `.trj` byte-level spec is in `src/tratrac/infrastructure/export/SSAM_FORMAT.md`, derived
from the two PDFs alongside it. Where SSAM's `Link ID` and `Lane ID` come from at each MVP is
in `docs/roadmap/road_topology.md`.

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

## Known limitations

- Coordinates are metric only when calibration is given; without world projection (MVP2 Approach A) they're still a single flat plane, not corrected for non-nadir gimbals or multi-level roads (bridges/overpasses need MVP3's multi-homography).
- Stabilization (MVP1.9) is feature-based ORB, not the target SuperPoint + LightGlue — fine for most footage, but the upgrade is tracked in `docs/BACKLOG.md` if measurement ever shows it's needed.
- Object shadows on the ground are occasionally detected as separate vehicles — a YOLOv8-VisDrone weakness, not a pipeline bug.
- No occlusion bridging: BoT-SORT is configured IoU-only with prediction-only tracks dropped from the output. Identity persistence arrives in MVP5, via DINOv3 appearance embeddings + motion-plausibility gating (not the originally planned FastReID — nadir footage discards too much of what vehicle-ReID models are trained to see; see `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md`).

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full MVP-by-MVP status table (capability ladder
vs. what's actually shipped) — kept in one place so it doesn't drift out of sync with copies
elsewhere.

## License

GPL-3.0 (see `LICENSE`). Both `boxmot` (tracker) and `ultralytics` (YOLOv8 runtime) are **AGPL-3.0**, which propagates to any distribution of the combined work. The `ultralytics` dependency is bounded to MVP1's emergency detector and is scheduled for removal in MVP1.5.

## Further reading

- `docs/README.md` — documentation map. Design docs live next to the code they describe; this
  is the index into all of them, plus the cross-cutting roadmap/architecture/tech-stack docs
  that don't belong to one module.
- `CLAUDE.md` — conventions for working in this repo with Claude Code.
