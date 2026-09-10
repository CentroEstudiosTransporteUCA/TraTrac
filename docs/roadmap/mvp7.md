# MVP 7 — PRODUCTION-GRADE ANALYTICS PLATFORM

> **Status — 🟡 Partially pulled forward; FiftyOne export and a Docker build now landed, async
> pipeline deliberately still held.** The MVP number is a capability ID, not execution order —
> see the roadmap reconciliation in `docs/ROADMAP.md`. This milestone's **Apache Parquet
> storage** was pulled forward and is already the canonical track record
> (`infrastructure/tracks/parquet.py` — see `src/tratrac/application/SMOOTHING.md`). Of the
> three items still open at the last pass: **FiftyOne visualization is now built**
> (`cli_fiftyone.py`, `tratrac-fiftyone` — see "Exploration pass" below, now updated to
> "Landed"), **Docker + CUDA has a written, tag-verified multi-stage `Dockerfile`** (not
> build-tested — no `docker` available in the environment it was written in), and the **async
> pipeline stays unbuilt by design** — the CPU timing profile below still stands as the reason
> to measure again on a GPU before designing it, not yet done.

---

## Goal

Scale:

- processing
- querying
- analytics
- deployment

---

## New Technologies

| Component | Technology |
| --- | --- |
| Storage | Apache Parquet |
| Visualization | FiftyOne |
| Runtime | Async Pipelines |
| Deployment | Docker + CUDA |

---

## Pipeline

```text
Distributed Video Processing
    ↓
Trajectory Generation
    ↓
Persistent Storage
    ↓
Analytics APIs
    ↓
Visualization Platform
```

---

## Exploration pass (Group F, `docs/IMPLEMENTATION_PLAN.md`)

### Visualization: FiftyOne — still the right call, landed

Unlike the detector/ReID/segmentation picks (RT-DETR, FastReID, SAM2), which a research pass
overturned, FiftyOne holds up on a 2026 check: it's still the actively maintained, widely
adopted (3M+ installs, Fortune 500 usage per Voxel51's own materials — treat as a vendor claim,
not independently verified) open-source tool for exactly this shape of work — curating and
debugging CV datasets, visualizing video tracking annotations and predicted vs. ground-truth
trajectories, computing/storing trajectory statistics. No better-fit alternative surfaced.

**Landed** (`src/tratrac/cli_fiftyone.py`, `tratrac-fiftyone`): a FiftyOne dataset populated
from TraTrac's existing outputs, not a new export format —
- one FiftyOne video **sample** per source clip,
- **frame-level detections** from the track record (`infrastructure/tracks/parquet.py`,
  `--record`) and/or a `.trj` (`infrastructure/export/ssam_trj.py`'s `read_trj`, `--trj`) —
  passing both adds two label fields per frame (`record_detections`/`trj_detections`) so raw
  and smoothed trajectories can be compared directly in the FiftyOne App,
- each vehicle's FiftyOne `Detection.index` carries its track/vehicle id (FiftyOne's own field
  for this, used by its video-tracking visualization).

A straightforward *reader*, not a pipeline change — the same "post-hoc tool over existing
outputs" shape as `scripts/plot_run.py` and `scripts/validate_trj.py`. `fiftyone` is an
**optional extra** (`uv sync --extra fiftyone`), not a core dependency — see `CLAUDE.md`
Dependency Notes for why, including a real `opencv-python`/`opencv-python-headless` conflict
this surfaced and how it's resolved.

Track-level fields (link/lane id, ReID merge provenance) as FiftyOne label attributes, beyond
the per-frame detections above, are **not yet added** — a natural next increment once this is
in real use.

**Not verified end-to-end in this environment**: the conversion logic
(`record_frame_detections`, `trj_frame_detections` — pure, no `fiftyone` import) is unit-tested
and was also run against this project's real `out/cruce.parquet`/`out/cruce.trj` outputs, all
the way up to the first live-`fiftyone` call. `fiftyone`'s bundled MongoDB
(`fiftyone-db`) fails to start on this NixOS sandbox (`ServiceExecutableNotFound: Could not
find mongod`) — a system/environment gap, not a code issue (nixpkgs does carry `mongodb`, but
it's SSPL-licensed/"unfree" and wasn't pulled in just to chase this). Confirm the actual
`fo.Dataset`/`fo.Sample`/`fo.launch_app` calls in `_build_dataset` work once run somewhere with
a working `mongod`.

### Runtime: measured — decode is not the bottleneck on this workload (CPU baseline)

"Async Pipelines" in the original stack table names an architecture, not a specific finding.
2026 evidence for GPU-video-inference pipelines in general is unambiguous that overlapping
decode with inference (rather than doing them serially) meaningfully improves GPU utilization
and latency — but *TraTrac didn't know whether decode-vs-inference serialization is actually
its bottleneck*, because no run had ever been profiled. That's no longer true: a real 10-second
window of the same real intersection clip Group B1 validated against
(`src/tratrac/application/REID_MERGE.md`) was run with `--timing-csv` on (CPU, the only runtime
available in this environment):

| Step | Mean | Share of measured time |
| --- | --- | --- |
| `detect` (YOLOv8-VisDrone) | 114.2 ms/frame | **87.4%** |
| `track` (BoT-SORT) | 16.4 ms/frame | 12.6% |
| `record` (Parquet write) | 0.02 ms/frame | ~0% |

The timed steps summed to 98.3% of total wall time (39.33s of ~40s) — meaning **decode +
Python loop overhead is only ~1.7%** on this CPU run. Decode is emphatically not idle-waiting on
anything here; inference (`detect`) dominates by nearly an order of magnitude over the next
largest step. **This is CPU-only evidence, not GPU evidence** — swapping the detector onto a
GPU (Group A1) would shrink `detect` by roughly an order of magnitude and could plausibly make
decode relatively significant enough to matter, which this measurement can't rule out. But it
does rule out the naive worry that decode is *already* silently serialized behind something
else on this codebase's actual per-frame loop shape — there's no such stall to find here. The
recommendation stands: re-run this same `--timing-csv` profile once a GPU is available (Group
A1) before designing an async runtime, now with a real CPU baseline to compare against instead
of zero data.

### Deployment: Docker + CUDA — a concrete multi-stage shape, landed but not build-tested

2026 best practice for a PyTorch+CUDA image is a **multi-stage build**: a `-devel` base
(compilers/headers) for the build stage, a matching `-runtime` base (no build tools, smaller)
for the final stage, copying over only the built virtualenv/artifacts — cited reductions of
~60% image size from this pattern alone. **Landed** as the repo-root `Dockerfile`:

- Build stage: `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` (tag verified against Docker
  Hub's registry as of 2026-09-10 — matches this project's `torch>=2.12.0` pin and the cu130
  index Group A1 already verified resolves for it), `uv sync` with `[tool.uv.sources]`
  rewritten at build time (via `sed`, visible in the `Dockerfile`, not baked into the checked-in
  `pyproject.toml`) to point at the CUDA torch/torchvision index instead of the CPU one this
  repo defaults to — the same manual-edit decision point Group A1 (`docs/IMPLEMENTATION_PLAN.md`)
  already identifies, reused for deployment rather than dev, and deliberately *not* the
  uv-extras approach A1 tried and rejected (bare `uv sync`/`uv run` silently resolving to CUDA
  would be exactly as unsafe in a Dockerfile's own dev-facing commands as it was found to be
  locally — the CUDA base image itself, not an extra flag, is what makes this build
  unambiguously GPU-bound).
- Runtime stage: the matching `-runtime` base (also tag-verified), `COPY --from=build` the
  built `.venv` + `src/` (the venv's editable-install `.pth` needs the source tree present, not
  just the venv), no compiler toolchain or `uv` binary shipped.
- **Not build-tested**: no `docker` available in the environment this was written in. The
  `pytorch/pytorch` devel/runtime tag pair and the `ghcr.io/astral-sh/uv` version pin were both
  confirmed to exist via their registries' own APIs (not guessed), but an actual `docker build`
  has not been run — do that before trusting this in a real deployment.
- Not designed here: which CUDA/cuDNN version to pin for the *actual deployment GPU* (this
  picks the version already validated for dev, not a specific deployment target — re-pin if
  they differ). `boxmot`/`ultralytics`'s AGPL-3.0 status is resolved, not open — TraTrac is
  GPL-3.0 and GPLv3 §13 explicitly permits the combination (see `CLAUDE.md` Dependency Notes); a
  deployed image does still carry AGPL's own network-interaction clause if the image is ever
  offered as a hosted/network service, which a Docker build is a plausible step toward.

---

## Link / Lane IDs

See `docs/roadmap/road_topology.md`.

- **Link ID** and **Lane ID** — inherited from MVP6 (Strategy A,
  hand-drawn polygons per scene).
- Possible migration to **Strategy C** (OpenStreetMap / GIS import via
  georeferenced homography) for ad-hoc scenes without per-scene
  configuration. Operational complexity is non-trivial; staying on
  Strategy A is acceptable for production if scenes are pre-configured.
