# MVP 7 — PRODUCTION-GRADE ANALYTICS PLATFORM

> **Status — 🟡 Partially pulled forward; exploration pass done, one finding now measured
> against real footage (Group F, no code).** The MVP number is a capability ID, not execution
> order — see the roadmap reconciliation in `docs/ROADMAP.md`. This milestone's **Apache
> Parquet storage** was pulled forward and is already the canonical track record
> (`infrastructure/tracks/parquet.py` — see `src/tratrac/application/SMOOTHING.md`). FiftyOne
> visualization, async pipelines, and Docker/CUDA deployment are **not implemented**, but
> `docs/IMPLEMENTATION_PLAN.md` Group F's "needs its own exploration/design pass" has now had a
> first pass — see "Exploration pass"
> below for each of the three, including a concrete Docker build shape and a reasoned
> recommendation to measure before building the async runtime.

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

### Visualization: FiftyOne — still the right call, no pivot needed

Unlike the detector/ReID/segmentation picks (RT-DETR, FastReID, SAM2), which a research pass
overturned, FiftyOne holds up on a 2026 check: it's still the actively maintained, widely
adopted (3M+ installs, Fortune 500 usage per Voxel51's own materials — treat as a vendor claim,
not independently verified) open-source tool for exactly this shape of work — curating and
debugging CV datasets, visualizing video tracking annotations and predicted vs. ground-truth
trajectories, computing/storing trajectory statistics. No better-fit alternative surfaced.

**Integration shape** (not built): a FiftyOne dataset populated from TraTrac's existing
outputs, not a new export format —
- one FiftyOne video **sample** per source clip,
- **frame-level detections** from the track record (`infrastructure/tracks/parquet.py`) or a
  `.trj` (`infrastructure/export/ssam_trj.py`'s `read_trj`) — both already have everything
  needed (bbox/OBB, track id, per-frame state),
- track-level fields (link/lane id, ReID merge provenance once `--reid-merge` is used) attached
  as FiftyOne label attributes rather than invented new fields.

This is a straightforward *reader*, not a pipeline change — it would live as its own script or
`cli_fiftyone.py`, reading already-produced artifacts, the same "post-hoc tool over existing
outputs" shape as `scripts/plot_run.py` and `scripts/validate_trj.py`.

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

### Deployment: Docker + CUDA — a concrete multi-stage shape

2026 best practice for a PyTorch+CUDA image is a **multi-stage build**: a `-devel` base
(compilers/headers) for the build stage, a matching `-runtime` base (no build tools, smaller)
for the final stage, copying over only the built virtualenv/artifacts — cited reductions of
~60% image size from this pattern alone. Concretely, mapping onto this repo:

- Build stage: `pytorch/pytorch:<version>-cuda<N>-cudnn<N>-devel`, `uv sync` with
  `[tool.uv.sources]` pointed at the CUDA torch/torchvision index (the same swap
  `docs/IMPLEMENTATION_PLAN.md` Group A1 already identifies as needed once a GPU exists — this
  is that same decision point, reused for deployment rather than dev).
- Runtime stage: the matching `-runtime` base, `COPY --from=build` the built venv, no compiler
  toolchain shipped.
- Not designed here: which CUDA/cuDNN version to pin (depends on the actual deployment GPU,
  unknown). `boxmot`/`ultralytics`'s AGPL-3.0 status is resolved, not open — TraTrac is GPL-3.0
  and GPLv3 §13 explicitly permits the combination (see `CLAUDE.md` Dependency Notes); a
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
