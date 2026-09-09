# `application/` Design Docs

The application layer orchestrates domain types into runnable capabilities. Each doc below
covers one capability owned by this layer:

| Doc | Covers |
| --- | --- |
| [`WORLD_PROJECTION.md`](WORLD_PROJECTION.md) | Coordinate systems background + MVP2's shipped post-hoc single-homography world projector |
| [`PROGRESS_REPORTING.md`](PROGRESS_REPORTING.md) | The `ProgressReporter` output port and `ProgressEvent` family |
| [`CONFIG_DESIGN.md`](CONFIG_DESIGN.md) | Why the run config has zero hardcoded defaults; `RunConfig.resolve`; `--check` |
| [`EXCLUSION_ZONES.md`](EXCLUSION_ZONES.md) | Post-hoc, track-aware "do-not-analyze" polygons |
| [`SMOOTHING.md`](SMOOTHING.md) | Constant-acceleration Kalman/RTS trajectory de-jittering — the only `.trj` path |
| [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) | External literature review backing the smoothing + stabilization design |

See [`../../../docs/README.md`](../../../docs/README.md) for the full repo documentation map.
