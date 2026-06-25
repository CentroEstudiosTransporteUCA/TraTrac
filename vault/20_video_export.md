# 20 — Overlay video: post-hoc trajectory rendering (`tratrac-render`)

## What this adds

An **overlay video** — each frame of a clip with its vehicles drawn on top
(front/rear bumpers, orientation line, `vN speed` label, per-track trails). It is
produced **after** a run by the standalone `tratrac-render` tool, which reads the
run's SSAM `.trj` (plus, for an ego-motion run, the transform CSV) and draws over
the source clip.

This is a debug/visualization output, not a new analytics format. It is neither
the SSAM export (A) nor the extended internal export (B) of the dual-export
architecture (see `01_architecture_principles.md`); it is a rendering of the
trajectories. New analytics data still goes into (B), never here.

## Why it is post-hoc (it used to run inside the pipeline)

Originally the overlay ran **inside the per-frame loop**: an `OverlayVideoExporter`
was composed onto the `.trj` exporter, so each run copied the frame, drew on it, and
encoded a video frame on every iteration. That made every run pay a large per-frame
cost for output that is **fully derivable from the `.trj` afterward**. So it was
pulled out:

- The run now only detects → tracks → exports the `.trj` (+ optional sidecars). No
  per-frame drawing or encoding.
- `tratrac-render` does the drawing as a separate step, reusing the *same*
  `OverlayVideoExporter` drawing engine. Only the **driver** changed: the vehicle
  states come from reading a `.trj`, not from live tracking.
- `tratrac-smooth` likewise dropped its in-process `--video-out`; to visualize a
  smoothed run, render its smoothed `.trj` with `tratrac-render` like any other.

This makes rendering a normal downstream step alongside `validate_trj` / `plot_run`,
not a tax on every run.

## The viz-only `.trj` reader

`tratrac-render` needs vehicle states to draw; it gets them from `read_trj`
(in `infrastructure/export/ssam_trj.py`, next to the writer so it shares the struct
defs). It inverts `_write_vehicle_record` exactly: multiply the grid coordinates by
the DIMENSIONS `scale`, undo the SSAM Y-flip, then reconstruct `centroid =
midpoint(front, rear)`, `heading = unit(front − rear)`, and the stored
dimensions/speed. float32-exact for the pixel-rounded drawing.

This is the **one** place the `.trj` is read back, and it is **viz-only**: it does
not reopen the load-bearing invariant that the SSAM `.trj` is export-only, never
re-ingested into the processing/analytics path (see `01` and `22`). Smoothing and
analytics consume the **raw track record** (`export.out`), not the lossy `.trj`.
Reading the `.trj` to *draw* it is terminal visualization, exactly like
`validate_trj.py` / `plot_run.py` parsing it.

## The export port is frameless; the renderer is standalone

The `TrajectoryExporter` port is `emit_frame(timestamp_seconds, states)` — a pure
data port, no pixels. It was briefly widened to carry a `Frame` so the overlay could
compose into the pipeline; with rendering now post-hoc, that parameter was removed
and `OverlayVideoExporter` is **no longer a `TrajectoryExporter`**. It is a standalone
renderer whose own `emit_frame(timestamp, states, frame)` takes the pixels to draw
on, driven directly by `tratrac-render`. So the data exporters (`SsamTrjExporter`,
`DecimatingTrajectoryExporter`, `TimedExporter`) stay frameless, and the one class
that needs pixels carries them explicitly.

## Coordinates: no y-flip, map back to the raw frame

`VehicleState` positions are world units of the stabilized (global) frame — a
uniform `scale` (metres-per-pixel) multiple of pixels; there is no homography yet
(MVP1.x). To draw, the overlay divides by `scale` and uses the **raw y** (the SSAM
y-flip `image_height − y` is export-specific and deliberately not applied), then maps
the point back onto the raw frame via `transform_source().inverse()`. For
`tratrac-render` that per-frame transform comes from the run's `--transforms` CSV
(identity when omitted / stabilization was off). Trails are stored in stabilized
coordinates and mapped through the *current* inverse each frame, showing the world
path from the current camera pose. `scripts/render_violations.py` is the sibling
overlay tool, marking validator violations on top of this (or any) video.

## Components

- `infrastructure/export/overlay_video.py` — `OverlayVideoExporter`
  (`TrajectoryExporter`). Owns per-track trail accumulation (`trail_length` 0 =
  whole path, N = rolling window of N frames); only currently-visible tracks are
  drawn so dead tracks stop ghosting. cv2 lives behind injected seams
  (`open_writer`, `draw`) plus the `transform_source` seam, so the orchestration
  (frame copy, trails, coordinate mapping, lifecycle) is unit-testable without cv2
  or a codec. Unchanged by the move — only its caller changed.
- `cli_render.py` — `tratrac-render`. Reads the `.trj` (`read_trj`) and the optional
  transforms CSV, buckets states onto absolute video frames by
  `round(timestamp * fps)` (fps from the **clip** — the `.trj` carries time but not
  fps — the same alignment `render_violations.py` uses), opens the clip, and drives a
  single `OverlayVideoExporter`.

(The old in-pipeline `CompositeTrajectoryExporter` was removed: with rendering gone
from both `process` and `smooth`, nothing composes exporters anymore.)

## Interaction with decimation

`tratrac-render` draws states where the `.trj` has them. If the run used
`export.timestep_precision` (or `input.process_fps`), the `.trj` only carries states
on the emitted timesteps, so the overlay shows bumpers/trails only on those frames
and bare frames in between. For a smooth, full-cadence overlay, render from a `.trj`
produced with `timestep_precision = 0`.

## CLI

| Tool | Invocation |
| --- | --- |
| `tratrac-render` | `tratrac-render VIDEO --trj RUN.trj --out OVERLAY.mp4 [--transforms TCSV] [--trail N] [--force]` |

`--out` must not pre-exist without `--force`. For an ego-motion run pass
`--transforms` (the run's `export.transform_csv`) so the global-frame trajectories
map back onto the raw video; omit it for a non-stabilized run.
