# 20 — Video export: overlay video + composite exporter

## What this adds

A third trajectory output alongside the SSAM `.trj`: an **overlay video** —
each processed frame with its vehicles drawn on top (front/rear bumpers,
orientation line, `vN speed` label, per-track trails). And a **composite
exporter** so a single run can emit the `.trj` and the video at once.

This is a debug/visualization output, not a new analytics format. It is neither
the SSAM export (A) nor the extended internal export (B) of the dual-export
architecture (see `01_architecture_principles.md`); it is a rendering of what the
pipeline saw. New analytics data still goes into (B), never here.

## Why the export port had to widen

The pipeline's last step is the `TrajectoryExporter` port. Its method was
`emit_frame(timestamp_seconds, states)` — **no pixels**. A video writer
fundamentally needs the frame image. Rather than introduce a second, parallel
output port, the port was widened to:

```python
def emit_frame(self, timestamp_seconds, states, frame: Frame) -> None
```

so every output is one uniform port and they compose behind a single
`CompositeTrajectoryExporter`. Trade-off accepted: `SsamTrjExporter` now receives
a `frame` it ignores (a mild ISP wart) in exchange for one port and a clean
Composite. The alternative — a separate `FrameSink` port driven by the pipeline —
was rejected because it would make "video + trj" a pipeline-level concern instead
of a single composable adapter.

The frame handed in is the **raw** frame the pipeline processed (MVP1.9 stabilizes
*coordinates*, not pixels — see `05_75_mvp1_9.md`). When `ego_motion.enabled` is on
the overlay maps the stabilized trajectory coordinates back onto that raw frame via
the inverse of the ego-motion transform, so the overlay shows the full, uncropped
footage with correctly-aligned trajectories.

## Coordinates: no y-flip, map back to the raw frame

`VehicleState` positions are world units of the stabilized (global) frame — a
uniform `scale` (metres-per-pixel) multiple of pixels; there is no homography yet
(MVP1.x). To draw, the overlay divides by `scale` and uses the **raw y** (the SSAM
y-flip `image_height - y` is export-specific and deliberately not applied), then
maps the point back onto the raw frame via `transform_source().inverse()` — an
injected seam supplying the current frame's ego-motion transform (identity when
stabilization is off, preserving exact today's behaviour). Trails are stored in
stabilized coordinates and mapped through the *current* inverse each frame, showing
the world path from the current camera pose. The retired
`scripts/render_trajectories.py` drew the same overlay post-hoc from a `.trj` over
the raw video and misaligned under stabilization; this adapter supersedes it.
`scripts/render_violations.py` is the only surviving overlay script, marking
validator violations on top of this adapter's output video.

## Components

- `infrastructure/export/overlay_video.py` — `OverlayVideoExporter`
  (`TrajectoryExporter`). Owns per-track trail accumulation (`trail_length` 0 =
  whole path, N = rolling window of N frames); only currently-visible tracks are
  drawn so dead tracks stop ghosting. cv2 lives behind injected seams
  (`open_writer`, `draw`) plus the `transform_source` seam, so the orchestration
  (frame copy, trails, coordinate mapping, lifecycle) is unit-testable without cv2
  or a codec. The frame is copied before drawing so a composite peer reading the
  same `Frame` is never disturbed.
- `infrastructure/export/composite.py` — `CompositeTrajectoryExporter`, a GoF
  Composite over the port. All-or-nothing on enter (rolls back already-entered
  children on failure), best-effort on exit (every child closes; first error
  propagates).

## Composition with decimation

`--timestep-precision` decimates the `.trj` via `DecimatingTrajectoryExporter`.
The CLI wraps **only the SSAM leg** in the decimator, then composes
`[decimated_ssam, overlay_video]`. Result: the `.trj` carries coarse TIMESTEPs
while the overlay video keeps every processed frame. The `TimedExporter` (when
`--timing-csv` is on) wraps the composite, so the EXPORT step records once per
frame covering both writes (see `15_step_timing.md`).

## CLI / config

Per the zero-defaults rule (`19_config_file.md`), `export.video_out` is a
required **toggleable** key — `""` disables it, exactly like `run.timing_csv`.
`export.video_trail` is required (and range-checked, `>= 0`) **only when**
`video_out` is set, mirroring how the ORB parameters are required only when
`ego_motion.enabled` is true.

| Config key | Flag | Meaning |
| --- | --- | --- |
| `export.video_out` | `--video-out` | `.mp4` overlay path; `""` / omitted-flag = off |
| `export.video_trail` | `--video-trail` | trail length in frames; `0` = whole path |

`video_out` must differ from `export.out` and `run.timing_csv`; an existing
output is guarded by `--force` like the other outputs.
