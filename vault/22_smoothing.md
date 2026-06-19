# 22 — Trajectory smoothing: constant-acceleration Kalman + RTS (two-pass)

## What this adds

A **de-jittering** stage for vehicle trajectories. Detector center jitter makes the
raw bbox-centroid path noisy, and the old `EmaOrientationEstimator` published that raw
centroid as position and **finite-differenced** it for velocity/acceleration — and
differentiation is a high-pass filter, so a sub-pixel wobble explodes into
acceleration/jerk (`research_high_precision_tracking.md` §1, Punzo 2011; the
`accel-noise-root-cause` memory). The fix the literature converges on (§2, highD): smooth
**position** with a constant-acceleration motion model and read velocity/acceleration out
of the **filter state**.

## Two-pass design (streaming forward, offline backward)

The optimal smoother is **RTS** (Rauch-Tung-Striebel: Kalman forward pass + backward
pass) — zero-phase, no lag — but it needs the whole track, so it can't run in the
streaming loop. To keep the pipeline streaming (and the real-time door open), smoothing is
split across a file boundary:

```
pass 1 (streaming, RT-capable):  detect → track → export B (raw tracked observations)
pass 2 (offline):                export B → forward KF + RTS per track → smoothed .trj
```

- **Pass 1** writes a **track-observation sidecar** (`export.tracks` / `--tracks`) — the
  first concrete instance of the dual-export **"B"** format (`01_architecture_principles.md`).
  It stores the **raw measurements** (centroid + bbox + class per track per frame) under a
  header (`fps,width,height,total_frames,meters_per_pixel`). Written by `RecordingTracker`
  teeing the tracker output to `CsvTrackSink` — the pipeline is untouched (same decorator
  idiom as the transform sidecar).
- **Pass 2** is `tratrac-smooth TRACKS.csv --out final.trj [--pos-noise PX] [--jerk Q]`:
  group by track → forward+RTS smooth → reconstruct `VehicleState` → write via the reused
  `SsamTrjExporter`.

**Why raw measurements, not filter state:** pass 2 re-runs the forward pass (cheap) so the
sidecar stays small and inspectable, and the smoother can be **re-tuned offline with no
re-detection** — rerun `tratrac-smooth` with different `--jerk`/`--pos-noise` to sweep.
Consequence: the forward pass in the pipeline (Stage 4, optional) does *not* feed the RTS —
it only serves the pipeline's own immediate `.trj` and the latent RT path.

## The core (`application/kalman.py`)

Hand-rolled numpy (no `filterpy` — only a transitive boxmot pin, untyped). Per-axis
constant-acceleration state `[position, velocity, acceleration]`, white-noise-**jerk**
process model, **variable `dt`** (survives `input.process_fps` decimation). x and y are
independent (a CA model has no cross-axis coupling).

- `smooth_track(xs, ys, timestamps, *, pos_noise, jerk)` — forward Kalman + RTS over a whole
  track; zero-phase.
- `KinematicKalmanFilter` — stateful forward-only filter for streaming (Stage 4).

Reconstruction (`application/track_smoothing.py`): smoothed pixel position → metric centroid
(×scale); heading from smoothed velocity (low-speed bbox-major-axis fallback); SSAM scalar
`acceleration` = `d|v|/dt` = `(v·a)/|v|`; dimensions from bbox major/minor (×scale).

## Tuning

- `--pos-noise` (px): measurement-noise std ≈ detector center jitter (~1–3 px).
- `--jerk`: process spectral density; **larger = more responsive, less smooth**. Lower it if
  real braking is over-smoothed; raise it if jitter survives.

The success metric is `scripts/validate_trj.py`: a smoothed `.trj` should show far fewer
physically-impossible-jerk violations than the EMA `.trj` (the Punzo metric, §1).

## Files
- `application/kalman.py` — CA filter + RTS core.
- `application/track_smoothing.py` — observations → smoothed `VehicleState`s.
- `domain/ports.py` — `TrackSink`; `infrastructure/tracking/recording.py` — `RecordingTracker`;
  `infrastructure/tracks/csv.py` — `CsvTrackSink` + `read_tracks`.
- `cli_smooth.py` — `tratrac-smooth`; `cli.py`/`config.py` — `export.tracks` wiring.
- `application/orientation_kalman.py` — inline forward `KalmanOrientationEstimator`
  (the streaming/RT path), selectable via `orientation.method = kalman` (EMA default).
  Reuses the shared `build_state` reconstruction and adds staleness eviction (the
  `EmaOrientationEstimator.forget` gap — per-track filters are dropped after a horizon).
