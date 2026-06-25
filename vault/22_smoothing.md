# 22 — Trajectory smoothing: constant-acceleration Kalman + RTS (two-pass)

## What this adds

A **de-jittering** stage for vehicle trajectories — and, since the export inversion, the
**only** path that produces an SSAM `.trj`. Detector center jitter makes the raw
bbox-centroid path noisy; finite-differencing that raw centroid for velocity/acceleration
is a high-pass filter, so a sub-pixel wobble explodes into acceleration/jerk
(`research_high_precision_tracking.md` §1, Punzo 2011; the `accel-noise-root-cause` memory).
The fix the literature converges on (§2, highD): smooth **position** with a
constant-acceleration motion model and read velocity/acceleration out of the **filter
state**. This pass owns the kinematics (heading/speed/accel) that the perception run no
longer computes.

## Two-pass design (streaming forward, offline backward)

The optimal smoother is **RTS** (Rauch-Tung-Striebel: Kalman forward pass + backward
pass) — zero-phase, no lag — but it needs the whole track, so it can't run in the
streaming loop. To keep the pipeline streaming (and the real-time door open), smoothing is
split across a file boundary:

```
pass 1 (the perception run): detect → track → record (raw tracked observations) = export.out
pass 2 (offline):            record → forward KF + RTS per track → smoothed .trj
```

- **Pass 1** is the `tratrac` run. Its **only** output is the **track record** (`export.out`)
  — the canonical dual-export **"B"** format (`01_architecture_principles.md`), now the
  pipeline's primary product, not an opt-in sidecar. It is an **Apache Parquet** file: the
  **raw measurements** (centroid + bbox + class per track per frame) as columns, with the run
  metadata (`fps,width,height,total_frames,meters_per_pixel`) in the Parquet **schema metadata**
  so the record is self-contained. The pipeline records to a `TrackSink` (`ParquetTrackSink`) it
  owns directly. (Parquet is the MVP7 storage choice, pulled forward for the canonical record.)
- **Pass 2** is `tratrac-smooth RECORD.csv --out final.trj [--pos-noise PX] [--jerk Q]
  [--timestep-precision S]`: group by track → forward+RTS smooth → reconstruct `VehicleState`
  (kinematics via `build_state`) → write via `SsamTrjExporter` (wrapped in
  `DecimatingTrajectoryExporter` when `--timestep-precision` thins the TIMESTEPs). It produces
  only the smoothed `.trj`; to visualize it, render with `tratrac-render` (vault/20).

**Why raw measurements, not filter state:** pass 2 re-runs the forward pass (cheap) so the
sidecar stays small and inspectable, and the smoother can be **re-tuned offline with no
re-detection** — rerun `tratrac-smooth` with different `--jerk`/`--pos-noise` to sweep.
Keeping the record raw (not filtered) is what makes this re-tuning possible: smoothing
always starts from the measurements, never from already-smoothed kinematics.

## The core (`application/kalman.py`)

Hand-rolled numpy (no `filterpy` — only a transitive boxmot pin, untyped). Per-axis
constant-acceleration state `[position, velocity, acceleration]`, white-noise-**jerk**
process model, **variable `dt`** (survives `input.process_fps` decimation). x and y are
independent (a CA model has no cross-axis coupling).

- `smooth_track(xs, ys, timestamps, *, pos_noise, jerk)` — forward Kalman + RTS over a whole
  track; zero-phase. This is what `tratrac-smooth` uses.
- `KinematicKalmanFilter` — stateful forward-only filter. Currently unused by the (offline)
  smoother; kept as the primitive for a future streaming/RT `.trj` path.

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
- `domain/ports.py` — `TrackSink` (the pipeline's primary output port);
  `infrastructure/tracks/parquet.py` — `ParquetTrackSink` + `read_tracks` (Parquet, `pyarrow`).
  The pipeline records to the sink directly (it owns its lifecycle); there is no
  `RecordingTracker` decorator anymore.
- `cli_smooth.py` — `tratrac-smooth`; `cli.py`/`config.py` — `export.out` is the record.
