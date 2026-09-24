# Trajectory smoothing: constant-acceleration Kalman + RTS (two-pass)

## What this adds

A **de-jittering** stage for vehicle trajectories — and, since the export inversion, the
**only** path that produces an SSAM `.trj`. Detector center jitter makes the raw
bbox-centroid path noisy; finite-differencing that raw centroid for velocity/acceleration
is a high-pass filter, so a sub-pixel wobble explodes into acceleration/jerk
(`src/tratrac/application/RESEARCH_NOTES.md` §1, Punzo 2011; the `accel-noise-root-cause` memory).
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
  — the canonical dual-export **"B"** format (`src/tratrac/domain/ARCHITECTURE.md`), now the
  pipeline's primary product, not an opt-in sidecar. It is an **Apache Parquet** file: the
  **raw measurements** (centroid + bbox + class per track per frame) as columns, with the run's
  video metadata (`fps,width,height,total_frames`) in the Parquet **schema metadata** so the
  record is self-contained. The GSD metric scale is **not** here — it's resolved once by
  `tratrac-preprocess estimate` and lives only as `scale` rows in the shared transforms file
  (`infrastructure/transform/records.py`), fed to pass 2 via `tratrac-postprocess --transforms`
  (the same file that already carries the ego-motion rows). The pipeline records to a
  `TrackSink` (`ParquetTrackSink`) it owns directly. (Parquet is the MVP7 storage choice, pulled
  forward for the canonical record.)
- **Pass 2** is `tratrac-postprocess RECORD.parquet --transforms TRANSFORMS.jsonl
  [--out final.trj] [--smoothed-record final.parquet]
  [--exclusion-zones … ] [--pos-noise PX] [--jerk Q] [--timestep-precision S]`:
  (optionally **filter** out tracks inside exclusion zones, src/tratrac/application/EXCLUSION_ZONES.md) → group by track →
  forward+RTS smooth (**once**) → reconstruct `VehicleState` (kinematics via `build_state`) →
  write **either or both** of two independent, optional outputs from that one smoothing pass —
  see "Dual-space export" below. At least one of `--out`/`--smoothed-record` is required;
  `--transforms` is always required (it carries the mandatory scale-or-homography rows —
  a run is never un-calibrated in this sense). To visualize the `.trj`, render with
  `tratrac-render` (infrastructure/export/overlay_video.py's module docstring); to visualize directly
  over raw video frames (e.g. in FiftyOne), use `--smoothed-record` instead.

**Why raw measurements, not filter state:** pass 2 re-runs the forward pass (cheap) so the
sidecar stays small and inspectable, and the smoother can be **re-tuned offline with no
re-detection** — rerun `tratrac-postprocess` with different `--jerk`/`--pos-noise` to sweep.
Keeping the record raw (not filtered) is what makes this re-tuning possible: smoothing
always starts from the measurements, never from already-smoothed kinematics.

## Dual-space export: one smoothing pass, two optional outputs

**The problem this solves.** A `--transforms` file carrying homography rows (MVP2, fitted
ahead of time by `tratrac-preprocess project`) projects a track onto the metric world plane
*before* smoothing (deliberately — see "Why smoothing runs after projection, not before"
below), so a projected `.trj`'s positions are real-world metres from a homography the `.trj`
file itself doesn't store. A tool that wants to overlay trajectories on the *raw video*
(pixel canvas) — `tratrac-fiftyone` is the motivating case — can't use a calibrated `.trj` for
that: dividing metres by the video's pixel width/height produces boxes clustered near the
origin, disconnected from where the vehicle actually is. This was a real bug, not a
hypothetical one: confirmed visually in the FiftyOne App, every box pinned near the top-left
corner regardless of the vehicle's true position, traced to exactly this — `cli_fiftyone.py`
naively normalizing a calibrated `.trj`'s world coordinates as if they were pixels.

**The fix is not "smooth twice."** The first design considered was running the Kalman/RTS
smoother a second time directly on the raw pixel-space record, independent of the
world-space smoothing pass that feeds `.trj`. Rejected: perspective distortion makes
pixel-space jitter and world-space jitter statistically different, so a smoother configured
(implicitly, through its physically-meaningful noise parameters) for one space risks damping
*real* motion when run on data in the other. Smoothing must run in whichever space is
physically justified — world space, when there's a homography involved — exactly once.

**The actual fix: smooth once, export twice.** `--out` (`.trj`) and `--smoothed-record`
(Parquet, `infrastructure/tracks/smoothed_parquet.py`) are both optional, independent outputs
built from the **same** `states_by_frame` the one smoothing pass produces (`_smooth_recording`
in `cli_postprocess.py`, unchanged) — `--out` is not required if `--smoothed-record` is given,
and vice versa. `--smoothed-record` is always raw image-space pixels: every run projects every
observation through the transforms file's `TransformTable` before smoothing (see
`cli_postprocess.py`'s module docstring — there is no separate "unprojected" path to fall back
to, not even for a plain GSD scale), so recovering pixels is always the same operation:
`invert_state_to_image` (`application/track_smoothing.py`) inverts the *same* projector
`_project_to_world` returns for the forward pass — not a fresh fit, the literal object — via
`InvertibleCoordinateTransform.reverse(point, frame_index)` (`domain/ports.py`). It inverts
**four points** independently (front/rear bumpers → centroid, heading, length; left/right side
points → width) rather than transforming `centroid`+`heading`+`dimensions` directly — the same
reason the SSAM `.trj` format itself stores front/rear bumper points instead of
centroid+heading+length: a point transforms correctly under an arbitrary coordinate change (a
homography, here), a direction+magnitude pair does not. This is a genuine geometric inverse of
the one smoothing pass that already ran, not a second filter. For a plain GSD scale this is
mathematically just a division, computed the general way instead of as a special case.
- Velocity/acceleration are **not** carried into `--smoothed-record` — there's no general,
  honest way to convert a world-space smoothed velocity into an image-space one without
  differentiating the (possibly per-frame) inverse homography, and nothing consumes it yet.
  `SmoothedObservation` only stores what a homography inverse actually recovers cleanly:
  position, heading, dimensions.

**The 0-origin shift is part of what gets inverted, not just the homography.**
`_project_to_world` also translates every projected point to a non-negative origin
(`_normalize_world_recording`) — a real detail the naive "just call `projector.reverse(point, frame_index)`"
version of this design missed at first. `TranslationTransform` (`application/coordinate_transforms.py`,
a small standalone `CoordinateTransform` for a constant pixel/metre shift) composes with the
already-built `TransformTable` via `compose_invertible` (`shift = TranslationTransform(shift_x,
shift_y)`; `shifted_projector = compose_invertible(projection, shift)`), so `_project_to_world`
returns the **shifted** projector — the one whose `.reverse()` undoes the translation *and*
the homography together, not just the homography.

**Scope: an invertible `TransformTable` only** — a single fitted homography (`zone = whole
canvas`) or a per-anchor materialization (one homography row per frame, still one row per
frame index) are both invertible, since exactly one row matches any given `(point, frame_index)`
query. A per-plane homography table (`--plane-zones`, MVP3, fitted by `tratrac-preprocess
project`) selects its homography by classifying the *input image* point's position — exactly
what's unknown when starting from a world point — so `TransformTable.is_invertible` is `False`
whenever more than one distinct `zone` is in play for a frame. Combining `--smoothed-record`
with a non-invertible (multi-zone) projection is rejected upfront with a clear error
(`postprocess` in `cli_postprocess.py`, see the `is_invertible` check near the top of the
command) rather than silently guessing or producing a misleading partial result. A
future version could carry each observation's plane id forward through smoothing so the reverse
pass knows which homography to invert — not built yet; no real footage has needed multi-plane
`--smoothed-record` output so far.

### Why smoothing runs after projection, not before

`_project_to_world` is called before `_smooth_recording` in `postprocess` — deliberately, and
this order is **not** changed by the dual-space export design above (both outputs still come
from that one, post-projection smoothing pass). The constant-acceleration motion model is more
physically meaningful applied in real-world metres than in perspective-distorted pixels: a
homography's local scale/shear varies across the frame (more so off-axis; TraTrac's nadir-only
target footage keeps this modest but doesn't eliminate it), so "constant pixel acceleration"
doesn't correspond to constant real acceleration in general, while `pos_noise`/`jerk` — tuned
against real physical units — stay meaningful only if the filter actually runs in those units.
`_project_to_world` converts `pos_noise`/`jerk` from pixels to world units via the homography's
local scale (`local_scale_at`) specifically so this ordering's physical meaning is preserved
through the noise parameters too, not just the positions.

## The core (`application/kalman.py`)

Hand-rolled numpy (no `filterpy` — only a transitive boxmot pin, untyped). Per-axis
constant-acceleration state `[position, velocity, acceleration]`, white-noise-**jerk**
process model, **variable `dt`** (survives `input.process_fps` decimation). x and y are
independent (a CA model has no cross-axis coupling).

- `smooth_track(xs, ys, timestamps, *, pos_noise, jerk)` — forward Kalman + RTS over a whole
  track; zero-phase. This is what `tratrac-postprocess` uses.
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

### Frontier upgrade: learned/adaptive noise covariance (KalmanNet family)

`--pos-noise`/`--jerk` above are **fixed, hand-tuned hyperparameters** — the correct foundation
(constant-acceleration Kalman + RTS is confirmed against the literature, see
`RESEARCH_NOTES.md`), but not the current research frontier. 2025 work has moved toward
**hybrid classical+learned filters**: keep the Kalman structure (interpretable, physically
grounded — worth preserving, not replacing wholesale) but learn the Kalman gain from data
instead of computing it analytically from hand-tuned noise parameters. **This is now a design
spike, not just a flagged frontier** — see `application/KALMANNET_DESIGN.md` (Group E,
GitHub Issues) for the architecture mechanics, a reconsidered recommendation
(plain **unsupervised** KalmanNet trained on TraTrac's own recorded track data, not
MAML-KalmanNet — TraTrac's blocker turned out to be data *diversity*, not the presence of
labels), a proposed integration surface, and what's still genuinely open. Still not a drop-in
swap — unlike the detector pivot in `docs/TECH_STACK.md`, there's no off-the-shelf checkpoint
to fine-tune here, and no code has been written yet.

## Files
- `application/kalman.py` — CA filter + RTS core.
- `application/track_smoothing.py` — observations → smoothed `VehicleState`s (`build_state`,
  `smooth_to_states`); `invert_state_to_image` — the dual-space export inverse (see above).
- `application/coordinate_transforms.py` — `TransformTable`, `TranslationTransform`,
  `compose`/`compose_invertible`; `TransformTable.reverse()` composed with
  `TranslationTransform` via `compose_invertible` backs the dual-space export.
- `domain/ports.py` — `TrackSink` (the pipeline's primary output port); `CoordinateTransform` +
  `InvertibleCoordinateTransform`; `infrastructure/tracks/parquet.py` — `ParquetTrackSink` +
  `read_tracks` (Parquet, `pyarrow`). The pipeline records to the sink directly (it owns its
  lifecycle); there is no `RecordingTracker` decorator anymore.
- `infrastructure/tracks/smoothed_parquet.py` — `SmoothedTrackParquetSink` + `read_smoothed_tracks`,
  the `--smoothed-record` output (Parquet, image-space).
- `cli_postprocess.py` — `tratrac-postprocess` (filter + smooth); `--out`/`--smoothed-record`
  both optional, at least one required; `cli.py`/`config.py` — `export.out` is the record.
