# Persisting the Per-Frame Transform (`TransformSink`)

> Split out of `src/tratrac/infrastructure/video/EGO_MOTION.md` (MVP1.9) — read that doc
> first for the ego-motion estimator this sidecar records the output of.

When stabilization is on, the `.trj` carries positions in the **global** frame, not
raw pixels. The overlay video maps them back onto the raw frame live (via the
inverse of `current_transform`), but any *offline* consumer — notably the post-hoc
`tratrac-render`, which draws the `.trj`'s trajectories (and optional `validate_trj.py`
violations) onto the video — has no access to that transform. The per-frame transforms
exist only in memory inside `OrbEgoMotionEstimator` during a run.

The fix persists them as a sidecar so the same global→raw inverse can be applied
afterwards. There is **one consistent global space** (the keyframe chain is
continuous — see `EGO_MOTION.md`), so the whole clip's coordinate↔pixel map is a *table*: one row
per frame of the 6 similarity coefficients. It is **not** a single static transform
— the camera pose changes every frame.

Captured exactly like step timing (`src/tratrac/infrastructure/timing/STEP_TIMING.md`): an output port plus an
opt-in **decorator around the port**, leaving the pipeline untouched.

- `domain/stabilization.py` — `FrameTransform(frame_index, transform)`, the pure
  per-frame record (sibling of `StepTiming`). `frame_index` is the absolute
  `Frame.index` — it matches `round(timestamp_s * fps)`, the key `tratrac-render`
  derives for each trajectory/violation row, so the two align on the same integer.
- `domain/ports.py` — `TransformSink`: `record(FrameTransform)`. Streaming, no Null
  sink (off = decorator not inserted, zero cost), exactly like `TimingSink`.
- `infrastructure/transform/recording.py` — `RecordingEgoMotionEstimator`, the
  `EgoMotionEstimator` analog of the `Timed*` decorators: forwards `estimate`,
  records `(frame.index, result)`, returns it. The CLI keeps the *concrete*
  estimator for the overlay's `transform_source` and the `DetectionObserver`, and
  hands the decorator to the pipeline.
- `infrastructure/transform/csv.py` — `CsvTransformSink`, header
  `frame,a,b,c,d,tx,ty`, one row written immediately per frame (no buffering — each
  record is self-contained, unlike the wide-row timing CSV).
- `cli_render.py` (`tratrac-render`) — `--transforms` loads the table (`read_transforms`)
  and maps each trajectory and violation position through `inverse(transform[frame])`
  before drawing, using the domain `Transform2D.inverse().apply()` directly (it is a
  package CLI, so it reuses the real geometry rather than re-deriving it).

**Config (zero-defaults).** `export.transform_csv` is a required toggleable key
(`""` = off), like `run.timing_csv`. Because it only makes
sense alongside stabilization, `RunConfig.resolve` **fails** if it is set while
`ego_motion.enabled` is false (with stabilization off every transform is the
identity — there is nothing to record). The validator is deliberately *not* a
consumer: it checks kinematics in the global frame, where the drone's motion has
already been removed; mapping back to raw pixels would re-introduce it.
