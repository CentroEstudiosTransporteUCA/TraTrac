# The Unified Transform File (`TransformSink`, `TransformTable`)

> Split out of `infrastructure/video/ego_motion_orb.py`'s module docstring (MVP1.9) — read that doc
> first for the ego-motion estimator this file records the output of.

When stabilization is on, the `.trj` carries positions in the **global** frame, not
raw pixels. The overlay video maps them back onto the raw frame live (via the
inverse of `current_transform`), but any *offline* consumer — notably the post-hoc
`tratrac-render`, which draws the `.trj`'s trajectories (and optional `validate_trj.py`
violations) onto the video — has no access to that transform. The per-frame transforms
exist only in memory inside `OrbEgoMotionEstimator` during a run.

The fix persists them as a file so the same global→raw inverse can be applied
afterwards. There is **one consistent global space** (the keyframe chain is
continuous — see `infrastructure/video/ego_motion_orb.py`'s module docstring), so the whole clip's coordinate↔pixel map is a
*table*: one row per frame of the 6 similarity coefficients. It is **not** a
single static transform — the camera pose changes every frame. GSD scale and a
world-projection homography are rows in this **same** file too — see "One file,
one row model" below.

Ego-motion rows are captured exactly like step timing
(`src/tratrac/infrastructure/timing/STEP_TIMING.md`): an output port plus an
opt-in **decorator around the port**, leaving the pipeline untouched.

- `domain/stabilization.py` — `FrameTransform(frame_index, transform)`, the pure
  per-frame record (sibling of `StepTiming`). `frame_index` is the absolute
  `Frame.index` — it matches `round(timestamp_s * fps)`, the key `tratrac-render`
  derives for each trajectory/violation row, so the two align on the same integer.
- `domain/ports.py` — `TransformSink`: `record(FrameTransform)`. Streaming, no Null
  sink (off = decorator not inserted, zero cost), exactly like `TimingSink`.
- `infrastructure/transform/recording.py` — `RecordingEgoMotionEstimator`, the
  `EgoMotionEstimator` analog of the `Timed*` decorators: forwards `estimate`,
  records `(frame.index, result)`, returns it. Used only by
  `tratrac-preprocess estimate` (`cli_preprocess.py`) — the tool that actually
  estimates ego-motion; `tratrac` itself never constructs an
  `OrbEgoMotionEstimator`, so it has nothing to record.
- `infrastructure/transform/sink.py` — `CoordinateTransformSink`: `record`
  (the `TransformSink` port) writes one ego-motion row per frame; `record_row`
  is the lower-level escape hatch `tratrac-preprocess estimate` also uses to
  interleave the per-frame scale rows in the same pass, and `project` uses to
  append homography rows afterwards. Each row is written immediately (no
  buffering — self-contained JSON Lines, unlike the wide-row timing CSV),
  staged to a `.partial` file and atomically published (`Path.replace`) only on
  a clean exit — a reader of the canonical path never observes a half-written
  table.
- `cli_render.py` (`tratrac-render`) — `--transforms` loads the ego-motion rows
  (`read_ego_motion`, returning a `TransformTable`) and maps each trajectory and
  violation position through the inverse of that frame's `Transform2D` before
  drawing, using the domain `Transform2D.inverse().apply()` directly (it is a
  package CLI, so it reuses the real geometry rather than re-deriving it).

## One file, one row model

Ego-motion, GSD scale, and world-projection homography all implement one
shared domain concept: `CoordinateTransform`/`InvertibleCoordinateTransform`
(`domain/ports.py`: `apply(point, frame_index)` / `reverse(point, frame_index)`).
Unlike an earlier revision (three separate sidecar *files* sharing one record
*schema*), they are now three **kinds of row in the same file** — see
`infrastructure/transform/records.py`'s module docstring for the row shape
(`frame_index` the primary key, `zone` always a concrete polygon, `function`
one of four `<Kind>Function`/`<Kind>Codec` pairs: `Identity`, `Scale`,
`Similarity`, `Homography`) and `application/coordinate_transforms.py`'s
`TransformTable` for the one class that reads any of them back (group by
frame, filter by zone, delegate to the row's function — no per-kind branching
anywhere in the reader). A zone's `reference_frame` pose lookup
(`application/exclusion.py`'s `_pose_for`/`to_global_polygons`) is just a
`TransformTable` built from the ego-motion rows; an anchor's pose is simply one
row of that same table, not a separate record (there is no anchor manifest —
see `src/tratrac/application/EXCLUSION_ZONES.md`).

Unifying the *representation* did not change *when* each kind gets built:
ego-motion and scale are both resolved during `tratrac-preprocess estimate`'s
live pass (ego-motion because masking needs a live frame — see `infrastructure/video/ego_motion_orb.py`'s module docstring;
scale because that's simply where the tool's `[calibration]`-equivalent CLI
flags live now), and world-projection stays a post-hoc second step
(`tratrac-preprocess project`, fitting from an operator-authored
`calibration.json` against the anchors `estimate` exported).

**Where it's built vs. consumed.** `tratrac-preprocess estimate --out PATH` (a
plain, required CLI option — that tool isn't config-driven) always writes the
ego-motion (if `--background-zones` given) and scale rows; `tratrac-preprocess
project --transforms PATH --calibration ...` fits and replaces the homography
rows in the same file (idempotent — re-running it replaces the projection
stage rather than appending duplicates). `tratrac`'s `input.transforms_in` is
a required config key naming this file: it reads only the ego-motion rows
(`read_ego_motion`) via `PrecomputedEgoMotionEstimator` — there is no
`[ego_motion]` section or `enabled` toggle for it to gate on; an ego-motion-free
file (a static camera) just means every query falls through to the identity.
`tratrac-postprocess --transforms PATH` reads the ego-motion rows (for zone/
correspondence pose resolution) *and* the scale-or-homography rows (for
projecting/scaling the record); `tratrac-render --transforms PATH` reads only
the ego-motion rows, to map global-frame positions back onto the raw video.
