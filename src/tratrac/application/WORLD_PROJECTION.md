# World Projection — MVP 2 (Coordinate Systems + Approach A)

**World projection** is implemented as a **post-hoc homography**, fitted by `tratrac-preprocess
project` — see "Implementation (Approach A)" below. **Multi-anchor projection** (GitHub Issues
Group C3) groups correspondences by `reference_frame` and fits one homography per anchor,
materializing a row (with the nearest anchor's matrix baked in) for every frame — one
homography *kind*, not a dedicated class; a single-anchor (or static) calibration is simply the
one-row-value case, not a separate code path. **Multi-homography plane projection** (MVP3, Group
C5) instead classifies each correspondence against elevation-plane zones (`--plane-zones`,
`application/ROAD_GRAPH.md`) and materializes one homography row per plane per frame, each
carrying that plane's own polygon as its `zone` — spatial selection, orthogonal to the
per-anchor path's temporal (`frame_index`) selection; the two are not composed (a calibration
spanning both multiple anchors and multiple planes pools an anchor's correspondences per plane
regardless of anchor, a known, documented limitation). **Automatic-calibration correspondence
proposal** (Group C4) is scoped to proposal-only per its resolved repo-boundary question — see
"Automatic calibration from road geometry" below and `src/tratrac/application/AUTO_CALIBRATION.md`.
The SuperPoint + LightGlue stabilization upgrade (MVP1.9's ORB still does ego-motion; Group B3,
GitHub Issues) is separate, tracked there. With projection, multi-anchor, and multi-plane all in
place, SSAM positions can be metric world coordinates for wide-swept, many-anchor,
grade-separated scenes, not just bounded flat ones.

---

## Goal

Generate:

- physically meaningful SSAM trajectories
- world-space coordinates
- stabilized motion

---

## Coordinate Systems Background

### MVP1

Produces:

- syntactically valid SSAM `.trj`

BUT:

- coordinates are not yet physically meaningful

This MVP validates:

- pipeline correctness
- serialization
- tracking
- exporter architecture

### MVP2+

All SSAM exports use:

#### world-space metric coordinates

This is mandatory because SSAM assumes:

- metric geometry
- real distances
- real speeds
- real accelerations

NOT:

- image pixels

#### Why Image-Space Coordinates Break SSAM

Passing image-space coordinates into SSAM causes:

| Problem | Consequence |
| --- | --- |
| Pixel distance ≠ meters | Invalid TTC calculations |
| Perspective distortion | Invalid vehicle sizes |
| Camera motion | Fake accelerations |
| Multi-level roads overlap visually | False conflicts |
| Spatial scale varies | Invalid analytics |

Meaning:

- SSAM may still parse the file
- but the analytics become scientifically invalid

### Multi-Homography Geometry (Group C5, MVP3)

> See "Multi-homography plane projection" further below for the per-plane `homography` row
> design. The rationale below (why single-homography breaks for grade separation, why not full
> 3D) is unchanged by it.

#### Why

Single homography assumes:

```text
all roads exist on the same plane
```

This breaks for:

- bridges
- ramps
- stacked highways
- overpasses

Multi-homography enables:

- multi-level road support
- physically correct trajectories

#### Why NOT Full 3D Reconstruction

Full 3D:

- expensive
- operationally difficult
- unnecessary for road-relative analytics

Roads are:

- piecewise planar

not arbitrary 3D scenes.

#### Cost

**Pros:** correct geometry for bridges/overpasses.
**Cons:** requires calibration + road-plane annotations.

---

## New Technologies

| Component | Technology |
| --- | --- |
| Stabilization | SuperPoint + LightGlue |
| Geometry | OpenCV |
| Projection | Homography |

> **Ego-motion compensation already exists as of MVP1.9** (`src/tratrac/infrastructure/video/EGO_MOTION.md`): a
> keyframe-anchored ORB + RANSAC similarity adapter behind the `EgoMotionEstimator`
> port, applied to *detection coordinates* (not pixels) before tracking. So MVP2's
> stabilization line is an *upgrade* (ORB → SuperPoint + LightGlue, see
> Group B3, GitHub Issues) gated on measurement, **not** a from-scratch addition.
> MVP2's genuinely new capability is **world projection** (the homography below);
> ego-motion compensation is inherited.

---

## Pipeline

```text
Video
    ↓
Stabilization
    ↓
YOLO-OBB (MVP1.5)
    ↓
BoT-SORT
    ↓
World Projection
    ↓
SSAM .trj Export
```

---

## Quality Improvement

### Added

- Metric coordinates
- Real speeds
- Reduced jitter
- Stable motion estimates

---

## Why This Matters

This is the FIRST MVP where:

- **SSAM analytics become scientifically meaningful**

---

## Scale Semantics

SSAM's file format already provides the bridge from abstract grid units
to physical units via the `DIMENSIONS.Scale` field. MVP2 is the first
MVP where it is filled in with a calibrated value instead of `1.0`.

The relationship the format defines (see `src/tratrac/infrastructure/export/SSAM_FORMAT.md`):

```text
real_measure = abstract_measure × Scale
```

Where:

- `abstract_measure` is the value written in each VEHICLE record's X / Y
  (and Z in v3.0). Think "calibrated pixels of the stabilised image".
- `Scale` is the `DIMENSIONS.Scale` Float — the metres-per-grid-unit
  multiplier.
- `real_measure` is what any SSAM reader recovers by multiplying.

So an SSAM file can keep coordinates in an abstract grid while still
expressing physically valid distances — the reader does the conversion.

### Important: what is and is not multiplied by Scale

| SSAM field | Storage |
| --- | --- |
| Front X / Y, Rear X / Y, Front Z / Rear Z (v3.0) | **Scaled** — abstract grid units; reader multiplies by `Scale` |
| Length, Width, Speed, Acceleration | **Un-scaled** — stored directly in the physical units declared by `DIMENSIONS.Units` (metres, m/s, m/s²) |

The exporter (`SsamTrjExporter`) does **not** multiply Length / Width /
Speed / Acceleration by Scale on write — those values must already be in
metres / m/s / m/s² before the exporter sees them. That conversion is
the application layer's job.

### Implementation contract for MVP2

- A `meters_per_pixel` calibration value must be available at pipeline
  construction (CLI flag, per-scene config, or derived from the
  stabilisation homography).
- `VehicleState.dimensions`, `velocity`, and `acceleration` are populated
  in **real units** (metres, m/s, m/s²) by the orientation estimator —
  pixel-space values are multiplied by `meters_per_pixel` before being
  packed into the state.
- `VehicleState.centroid` stays in (calibrated) grid units; the exporter
  passes it through unchanged into the SSAM record.
- The exporter is constructed with `scale = meters_per_pixel`. That value
  goes into the `DIMENSIONS.Scale` field, telling readers how to recover
  real metres from the stored coordinates.

### Worked example

Camera calibrated at `meters_per_pixel = 0.05`. A car with a 60-pixel
bounding box, centroid at pixel `(800, 500)`, moving at 200 pixels/sec.

- Stored coordinates: `centroid = (800, 500)` in the file (abstract).
- Stored `Length = 60 × 0.05 = 3.0` metres (un-scaled).
- Stored `Speed = 200 × 0.05 = 10.0` m/s (un-scaled).
- `DIMENSIONS.Scale = 0.05`.
- Reader recovers `real centroid = (800 × 0.05, 500 × 0.05) = (40, 25)`
  metres.

All three quantities are physically meaningful with a single
`meters_per_pixel` constant — no homography matrix required for the
sizes-and-speeds part. Homography is still needed for stabilisation and
for non-nadir perspective correction.

### Relation to MVP1.75

For drone footage the `meters_per_pixel` factor is computable from
metadata alone via the Ground Sample Distance formula — no vision
algorithms required. That sub-deliverable is split into its own
milestone in `src/tratrac/calibration/GSD_CALIBRATION.md` and ships **before** MVP2.

After MVP1.75 lands:

- `DIMENSIONS.Scale` is already populated correctly.
- `Length`, `Width`, `Speed`, `Acceleration` already carry real metric
  values.
- The orientation estimator already produces metric `VehicleState`s.

MVP2 then adds what the GSD shortcut **cannot** provide:

- **Stabilisation** — when the drone (or camera) translates between
  frames, vehicle velocities in pixel space include the camera's motion.
  GSD doesn't know about this; SuperPoint + LightGlue do.
- **Perspective correction** — when the gimbal is not at −90° (true
  nadir), distance-per-pixel varies across the frame. A single scalar
  Scale collapses; a homography matrix encodes the variation.
- **Non-drone cameras** — fixed CCTV / mast cameras have no telemetry to
  feed GSD. Homography from in-scene reference distances takes over.

---

## Link / Lane IDs

See `application/ROAD_GRAPH.md`.

- **Link ID** — still hardcoded `0`. MVP2 introduces world-space coordinates
  but no road graph; segment identity arrives in MVP3.
- **Lane ID** — still hardcoded `0`.

Conflict TTC / PET become physically valid in this MVP; conflict
*classification* remains limited to "any conflict counts as same-link".

---

## Implementation (Approach A)

> The sections above are the **conceptual** MVP2 (in-pipeline stabilization +
> projection, SuperPoint + LightGlue). What's actually implemented is a narrower,
> cheaper intermediate that satisfies the load-bearing requirement — metric
> world coordinates in SSAM — for single-/few-anchor bounded scenes, behind ports
> shaped so the multi-anchor version drops in without touching callers. This
> mirrors how MVP1.9 uses ORB as an intermediate before the learned stabilizer.

### Decision: projection is **post-hoc**, not in the pipeline

World projection is **fitted** by `tratrac-preprocess project`, a step that runs against
the anchors/transforms `tratrac-preprocess estimate` already produced — before the
perception run, but outside it, not as part of it. `tratrac-postprocess` (pass 2)
only ever **applies** the already-fitted homography rows it finds in the shared
transforms file; it never fits anything itself. Rationale, consistent with the
project's B-first / post-hoc bias (src/tratrac/infrastructure/export/VIDEO_EXPORT.md,
src/tratrac/application/SMOOTHING.md, and the `post-hoc-rendering-principle` memory):

- Projection is a pure coordinate map over already-recorded measurements — fitting it
  needs no pixels beyond the anchor PNGs already exported, and applying it needs only
  the track record. Anything derivable post-hoc stays out of the live perception loop.
- Fitting is **re-tunable cheaply**: fix a bad correspondence, re-run `project`
  (file-in/file-out, no video, no re-detection) — it replaces the homography rows in
  place rather than appending, so the same command is safe to re-run as many times as
  the operator tweaks `calibration.json`.
- `tratrac-postprocess` still composes with the existing post-hoc stages: read record
  → *(filter exclusion)* → **apply projection** → smooth → export. The projection
  slots in **before** smoothing so the Kalman/RTS de-jitter runs in world metres.

This keeps the perception run byte-identical to MVP1.75 when the transforms file carries
no homography rows (a scale-only, image-space run).

### The transform-coordinates-not-pixels invariant (again)

As in MVP1.9, we project **track points, never the video**. The homography maps the
stabilized image point of each observation onto the ground plane; the detector and
tracker never see a warped frame. Warping pixels would resample/balloon the far
field and feed the detector unnatural images — the projection is applied to
coordinates only, downstream of tracking.

### Two homographies, one world map

There are two distinct transforms and they must not be conflated:

- **Ego-motion** (`image → global`, automatic, MVP1.9): the ORB estimator already
  collapses a moving-camera clip into one continuous global frame. The track record
  is written **in that global frame**.
- **World projection** (`global → metric`, this MVP, needs external ground truth):
  one homography fitted from operator-supplied `image ↔ world` correspondences.

Because stabilization already reduces every frame to **one** global frame, only
**one** world homography is needed: `world = H · global_point`. Correspondences
authored on an anchor frame are first lifted into the global frame by that anchor's
pose (`pose(reference_frame).apply(image_point)`) — the *same* anchor-manifest
mechanism exclusion zones use (src/tratrac/application/EXCLUSION_ZONES.md) — then the homography is fit in global
coordinates.

### Components (where each piece lives — onion layers)

| Layer | Artifact | Responsibility |
| --- | --- | --- |
| `domain/world.py` | `Correspondence`, `Calibration` | pure value objects (an `image`/`world` pair + its `reference_frame`) |
| `domain/ports.py` | `CoordinateTransform` Protocol | `apply(point, frame_index) -> Point2D`, `reverse(point, frame_index) -> Point2D` |
| `infrastructure/transform/records.py` | `HomographyFunction`, `HomographyCodec` | the row's `function` payload: pure projection math (numpy only — the projective multiply + perspective divide) + its JSON encode/decode |
| `application/coordinate_transforms.py` | `TransformTable`, `local_scale_at` | the one generic reader: group rows by `frame_index`, filter by which `zone` contains the query point, delegate to that row's `function` |
| `infrastructure/world/calibration.py` | `load_calibration`, `compute_homography` | sidecar-JSON reader + the cv2 homography **fit** (the only cv2 in the MVP2 path) |
| `cli_preprocess.py` | `project` subcommand, `_fit_whole_scene`, `_fit_per_plane` | composition root for *fitting*: load calibration → lift correspondences to global → fit → replace the homography rows in the shared transforms file |
| `cli_postprocess.py` | `_project_to_world`, `_project_observation` | composition root for *applying* an already-fitted `TransformTable` to the recording — no fitting happens here |

### Minimal-blast integration: project, then reuse the existing path

The projected recording is stamped `scale = 1.0` and handed to the **unchanged**
smoother/exporter. Because the coordinates are already metric, `DIMENSIONS.Scale`
becomes `1.0` and `build_state` / `SsamTrjExporter` run exactly as before — no new
branch in the smoothing or export code. A no-calibration run is the pre-MVP2
image-space path, untouched.

- **Centroid**: `H` applied to `(cx, cy)`.
- **Dimensions**: the bbox **extent points** (`cx ± w/2`, `cy ± h/2`) are each
  projected and the world distances measured — so `Length`/`Width` are metric and
  account for local perspective, not just a single scalar.

### Smoother noise units

The Kalman defaults (`pos_noise`, `jerk`) are tuned in **pixels**. Smoothing in
world metres with pixel-tuned noise would mis-scale the filter. Fix: multiply
`pos_noise` by the homography's **local scale** `s` at a representative point
(metres-per-pixel, from `local_scale_at`) and `jerk` by `s²`. This preserves the
Q/R *ratio* (hence the filter's behaviour) while matching the world magnitudes —
the filter de-jitters identically, just in metres.

### Multi-anchor projection (Group C3)

The seams described below (written when only Approach A existed) let the multi-anchor
projector drop in **behind the same row model**, with no caller changes — and that's exactly
what happened, though it has since been folded into the unified transforms file (see
`src/tratrac/infrastructure/transform/TRANSFORM_SINK.md`) rather than staying a dedicated class:

- `TransformTable.apply` already took `frame_index`; fitting (`cli_preprocess._fit_whole_scene`)
  picks the nearest anchor's homography (nearest by frame-index distance — a deliberate hard
  switch, not an interpolated blend between neighboring anchors' homographies — see the
  function's docstring for why interpolation isn't the simple choice it sounds like for
  projective transforms) and materializes it as one `homography` row per frame, `zone = whole
  canvas`.
- `Calibration`/`Correspondence` already carried `reference_frame` per correspondence, so a
  calibration spanning many anchors parsed before this landed — only the *fitter* changed:
  `_fit_whole_scene` groups correspondences by `reference_frame` and fits one `H` per group,
  then assigns each real frame's row the nearest group's matrix. A single-anchor (or static)
  calibration is simply the one-group case — there is no separate code path for it, and no
  separate "per-anchor" homography *kind*: the wire format has exactly one `homography` type,
  differentiated only by what `zone` a row carries (see "One file, one row model" in
  `TRANSFORM_SINK.md`).
- The anchor-manifest lift (`pose(reference_frame)`) was already in place and needed no
  changes.

**Not implemented:** each anchor's `pos_noise`/`jerk` scale conversion is computed
**per observation** (`cli_postprocess._representative_local_scale` averages each observation's
own local scale, not a pre-averaged point — see the fix note in `application/SMOOTHING.md` if
present), so a track crossing an anchor boundary mid-life still gets one averaged
`(pos_noise, jerk)` pair for its whole life rather than a value that itself varies mid-track.
Full pose-interpolated control regions (the "Approach D" end of the original comparison) remain
unimplemented — the nearest-anchor hard switch was judged sufficient unless real footage shows
a visible seam.

### Multi-homography plane projection (Group C5, MVP3)

Fitting (`cli_preprocess._fit_per_plane`) selects a homography by **spatially classifying the
point itself** against elevation-plane zones (ground, bridge, overpass, ...; `--plane-zones`,
`application/ROAD_GRAPH.md`), rather than by `frame_index` — plane membership is *where* a
point is, not *when* it was observed, which is why this is a genuinely different selection axis
from the per-anchor path above, not a variant of it. It is still the same `homography` row
*kind* either way; what differs is only which `zone` polygon each fitted row carries (the
plane's own global polygon, materialized once per frame, instead of the whole-canvas rectangle):

- `_fit_per_plane` classifies each calibration correspondence's global-mapped image point
  against the same plane zones the projector will later use, groups correspondences by the
  resulting plane id, and fits one `H` per group — the same "classify with the exact rule the
  consumer will use" principle `application/ROAD_GRAPH.md` uses for Link/Lane, just feeding a
  homography selector instead of a `VehicleState` field. `plane_zones.json` is needed only here,
  at fit time — the fitted row embeds the resulting polygon directly, so nothing downstream
  (`TransformTable`, `tratrac-postprocess`) ever needs `plane_zones.json` again.
- `0` is **not** a reserved "unknown" sentinel for plane ids the way it is for Link/Lane — it's
  a legitimate label (e.g. the ground plane), since plane assignment is purely internal
  (never written into `VehicleState`) and a projector must resolve *some* homography for every
  point, so there's no safe place to default to "unclassified." A point that falls outside
  every explicit plane zone still needs a `plane_id: 0` calibration group to be projectable;
  `_fit_per_plane` raises if a classified plane has no drawn zone polygon, and `TransformTable`
  itself raises (naming the point and frame) if a query point matches no row's zone at all —
  there's no principled distance metric between elevation surfaces the way there is between
  anchors in time, so neither layer guesses a "closest" plane.
- `--plane-zones` requires `--calibration` and **supersedes** the anchor-based fit entirely when
  given — a calibration spanning both multiple anchors and multiple planes at once is not
  composed; an anchor's correspondences are pooled per plane regardless of which anchor they
  came from. This is a real, documented limitation, not a silently-swept edge case: a
  moving-drone shoot over a grade-separated site needs one or the other landing (Group C3/C5
  combined) to be fully served, which is unstarted.
- Same per-observation `pos_noise`/`jerk` scale-averaging note as the per-anchor path above — a
  track crossing a plane boundary mid-life still gets one averaged noise pair for its whole
  life.

### Operator workflow

```text
tratrac-preprocess estimate VIDEO --background-zones bg.json \
    --out transforms.jsonl --anchors-dir anchors/ \
    --meters-per-pixel 0.05                                  # pass 0 (ego-motion + scale, no detector)
# operator draws image↔world correspondences on an anchor PNG -> calibration.json
tratrac-preprocess project --transforms transforms.jsonl \
    --calibration calibration.json [--plane-zones plane_zones.json]   # pass 0.5 (fit + replace homography rows)
tratrac --config run.toml   # input.transforms_in = transforms.jsonl  # pass 1 (perception)
tratrac-postprocess run.parquet --out run.trj \
    --transforms transforms.jsonl                            # pass 2 (apply projection + smooth)
```

`calibration.json` schema (≥ 4 correspondences; `reference_frame` defaults to `0`
for a static camera):

```json
{ "correspondences": [
    { "reference_frame": 0, "image": [1820, 640], "world": [0.0, 0.0] },
    { "reference_frame": 0, "image": [1900, 642], "world": [12.0, 0.0] }
] }
```

### Automatic calibration from road geometry (proposal-only, Group C4)

The operator workflow above requires manually clicking image↔world correspondence points —
URBAn's "Visual calibration tool" issue already flags this as a UX gap (hand-authored JSON). A May 2026
pipeline demonstrates deriving the road-plane homography **automatically from visible road
geometry** — lane markings, road borders, crosswalks — instead of manual correspondences. Its
caveats match what this doc already documented independently, not new information: far-field
vehicles are most sensitive to homography error, and manual validation currently outperforms
fully-automatic calibration.

**Group C4's repo-boundary question — does TraTrac ship only correspondence-proposal, with
interactive confirm/adjust living in URBAn — resolves to yes.** `scripts/propose_calibration.py`
auto-proposes candidate image-side points from road markings (Canny + Hough line detection,
filtered by brightness *and* local contrast against the surrounding surface — see
`src/tratrac/application/AUTO_CALIBRATION.md` for the full design, the real-footage validation
run, and why brightness alone isn't enough); a human (via URBAn, or by hand) confirms/adjusts
and supplies `world` coordinates before the result becomes a usable `calibration.json`. The
single-homography math itself (confirmed against the literature as still the right approach for
piecewise-planar road surfaces, no qualitatively better alternative exists) is unchanged. See
`docs/TECH_STACK.md`'s Geometry section.

**Source:** [Mobile Traffic Camera Calibration from Road Geometry for UAV-Based Traffic Surveillance (2026)](https://arxiv.org/abs/2605.11900)

### SSAM output frame (world-extent normalization)

The projected coordinates are **metric**, so the `.trj` must not describe them against a
pixel-sized canvas. After projection the post-process step:

- **translates** all coordinates to a non-negative origin (the operator's absolute world
  origin is arbitrary in Approach A and irrelevant to the translation-invariant conflict
  analytics — TTC/PET depend only on relative positions and closing speeds);
- sets `DIMENSIONS.{MaxX,MaxY}` to the **world extent in metres** (the data bounding box,
  padded by the largest projected vehicle so the smoother's front/rear bumpers stay
  in-bounds), not the pixel grid;
- keeps `Scale = 1.0`.

This matters because the SSAM exporter flips Y about `MaxY × Scale` (src/tratrac/infrastructure/export/SSAM_FORMAT.md). With the
pixel height left in place and `Scale = 1.0`, an external SSAM reader would see metric
coordinates flipped about the pixel height and bounded by a pixel-sized box — internally
self-consistent (our own `read_trj` round-trips) but wrong for any third-party consumer.
Sizing the bounds to the world extent makes the file correct stand-alone. (Image-space /
MVP1.75 output is unchanged: there the pixel grid *is* the coordinate space.)

### Validated on real footage

Exercised end-to-end on a 20 s near-nadir window of a roundabout clip (`rotonda`,
moving/zooming drone, ego-motion ON, one keyframe anchor): the two-pass runs cleanly on
real BoT-SORT tracks, the `.trj` carries metric world coordinates with `Scale = 1.0` and
world-extent bounds, `validate_trj` reports ~100 % compliance, and the trajectories form
the roundabout ring. Speeds (≈18 km/h median, ≈32 km/h max) and sizes (≈4.6 m length) are
physically credible. Because that clip is near-nadir and the correspondences were derived
from the GSD estimate, the fitted homography was ≈ pure scale, so MVP2 reproduced the
MVP1.75 baseline **to the digit** — a consistency check, **not** a validation of
perspective correction. Stressing the projective part needs a non-nadir clip or surveyed
ground points.

### Known limitations (carried, not hidden)

- **Single homography** ⇒ accurate only where the ground stays one plane within the
  fit's neighbourhood; bridges/overpasses need MVP3 multi-homography.
- **Exact 4-point fits do not detect degeneracy** — cv2 returns a finite-but-garbage
  matrix for collinear correspondences (only the > 4 RANSAC path returns `None`).
  Four well-spread, non-collinear points are the operator's responsibility (see the
  `compute_homography` docstring).
- **Stabilization is still ORB** (MVP1.9), not SuperPoint + LightGlue — the projection
  inherits whatever drift the ego-motion fit carries. The SuperPoint upgrade is
  Group B3 in GitHub Issues.
