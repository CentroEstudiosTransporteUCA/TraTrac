# Ego-Motion Estimation: Keyframe-Anchored ORB (MVP1.9, intermediate)

Always attempted by `tratrac-preprocess estimate`, a no-op when a clip has no camera motion to
correct — not a config toggle, see "Config Surface" below. An intermediate
"keep if good enough" shortcut before MVP2's learned stabilizer + world projection; the MVP
number is a capability ID, not execution order.

---

## What This Is

A **basic-but-good-enough ego-motion compensation** step that removes drone
motion from the exported trajectories. It slots between MVP1.75 (metric
sizes/speeds from GSD) and MVP2 (world projection), and is explicitly an
**intermediate, measure-then-keep** deliverable: ship the cheap feature-based
estimator behind the `EgoMotionEstimator` port, quantify how much it improves
trajectories on real footage, and keep it only if the gain justifies the cost. If
not, the *port* survives as the seam MVP2 plugs the learned stabilizer into (see
the B3 SuperPoint+LightGlue issue in GitHub Issues).

It is **not** world projection. Coordinates remain image-space and therefore still
non-physical per the coordinate-semantics invariant (`src/tratrac/domain/ARCHITECTURE.md`).
What changes is that they are image-space **in a single stabilized (keyframe-chained
global) frame** instead of per-frame raw pixels, so frame-to-frame displacement —
and therefore the Speed / Acceleration the orientation estimator derives — no
longer contains the camera's own motion.

---

## Design Decision: Stabilize Coordinates, Not Pixels

Ego-motion is removed by transforming **detection coordinates**, not by warping
the image. The detector and tracker run on the **raw, full-resolution frame**; the
estimated transform is applied to each detection's bounding box *before* tracking,
mapping it into the stabilized frame.

| Approach | Where | Consequence |
| --- | --- | --- |
| **Coordinate stabilization (chosen)** | Detector + tracker run on the raw frame; the ego-motion transform is applied to detections before tracking | Detection sees the full, sharp, native frame — nothing is ever cropped; trajectories are still ego-motion-free |
| Pixel warp (rejected) | A `VideoSource` decorator warps each frame into a fixed reference before the pipeline sees it | **Resampling into a fixed-size canvas crops** everything that drifts off frame 0 to black, starving the detector late in a panning/zooming clip; also blurs and rescales the cars |

**Why the reversal.** An earlier slice warped the *pixels* into a fixed frame-0
viewport (`StabilizedVideoSource`). On a real moving+zooming clip (rotonda) the
cumulative transform drifted ~72% of frame width and the warped content slid out
of the fixed canvas — by the end the frame was mostly black and the detector found
almost no cars. The flaw is intrinsic to warping into a bounded image: a picture
has edges, so anything that no longer fits is discarded. A *point* has no edges —
mapping a detection at (1800, 1000) to (2400, −150) in the global frame is a fine
value. So we transform points, not pixels. This is conceptually a poor-man's
version of what MVP2 does (project coordinates to a stable frame), using the
similarity transform we already estimate. (The previously-"rejected"
centroid-compensation idea was right in spirit; doing it at the *detection* level,
before tracking, makes the tracker ego-motion-free too.)

---

## The Transform Model

A **4-DOF similarity** (translation + rotation + uniform scale), estimated with
`cv2.estimateAffinePartial2D` (RANSAC). Right model for near-nadir aerial: full
affine (6-DOF) absorbs residual vehicle motion into a skew; full perspective
homography (8-DOF) is deliberately MVP2's job.

`domain/geometry.py`'s `Transform2D` is the pure value object: the six affine
coefficients with `identity()`, `apply(Point2D)`, `compose`, `inverse`, and
`scale` (`sqrt` of the determinant — the uniform scale factor, used to resize a
box when mapping a detection between frames). No cv2 matrix leaks into the domain.

### Keyframe anchoring (why not a fixed frame 0)

Matching every frame against a *fixed* frame 0 has two problems: ORB
correspondences thin out as the view diverges from frame 0, and per-frame
composition compounds error. Instead the estimator matches each frame against a
**keyframe anchor** and re-anchors when too little of the anchor is still visible:

- Match the current frame against the **anchor** → `L` (current → anchor).
- The anchor carries a **global pose** `A` (anchor → global frame). The current
  frame's returned transform is `G = A ∘ L` (current → global).
- Measure `clipped_overlap_fraction(L)` — the share of the anchor rectangle still
  covered by the mapped current frame. When it drops below `min_anchor_overlap`,
  promote the current frame to the new anchor with global pose `G`.

This keeps matching against a *recent, high-overlap* frame (robust ORB, no
per-frame error compounding — drift accrues only at the sparse re-anchor
compositions) while the composed global pose stays **continuous** across
re-anchors. Continuity matters: the tracker associates in the global frame, so a
coordinate jump at a re-anchor would break every track ID. The anchor is purely an
*estimation reference*, not a reset of the output frame. The anchor/overlap/chain
policy lives in the pure, unit-tested `_AnchorChain`; the cv2 feature work stays in
`OrbEgoMotionEstimator`.

Note: keyframe anchoring bounds the *conditioning of the estimate*, not the
*magnitude* of coordinates — a car far from frame 0 still has large/negative global
coordinates. That is harmless for points and tracking (see SSAM caveat below).

---

## Why Feature-Based ORB, Not Intensity-Based ECC

The dominant failure mode of aerial traffic footage is **moving foreground**: much
of the frame is the vehicles we track. A stabilizer must estimate background motion
*while ignoring the vehicles*.

- **ORB (feature-based)** gives explicit point correspondences, so **RANSAC rejects
  moving-vehicle matches as outliers**.
- **ECC (intensity-based)** optimizes over all pixels with no outlier rejection; it
  cannot be told to ignore the cars, and on bare asphalt it often locks onto them.

Both are zero-new-dependency (`cv2` already present). `docs/TECH_STACK.md` /
`src/tratrac/application/WORLD_PROJECTION.md` name **SuperPoint + LightGlue** as the eventual target; ORB is the
intermediate, tracked as Group B3 in GitHub Issues.

---

## Vehicle-Masked Feature Extraction

RANSAC alone is not enough on low-texture aerial footage: the background is
feature-poor (bare asphalt), so coherently-moving vehicles can become the inlier
majority and bias the fit. The fix masks vehicles out of `detectAndCompute`. Where
the mask comes from is pluggable behind the `MaskSource` Protocol, but there is
only one implementation now: `BackgroundZoneMaskSource`, reading operator-authored
polygons — see "Detector-free ego-motion" below. (An earlier revision also had a
`DetectionMaskSource`, masking from a live detector's own detections; it was
removed once `tratrac` stopped estimating ego-motion itself, since nothing
constructed it any more — see "Detector-free ego-motion is now the only path".)

Masking is intrinsic to the estimator (`mask_source` is a required constructor
argument, not a config key). It addresses the bias source but not every failure
mode — a genuinely static camera has no ego-motion to remove, so
`tratrac-preprocess estimate` should simply omit `--background-zones` there
regardless (ORB would otherwise inject phantom motion into the *coordinates*
of parked cars).

---

## Detector-free ego-motion is the only path (`MaskSource`, `tratrac-preprocess`)

**The idea.** Masking from a *live detector's* own detections would couple
ego-motion estimation to detection — which is exactly why anchor discovery
couldn't be a separate pass without either losing mask quality or running the
detector twice (both explored and rejected; see "Why not a genuinely separate
pre-pass" below). Swapping the mask *source* removes that coupling entirely: the
mask comes from operator-drawn polygons instead, so ORB needs no detector at all,
ever, and `tratrac` itself never estimates ego-motion — it only ever reads an
already-built transform table.

**`MaskSource`** (`infrastructure/video/ego_motion_orb.py`) is the seam this
required: `observe(detections)` / `mask_for(frame_index, height, width) ->
NDArray[np.uint8] | None`. `OrbEgoMotionEstimator` takes one at construction
instead of building the mask inline. `BackgroundZoneMaskSource(zones:
BackgroundZones)` is the only implementation: each zone is a polygon plus the
frame it starts applying from ("use this mask from here until a later entry
supersedes it" — not tied to ORB's own re-anchor points). `mask_for` resolves the
most recent zone at or before `frame_index` (falling back to the earliest zone
for a frame before the first entry) and rasterizes it as the **keep** region.
`observe` is a no-op: the mask is entirely operator-authored, no detections
needed. (An earlier revision also had `DetectionMaskSource`, masking from a live
detector's own detections; it was removed once nothing constructed it any more.)

`background_zones.json` (`infrastructure/background/json.py`,
`src/tratrac/infrastructure/background/BACKGROUND_ZONES.md`) is the sidecar an
external tool (out of scope for this repo — same footing as `calibration.json` and
`exclusion_zones.json` today) produces: an operator watches the video and draws the
region safe for ORB feature extraction, redrawing only when the view has changed
enough to warrant it.

**`tratrac-preprocess estimate`** (`cli_preprocess.py`) is the tool that runs
this: walks a clip once with
`OrbEgoMotionEstimator(mask_source=BackgroundZoneMaskSource(zones))` — no
detector, no tracker — writing an ego-motion row *and* a GSD-scale row per
frame into one transforms file (`--out`), plus the anchor PNGs
(`--anchors-dir`, no separate manifest: an anchor's pose is already that same
file's row at that frame index — see
`src/tratrac/application/EXCLUSION_ZONES.md`). `--background-zones` is itself
optional: omit it for a static camera and only the scale rows get written (no
ego-motion rows at all). `tratrac`'s `input.transforms_in` is a **required**
key naming this file — there is no `[ego_motion]` section, no `enabled` toggle,
and no live-ORB fallback: `cli.py` loads the ego-motion-only rows via
`read_ego_motion` into a `TransformTable` and wraps it in
`PrecomputedEgoMotionEstimator` (`infrastructure/transform/sink.py`) — an
`EgoMotionEstimator` that's a plain per-frame lookup, no ORB call, falling
back to the identity when the table has no ego-motion rows at all (the static
case). The detector then runs exactly once, ever, against already-known
ego-motion. See `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md` for
the full unified-row picture (scale and world-projection homography rows live
in the same file, written by `estimate`/`tratrac-preprocess project`
respectively).

Operator workflow:

```
[external tool] operator watches VIDEO, draws background zones -> background_zones.json
tratrac-preprocess estimate VIDEO --background-zones background_zones.json \
    --out transforms.jsonl --anchors-dir anchors/ --meters-per-pixel 0.05
    # detector-free; also resolves + writes the GSD scale rows
[external tool] operator draws exclusion zones / world-projection
    correspondences on anchors/*.png -> zones.json / calibration.json
tratrac --config run.toml   # input.transforms_in = transforms.jsonl
    # detector runs exactly once here; ego-motion already resolved
tratrac-preprocess project --transforms transforms.jsonl \
    --calibration calibration.json   # fits + appends homography rows, still post-hoc
tratrac-postprocess run.parquet --out run.trj --transforms transforms.jsonl
```

**Why not a genuinely separate pre-pass, before this landed.** Two things were
tried and rejected before settling on operator-authored zones: (1) a cheap
*detector-free* self-referential masking scheme (fit unmasked, treat RANSAC's own
outliers as the mask) — real technique in the literature, but the same
sparse-background/dense-foreground failure mode that motivated detection-based
masking in the first place (RANSAC's *first*, unmasked pass can itself get captured
by the vehicle majority on bare asphalt) means it needs validation this project
hasn't done, and domain-specific literature for exactly this footage (dense urban
traffic, aerial) converges on detection-based masking, not self-referential
schemes; (2) reusing a *live-detector's* mask in an earlier, separate pass would
mean either running the detector twice (the "scout + replay = ORB twice" pattern
`EXCLUSION_ZONES.md` already documents rejecting, generalized to the detector) or
accepting an unmasked, lower-quality estimate whose discovered anchor set could
diverge from what the real masked run would produce — invalidating any
correspondences authored against it. Operator-authored zones route around both: no
detector dependency, and (since it *is* the real computation, not an approximation
of it) no anchor-set divergence risk.

---

## Where It Lives

- `domain/geometry.py` — `Transform2D` (+ `scale`), and `clipped_overlap_fraction`
  (shapely-backed polygon intersection/area; pure overlap geometry for re-anchoring).
- `domain/ports.py` — `EgoMotionEstimator`: `estimate(frame) -> Transform2D`
  (stateful; returns the current frame → global transform).
- `infrastructure/video/ego_motion_orb.py` — `OrbEgoMotionEstimator` (ORB →
  ratio-tested Hamming match against the anchor → `estimateAffinePartial2D` RANSAC)
  plus the pure `_AnchorChain`. Exposes `current_transform` for the overlay.
- `application/stabilization.py` — `apply_transform(detection, transform)`: maps a
  detection's box centre exactly and scales its size by `transform.scale`
  (axis-aligned; see caveats). Pure.
- `application/pipeline.py` — owns the per-frame order: detect (raw) → observe →
  `estimate` → `apply_transform` to each detection → track → orient → export. The
  exporter receives the **raw** frame.
- `infrastructure/tracking/boxmot_bot_sort.py` — `compensate_camera_motion`
  (`cmc_method=None` when stabilization is on) so BoT-SORT does not *also* correct
  the already-stabilized boxes.
- `infrastructure/export/overlay_video.py` — maps stabilized coordinates back onto
  the raw frame for drawing via the inverse of `current_transform` (see
  `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`).
- `domain/background.py` — `BackgroundZone`/`BackgroundZones`, the operator-authored
  polygon collection `BackgroundZoneMaskSource` reads.
- `infrastructure/background/json.py` — `load_background_zones`, the
  `background_zones.json` reader (shares `infrastructure/zones.py`'s parser with
  `infrastructure/exclusion/json.py`). See
  `src/tratrac/infrastructure/background/BACKGROUND_ZONES.md`.
- `infrastructure/transform/sink.py` — `PrecomputedEgoMotionEstimator`, the
  `input.transforms_in` side: an `EgoMotionEstimator` that's an exact
  `TransformTable` lookup over the ego-motion-only rows (`read_ego_motion`), no
  ORB call. This is the **only** way `tratrac` gets ego-motion.
- `cli_preprocess.py` (`tratrac-preprocess estimate`) — the detector-free
  pre-pass: walks a clip once with `BackgroundZoneMaskSource` (when
  `--background-zones` is given), writing the transforms file's ego-motion and
  scale rows and the anchor PNGs (no manifest — see
  `src/tratrac/application/EXCLUSION_ZONES.md`).

There is **no `StabilizedVideoSource`** — pixel warping was removed.

---

## Known Limitations (What the Measurement Must Watch)

1. **Unbounded coordinate magnitude.** The global frame is anchored to frame 0, so
   coordinates grow without bound under sustained motion. Harmless for points and
   tracking; only the SSAM export is affected (next item). Keyframe anchoring fixes
   estimation robustness, *not* coordinate magnitude.

2. **SSAM bounds / y-flip.** Stabilized positions can fall outside `[0,W]×[0,H]`, so
   the SSAM y-flip (`height − y`) can go negative and `DIMENSIONS` (kept at `W×H`)
   no longer bounds the data. For MVP1.x image-space export this is already "valid
   but not physically meaningful"; proper world bounds are MVP2's job.

3. **Axis-aligned box approximation.** `apply_transform` transforms the box centre
   exactly and scales its size, but does not re-fit a rotated box. Exact for the
   centroid trajectory (what velocity/heading use) and correctly zoom-normalises
   length/width (fixing the GSD-varies-with-zoom problem); only the never-moved
   bbox-major-axis *fallback* heading is approximate.

4. **ORB robustness on static / low-texture clips.** ORB can still mis-estimate; on
   a static camera it manufactures phantom motion — now into coordinates, so omit
   `--background-zones` from `tratrac-preprocess estimate` there. SuperPoint + LightGlue is the upgrade if measurement
   demands it.

---

## Config Surface (Zero-Defaults Rule)

Per `src/tratrac/application/CONFIG_DESIGN.md`, every key is mandatory and "off is explicit".
There is no `[ego_motion]` section, no `enabled` toggle, and no per-key
"disabled" value: `input.transforms_in` is a plain required path, always —
`tratrac-preprocess estimate` is mandatory before every run, even a fully
static camera (it's the only place GSD scale gets resolved now too). Whether
ego-motion is "on" is entirely a property of that file's *content*: if it has
no ego-motion rows, `tratrac`'s stabilization stage is the identity, with no
config-level distinction from a moving-drone run. The ORB tuning parameters
(`n_features`, `match_ratio`, `min_matches`, `ransac_threshold`,
`min_anchor_overlap`) are not part of `tratrac`'s config at all; they live only on
`tratrac-preprocess estimate`'s CLI, the one tool that actually runs ORB.

---

## Persisting the Per-Frame Transform (`TransformSink`)

Split into its own doc: `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md` —
how the per-frame global↔raw transform gets persisted as a sidecar so offline
consumers (`tratrac-render`) can map stabilized coordinates back onto the raw video.

---

## Relation to MVP2

MVP2 keeps the `EgoMotionEstimator` port and replaces the adapter:

- **Stabilizer upgrade** — ORB → SuperPoint + LightGlue (Group B3, GitHub Issues),
  if the ORB measurement shows it is needed.
- **World projection** — a homography from the (now ego-motion-free) image plane to
  metric world coordinates, making SSAM positions physically meaningful and giving
  bounded, real-world coordinates.

If MVP1.9's ORB measures as "not worth keeping", the adapter is dropped but the
port, `Transform2D`, and the coordinate-stabilization pipeline seam remain — MVP2
inherits a ready structure.
