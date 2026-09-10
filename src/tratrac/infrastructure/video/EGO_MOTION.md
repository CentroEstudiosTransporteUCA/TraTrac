# Ego-Motion Estimation: Keyframe-Anchored ORB (MVP1.9, intermediate)

> **Status — ✅ Shipped** (optional, off by default). The MVP number is a capability ID, not
> execution order — see the roadmap reconciliation in `docs/ROADMAP.md`. An intermediate
> "keep if good enough" shortcut before MVP2's learned stabilizer + world projection.

---

## What This Is

A **basic-but-good-enough ego-motion compensation** step that removes drone
motion from the exported trajectories. It slots between MVP1.75 (metric
sizes/speeds from GSD) and MVP2 (world projection), and is explicitly an
**intermediate, measure-then-keep** deliverable: ship the cheap feature-based
estimator behind the `EgoMotionEstimator` port, quantify how much it improves
trajectories on real footage, and keep it only if the gain justifies the cost. If
not, the *port* survives as the seam MVP2 plugs the learned stabilizer into (see
`docs/BACKLOG.md` item 1).

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
intermediate, recorded as `docs/BACKLOG.md` item 1.

---

## Vehicle-Masked Feature Extraction

RANSAC alone is not enough on low-texture aerial footage: the background is
feature-poor (bare asphalt), so coherently-moving vehicles can become the inlier
majority and bias the fit. The fix masks detected vehicles out of
`detectAndCompute`. Where the mask comes from is pluggable — see "Detector-free
ego-motion" below — but a live `tratrac` run's default source
(`DetectionMaskSource`) still reuses the pipeline's detections via the
`DetectionObserver` port (no second detector).

With coordinate stabilization this is simpler than before: the detector now runs on
the **raw** frame, so the detections are already in raw coordinates and mask the
**same** frame's feature extraction directly — **no inverse remapping, no one-frame
lag**. The pipeline calls `observe(detections)` right after detection and before
`estimate(frame)`, so the mask reflects exactly the frame being estimated.

Masking is intrinsic to the estimator (`mask_source` is a required constructor
argument, not a config key). It addresses the bias source but not every failure
mode — a genuinely static camera has no ego-motion to remove, so `ego_motion`
should stay **off** there regardless (ORB would otherwise inject phantom motion
into the *coordinates* of parked cars).

---

## Detector-free ego-motion (`MaskSource`, `tratrac-stabilize`)

**The idea.** `DetectionMaskSource` couples masking to the live detector — which is
exactly why anchor discovery couldn't be a separate pass without either losing mask
quality or running the detector twice (both explored and rejected; see
"Why not a genuinely separate pre-pass" below). Swapping the mask *source* removes
that coupling: if the mask instead comes from operator-drawn polygons, ORB needs no
detector at all, ever.

**`MaskSource`** (`infrastructure/video/ego_motion_orb.py`) is the seam this
required: `observe(detections)` / `mask_for(frame_index, height, width) ->
NDArray[np.uint8] | None`. `OrbEgoMotionEstimator` takes one at construction instead
of building the mask inline.

- `DetectionMaskSource` — today's behaviour, extracted unchanged: rasterizes the
  most recently `observe`d detections' bounding boxes, `None` when there are none
  yet. `observe` is real (stores the detections).
- `BackgroundZoneMaskSource(zones: BackgroundZones)` — reads a `domain/background.py`
  `BackgroundZones` collection instead: each zone is a polygon plus the frame it
  starts applying from ("use this mask from here until a later entry supersedes
  it" — not tied to ORB's own re-anchor points). `mask_for` resolves the most
  recent zone at or before `frame_index` (falling back to the earliest zone for a
  frame before the first entry) and rasterizes it as the **keep** region — the
  opposite construction from `DetectionMaskSource`'s **exclude**-the-boxes mask.
  `observe` is a no-op: the mask is entirely operator-authored, no detections
  needed.

`background_zones.json` (`infrastructure/background/json.py`,
`src/tratrac/infrastructure/background/BACKGROUND_ZONES.md`) is the sidecar an
external tool (out of scope for this repo — same footing as `calibration.json` and
`exclusion_zones.json` today) produces: an operator watches the video and draws the
region safe for ORB feature extraction, redrawing only when the view has changed
enough to warrant it.

**`tratrac-stabilize`** (`cli_stabilize.py`) is the new tool this unlocks: walks a
clip once with `OrbEgoMotionEstimator(mask_source=BackgroundZoneMaskSource(zones))`
— no detector, no tracker — writing the same per-frame transform sidecar and anchor
manifest a stabilized `tratrac` run would. `tratrac`'s `[ego_motion]` then gains
`transforms_in` (`""` = off, the default): when set, `cli.py` loads the table via
`read_transforms` into a `PerFrameTransform` and wraps it in
`PrecomputedEgoMotionEstimator` (`infrastructure/transform/sink.py`) — an
`EgoMotionEstimator` that's a plain per-frame lookup, no ORB call — instead of
constructing `OrbEgoMotionEstimator`. The detector then runs exactly once, ever,
against already-known ego-motion. `export.anchors_dir` is rejected alongside
`transforms_in` at config-resolve time: a precomputed run discovers no new anchors
of its own (`tratrac-stabilize` already produced them).

Revised operator workflow:

```
[external tool] operator watches VIDEO, draws background zones -> background_zones.json
tratrac-stabilize VIDEO --background-zones background_zones.json \
    --out transforms.jsonl --anchors-dir anchors/          # detector-free
[external tool] operator draws exclusion zones / world-projection
    correspondences on anchors/*.png -> zones.json / calibration.json
tratrac --config run.toml   # ego_motion.transforms_in = transforms.jsonl
    # detector runs exactly once here; ego-motion already resolved
tratrac-postprocess run.parquet --out run.trj \
    --calibration calibration.json --anchors anchors/manifest.json   # unchanged, still post-hoc
```

The old workflow (no `background_zones.json`, `ego_motion.transforms_in = ""`)
still works unchanged — `tratrac-stabilize` is additive, not a replacement.

**Why not a genuinely separate pre-pass, before this landed.** Two things were
tried and rejected before settling on operator-authored zones: (1) a cheap
*detector-free* self-referential masking scheme (fit unmasked, treat RANSAC's own
outliers as the mask) — real technique in the literature, but the same
sparse-background/dense-foreground failure mode that motivated detection-based
masking in the first place (RANSAC's *first*, unmasked pass can itself get captured
by the vehicle majority on bare asphalt) means it needs validation this project
hasn't done, and domain-specific literature for exactly this footage (dense urban
traffic, aerial) converges on detection-based masking, not self-referential
schemes; (2) reusing the *live-detector's* mask in an earlier, separate pass would
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
  (stateful; returns the current frame → global transform) and `DetectionObserver`.
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
  `transforms_in` side: an `EgoMotionEstimator` that's an exact `PerFrameTransform`
  lookup, no ORB call.
- `cli_stabilize.py` (`tratrac-stabilize`) — the detector-free pre-pass: walks a
  clip once with `BackgroundZoneMaskSource`, writing the transforms sidecar +
  anchor manifest via the same sinks a stabilized `tratrac` run uses.

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
   a static camera it manufactures phantom motion — now into coordinates, so keep
   `ego_motion` off there. SuperPoint + LightGlue is the upgrade if measurement
   demands it.

---

## Config Surface (Zero-Defaults Rule)

Per `src/tratrac/application/CONFIG_DESIGN.md`, every key is mandatory and "off is explicit". The
`[ego_motion]` section has a **required** `enabled` boolean; when true the ORB
parameters, `min_anchor_overlap` (the re-anchor threshold, in `(0, 1)`), **and**
`transforms_in` (`""` = live ORB, the default; else a `tratrac-stabilize`
transforms file — see "Detector-free ego-motion" above) are required, when false
they may be absent — the same conditional pattern as `[calibration]`'s one-of. The
ORB parameters stay required even when `transforms_in` is set (unused in that path)
rather than adding a second conditional-requirement branch.

---

## Persisting the Per-Frame Transform (`TransformSink`)

Split into its own doc: `src/tratrac/infrastructure/transform/TRANSFORM_SINK.md` —
how the per-frame global↔raw transform gets persisted as a sidecar so offline
consumers (`tratrac-render`) can map stabilized coordinates back onto the raw video.

---

## Relation to MVP2

MVP2 keeps the `EgoMotionEstimator` port and replaces the adapter:

- **Stabilizer upgrade** — ORB → SuperPoint + LightGlue (`docs/BACKLOG.md` item 1),
  if the ORB measurement shows it is needed.
- **World projection** — a homography from the (now ego-motion-free) image plane to
  metric world coordinates, making SSAM positions physically meaningful and giving
  bounded, real-world coordinates.

If MVP1.9's ORB measures as "not worth keeping", the adapter is dropped but the
port, `Transform2D`, and the coordinate-stabilization pipeline seam remain — MVP2
inherits a ready structure.
