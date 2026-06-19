# Final-Product Polish — Deferred Robustness Upgrades

---

## What This Is

A backlog of **deliberate "good enough for now, upgrade later" decisions**: places
where we shipped a cheaper implementation behind a stable port and recorded the
intended final-product replacement here. Each entry names the seam, the current
adapter, the target adapter, and *why* the upgrade is worth it.

This is not the MVP roadmap (`05_mvp1.md` … `11_mvp7.md`) — those add *new
capabilities*. This file tracks *quality upgrades to existing capabilities* that
were intentionally deferred. An item leaves this file when it ships.

The recurring pattern: a port isolates the seam, so each upgrade is a one-adapter
swap with no change to the domain, the pipeline, or the other adapters.

---

## Items

### 1. Ego-motion estimation: ORB → SuperPoint + LightGlue

| | |
| --- | --- |
| Port | `EgoMotionEstimator` (stabilization seam, **shipped in MVP1.9** — see `05_75_mvp1_9.md`) |
| Ships with | ORB + RANSAC similarity adapter (`OrbEgoMotionEstimator`) — no new dependency |
| Target | SuperPoint + LightGlue adapter (`kornia` or the standalone `lightglue`) |
| Trigger to upgrade | Measured stabilization error on real aerial footage exceeds tolerance |

**Why ORB ships first (chosen over ECC).** The interim stabilizer is **ORB**, not
the ECC adapter this entry originally proposed. Both are zero-new-dependency
(`cv2` is already present), but they differ on the axis that matters for this
domain — **moving foreground**, where much of the frame is the vehicles we track:

- **ORB is feature-based**, so it produces explicit correspondences and **RANSAC
  rejects the moving-vehicle matches as outliers**.
- **ECC is intensity-based** and optimizes over *all* pixels with no outlier
  rejection; it cannot ignore the cars, and on bare asphalt the high-contrast
  pixels it locks onto are often the vehicles themselves.

ECC's only edge was that BoT-SORT already runs it internally (`cmc_method=ecc`) —
a familiarity argument, not a quality one. So MVP1.9 ships ORB as the cheap
interim and keeps measure-before-optimize: quantify how much camera motion
actually pollutes exported trajectories, and only pay for the heavy learned
estimator if the gap warrants it.

**Why the upgrade to SuperPoint + LightGlue.** Even ORB's handcrafted features
thin out on the worst aerial footage — low-texture, repetitive (lane markings,
asphalt), motion-blurred. SuperPoint + LightGlue match *learned* keypoints, far
more robust there, and LightGlue's attention rejects ambiguous matches that ORB's
local ratio test would accept. This is why `03_tech_stack.md` and `06_mvp2.md`
name SuperPoint + LightGlue as the target stabilizer.

**Why it is a clean swap.** The `EgoMotionEstimator` port returns a `Transform2D`
per frame; nothing downstream knows or cares how it was estimated. Replacing ORB
with SuperPoint + LightGlue is a single new adapter wired in the CLI — no domain,
pipeline, exporter, or test changes outside the new adapter and its own tests.

> The MVP1.9 ORB slice has landed; `06_mvp2.md` still specifies SuperPoint +
> LightGlue from the start — its stabilization box is now an *upgrade* of MVP1.9's
> ORB adapter, not a from-scratch addition.

---

### 2. Kinematics: EMA finite-difference → Kalman/RTS smoothing

| | |
| --- | --- |
| **Seam** | `OrientationEstimator` port (the ORIENT step) |
| **Ships with** | `EmaOrientationEstimator` — raw centroid + windowed finite differences |
| **Target** | constant-acceleration **Kalman/RTS** smoother (`application/kalman.py`) |
| **Trigger** | acceleration/jerk noise in the exported `.trj` (the standing accel-noise issue) |

Both paths have **shipped** (`22_smoothing.md`): the offline two-pass (`export.tracks` +
`tratrac-smooth` forward+RTS) and the inline forward `KalmanOrientationEstimator`
(`orientation.method = kalman`) for the streaming `.trj`. Clean swap: the port returns
`VehicleState` per frame; nothing downstream knows whether kinematics came from EMA, a
forward Kalman, or the RTS post-pass. What remains genuinely deferred is **tuning** —
picking `pos_noise`/`jerk` against real footage using the `validate_trj.py` jerk metric —
and a possible **fixed-lag** middle ground if the offline pass is ever too heavy.
