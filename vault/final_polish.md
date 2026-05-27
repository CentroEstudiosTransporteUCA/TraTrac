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

### 1. Ego-motion estimation: OpenCV ECC → SuperPoint + LightGlue

| | |
| --- | --- |
| Port | `EgoMotionEstimator` (MVP2 stabilization seam) |
| Ships with | OpenCV ECC adapter (`cv2.findTransformECC`) — no new dependency |
| Target | SuperPoint + LightGlue adapter (`kornia` or the standalone `lightglue`) |
| Trigger to upgrade | Measured stabilization error on real aerial footage exceeds tolerance |

**Why the upgrade.** ECC is an intensity-based global image-alignment: it assumes
a mostly-static scene and a single parametric warp. On aerial traffic footage that
assumption frays — much of the frame is moving vehicles (foreground that violates
the static-scene model), and aerial scenes are often low-texture, repetitive
(lane markings, asphalt), and motion-blurred, which is exactly where intensity
alignment is weakest. SuperPoint + LightGlue match *learned* keypoints, which are
far more robust on these scenes; this is why `03_tech_stack.md` and `06_mvp2.md`
specify SuperPoint + LightGlue as the target stabilizer.

**Why ECC ships first anyway.**

- Zero new dependencies — `cv2` is already in the project; SuperPoint + LightGlue
  add a heavy dep on top of the currently CPU-pinned `torch`.
- It is the same method BoT-SORT already runs for its internal camera-motion
  compensation (`cmc_method=ecc`), so it is known to execute on our footage.
- Measure-before-optimize: ship ECC, quantify how much camera motion actually
  pollutes exported trajectories, and only pay for the heavy estimator if the gap
  warrants it.

**Why it is a clean swap.** The `EgoMotionEstimator` port returns a `Transform2D`
per frame; nothing downstream knows or cares how it was estimated. Replacing ECC
with SuperPoint + LightGlue is a single new adapter wired in the CLI — no domain,
pipeline, exporter, or test changes outside the new adapter and its own tests.

> When the ECC slice lands, record the ECC-first deviation (and this upgrade
> pointer) in `06_mvp2.md`, which currently specifies SuperPoint + LightGlue from
> the start.
