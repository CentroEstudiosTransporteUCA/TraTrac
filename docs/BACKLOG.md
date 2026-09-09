# Final-Product Polish — Deferred Robustness Upgrades

---

## What This Is

A backlog of **deliberate "good enough for now, upgrade later" decisions**: places
where we shipped a cheaper implementation behind a stable port and recorded the
intended final-product replacement here. Each entry names the seam, the current
adapter, the target adapter, and *why* the upgrade is worth it.

This is not the MVP roadmap (see `docs/ROADMAP.md` and the per-MVP docs it indexes) — those
add *new capabilities*. This file tracks *quality upgrades to existing capabilities* that
were intentionally deferred. An item leaves this file when it ships.

The recurring pattern: a port isolates the seam, so each upgrade is a one-adapter
swap with no change to the domain, the pipeline, or the other adapters.

---

## Items

### 1. Ego-motion estimation: ORB → SuperPoint + LightGlue

| | |
| --- | --- |
| Port | `EgoMotionEstimator` (stabilization seam, **shipped in MVP1.9** — see `src/tratrac/infrastructure/video/EGO_MOTION.md`) |
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
local ratio test would accept. This is why `docs/TECH_STACK.md` and `src/tratrac/application/WORLD_PROJECTION.md`
name SuperPoint + LightGlue as the target stabilizer.

**Why it is a clean swap.** The `EgoMotionEstimator` port returns a `Transform2D`
per frame; nothing downstream knows or cares how it was estimated. Replacing ORB
with SuperPoint + LightGlue is a single new adapter wired in the CLI — no domain,
pipeline, exporter, or test changes outside the new adapter and its own tests.

> The MVP1.9 ORB slice has landed; `src/tratrac/application/WORLD_PROJECTION.md` still specifies SuperPoint +
> LightGlue from the start — its stabilization box is now an *upgrade* of MVP1.9's
> ORB adapter, not a from-scratch addition.

---

### 2. Kinematics: Kalman/RTS smoothing (now the only path)

| | |
| --- | --- |
| **Seam** | the offline `tratrac-postprocess` pass over the track record |
| **Target** | constant-acceleration **Kalman/RTS** smoother (`application/kalman.py`) |
| **Trigger** | acceleration/jerk noise in the `.trj` (the standing accel-noise issue) |

The offline two-pass has **shipped** (`src/tratrac/application/SMOOTHING.md`): the perception run writes the raw
track record, and `tratrac-postprocess` runs a forward+RTS smoother to reconstruct kinematics and
write the `.trj`. The **export inversion** removed the in-pipeline EMA/forward-Kalman
orientation entirely — there is no longer a streaming/inline `.trj`, so kinematics is always
the offline smoother. What remains deferred: **tuning** `pos_noise`/`jerk` against real
footage using the `validate_trj.py` jerk metric; and, if a real-time `.trj` is ever needed,
a streaming path built on the kept `KinematicKalmanFilter` (or a **fixed-lag** middle ground).

---

### 3. Video I/O: cv2 decode → PyAV (+ NVDEC)

| | |
| --- | --- |
| Port | `VideoSource` (`infrastructure/video/opencv.py`, `OpenCvVideoSource`) |
| Ships with | `cv2.VideoCapture` — decode, `--start`/`--end` seeking (`src/tratrac/infrastructure/video/TIME_WINDOW.md`), and the `cv2.grab()`-without-decode skip used by `--process-fps` (`src/tratrac/infrastructure/TIMESTEP_PRECISION.md`) |
| Target | PyAV decode, with NVIDIA NVDEC hardware acceleration — `docs/TECH_STACK.md`'s "Video Decoding: NVIDIA NVDEC + PyAV" row |
| Trigger to upgrade | decode throughput becomes the pipeline bottleneck, or the `mp4v`-style codec gaps below start affecting decode too |

**How this gap was found.** `overlay_video.py`'s writer was found producing overlay
videos 3-5x the size of their source (see `src/tratrac/infrastructure/export/VIDEO_EXPORT.md`) because this
project's `opencv-python` build has no software H.264 encoder, so `cv2.VideoWriter`
silently fell back to the far less efficient `mp4v` (MPEG-4 Part 2). Fixing it moved
the **encode** side of `OverlayVideoExporter` to PyAV, which is what `docs/TECH_STACK.md`
already named as the target for video I/O — but unlike the RT-DETR→YOLOv8 detector
override (`src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`), nothing had recorded *that* PyAV was still unadopted, so the
gap had sat untracked since MVP1. This entry closes that: the encode half shipped,
decode is deferred, and it should stay tracked next time.

**Why decode wasn't swapped in the same change.** `OpenCvVideoSource` backs the
**live pipeline's** per-frame decode (every `tratrac` run), not a standalone
post-hoc tool, so it carries real behavioral risk the writer didn't: `--start`/
`--end` window seeking and, more importantly, `--process-fps` decimation depends on
`cv2.grab()` to skip a frame's *full decode* cost, not just discard it after
decoding. PyAV has no direct one-line equivalent — the same effect needs discarding
undecoded packets before the decoder, which is a real reimplementation to verify
against the existing `FrameWindow`/`DecimationGrid` tests, not a drop-in adapter
swap like the writer was.

**Why NVDEC is bundled into the same upgrade, not split further.** `docs/TECH_STACK.md`
pairs PyAV with NVDEC specifically for the decode *performance* win; adopting PyAV
for decode without NVDEC gets consistency with the target stack but not the payoff
the stack entry is actually about, so there's no reason to make it a two-step migration.

**Why it is a clean swap (once done).** `VideoSource` is a Protocol port; replacing
`OpenCvVideoSource` with a PyAV(+NVDEC) adapter is a single new adapter behind the
existing seam, same as the other entries in this file — no domain or pipeline change,
only `FrameWindow`/`DecimationGrid`'s interaction with the new adapter needs re-verifying.
