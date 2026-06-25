# Study Guide: High-Precision Aerial Vehicle Trajectory Extraction

> **Purpose:** a guided reading path so *you* can work through the literature and come out
> understanding **what TraTrac delivers, how, and why** — not a citation dump. Each paper
> entry tells you *what to read*, *the one idea*, *what to extract*, and *why it matters to
> your deliverable*.
>
> **Status:** external literature review, not an authoritative design doc. Compiled
> 2026-06-13 via adversarially-verified web search (18 sources, 25 claims verified, 22
> confirmed, 3 refuted). Read alongside the `accel-noise-root-cause` memory and `05_5_mvp1_75.md`.

---

## 0. The mental model you're building toward

TraTrac delivers **metric vehicle trajectories** — position, speed, acceleration over time —
for traffic-safety analytics (SSAM). The whole accuracy problem is one causal chain. Hold
this in your head; every paper below illuminates exactly one link:

```
detector center jitter ─┐
camera ego-motion ──────┼──▶ noisy POSITION(t)  ──d/dt──▶ noisy speed  ──d/dt──▶ EXPLODING accel/jerk
projection (GSD) error ─┘                         (differentiation amplifies high-freq noise)
```

The fixes, in the order they belong in the pipeline:

1. **Remove ego-motion** so position is in a fixed frame (stabilization).
2. **Project to metric** so position is in metres (GSD / homography).
3. **Smooth POSITION, then differentiate** — never smooth speed/accel directly.

If you understand *why differentiation amplifies noise* and *why you smooth position not
acceleration*, you understand 80% of what you're delivering. Start there (§1).

---

## 1. Why this is hard — the one concept that explains everything

### Punzo, Borzacchiello & Ciuffo (2011)
*On the assessment of vehicle trajectory data accuracy and application to the NGSIM
program data.* **Transportation Research Part C** 19(6):1243–1262.
<https://www.sciencedirect.com/science/article/abs/pii/S0968090X10001701>

- **What to read:** the abstract + the section quantifying jerk in raw NGSIM. Skim the
  rest. ~2000 citations — this is *the* reference for the problem.
- **The one idea:** numerical differentiation is a high-pass filter — it amplifies
  high-frequency noise. So a tiny, invisible wobble in position becomes a large error in
  speed and a *huge* error in acceleration/jerk. They show raw NGSIM jerk exceeds 15 m/s³
  in 6.5–17.4% of samples and >74% of one-second windows have physically-impossible jerk
  sign flips.
- **Extract:** the intuition that **acceleration noise is a symptom, position noise is the
  disease.** Internalise the numbers as evidence the problem is real and well-known.
- **Why it matters to your delivery:** this is the *why* behind your acceleration tail
  (your `accel-noise-root-cause` memory). When you explain TraTrac, this is the
  first-principles justification for every smoothing decision you make. If you can only
  read one thing, read this.

---

## 2. The recipe — how the best aerial datasets actually fix it

### Krajewski, Bock, Kloeker & Eckstein (2018)
*The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German
Highways…* **IEEE 21st ITSC.** <https://arxiv.org/pdf/1810.05642> — *code/data: public.*

- **What to read:** Section IV "Track Extraction", especially **§D Track Postprocessing**.
  Two pages. This is the core of your study.
- **The one idea:** detect each frame independently (jitter and all), then refine the whole
  track *offline* with a **Rauch–Tung–Striebel (RTS) smoother on a constant-acceleration
  motion model**. This recovers smooth position, speed *and* acceleration simultaneously,
  and it's **zero-phase** (forward+backward pass) so it adds no lag.
- **Extract:** (1) *why offline smoothing beats a live filter* — your export is batch, so
  you can afford the backward pass an EMA/forward-Kalman can't; (2) *why a constant-accel
  motion model* — it encodes vehicle physics, so it suppresses noise without inventing
  motion; (3) their result: positioning error down to **~1 px (~10 cm)**.
- **Why it matters to your delivery:** this is exactly the smoother `tratrac-postprocess` runs
  over the track record to build the `.trj` (the old in-pipeline EMA was removed). Understand RTS here and you can
  defend a real improvement to your trajectory quality — and explain the trade-off (offline
  vs causal).

> **Background concept to look up while reading this:** what a **Kalman filter** is (predict
> + update), and what a **smoother** adds (it uses *future* measurements too — the backward
> pass — which a real-time filter can't). RTS = Kalman forward pass + RTS backward pass.

---

## 3. Stabilization done right — and why you don't just adopt Geo-trax

### Fonod, Cho, Yeo & Geroliminis (2025) — "Geo-trax"
*Advanced computer vision for extracting georeferenced vehicle trajectories from drone
imagery.* **Transportation Research Part C** 178:105205. (arXiv 2411.02136)
<https://www.sciencedirect.com/science/article/pii/S0968090X25002098> — *code "Stabilo" + data released.*

- **What to read:** the stabilization/registration section (BB exclusion masks; the
  detector×RANSAC grid search) and the kinematics/validation section (Gaussian smoothing,
  RTK-GNSS validation giving 2.21 px / 12.2 cm BB-center error).
- **The one idea (the transferable technique):** when you estimate camera ego-motion from
  feature matches, **mask out the detected vehicle boxes** so keypoints land only on static
  background — moving traffic can no longer hijack the homography. And apply the transform
  to **coordinates, not pixels** (no full-frame warp).
- **Extract:** the masking trick — that's the part worth copying into your
  `OrbEgoMotionEstimator`. Note it fixes exactly your `orb-stabilization-traffic-hijack`
  failure. Note also it stabilizes *after* tracking; you do it *before* (MVP1.9) — same
  principle, different placement.
- **Why it matters to your delivery — and why NOT to wholesale adopt it:** "Geo-trax already
  does what I need" is half true. Take the *technique* (own it, defend it, fits your
  architecture); be wary of adopting the *system* — it emits its own georeferenced format
  (not SSAM, which is your hard invariant), it has none of your onion/port structure, its
  georeferencing assumes survey/RTK inputs you may not have, and a black-box pipeline
  defeats the understanding you're trying to build. Borrowing its "Stabilo" library behind
  your `EgoMotionEstimator` port is a *separate, plausible* option for MVP2 — **gated on its
  license** (this repo already carries AGPL distribution risk from `boxmot`/`ultralytics`;
  don't compound it blind).

### Zheng et al. (2022/2024) — "CitySim"
*CitySim: A Drone-Based Vehicle Trajectory Dataset…* **Transportation Research Record.**
(arXiv 2208.11036) <https://arxiv.org/pdf/2208.11036> — *dataset public.*

- **What to read:** the stabilization subsection only (vehicle-free median reference frame
  + 3×3 homography; CSRT fallback at high altitude).
- **The one idea:** a second, independent team reaches the *same* conclusion — register
  against a **vehicle-free** background reference.
- **Extract:** confirmation that the masking principle is standard practice, not a one-off.
- **Why it matters:** corroboration. **Caveat:** CitySim reports accuracy only as **IOU**
  (0.53→0.76→0.976), *not* metric RMSE — so it is **not** a positional-precision benchmark.
  Don't quote it as one.

---

## 4. Smoothing alternatives — read these only if RTS over-smooths real braking

These are the "escalation path." Skim unless §2's RTS proves too aggressive on hard
accel/decel events.

### "Evaluation of optimal trajectory smoothing techniques… UAV dataset" (2026)
**Innovative Infrastructure Solutions** (Springer) 11:136. <https://link.springer.com/article/10.1007/s41062-026-02522-3>
- **The one idea:** head-to-head of nine smoothers; **Savitzky–Golay (window 7, poly 2)**
  best for *position* — but no method wins on all metrics; it's dataset-dependent.
- **Extract:** the position-first principle, again, plus SG as a cheap baseline to compare
  your RTS against. **Why it matters:** gives you a defensible alternative and the honest
  caveat that "best smoother" depends on your data.

### Chen et al. (2023)
*An Acceleration Denoising Method Based on an Adaptive Kalman Filter…* **J. Advanced
Transportation**, DOI 10.1155/2023/2661136. <https://onlinelibrary.wiley.com/doi/10.1155/2023/2661136>
- **The one idea:** switch between a linear KF (steady car-following) and an Unscented KF
  (merging/hard manoeuvres) so a single fixed model doesn't smear real dynamics. Cut jerk
  range from ±~4900 to ±~45.
- **Extract:** the notion that *one* motion model can be too rigid — the answer to "what if
  RTS flattens a real hard brake?" **Why it matters:** your fallback if constant-accel
  RTS loses fidelity on aggressive driving.

### "Consistent vehicle trajectory extraction… oriented object detection" (2025)
**Scientific Reports** s41598-025-12301-2. <https://www.nature.com/articles/s41598-025-12301-2> — *open access.*
- **The one idea:** EKF + sliding-window optimisation enforcing physical limits (max accel,
  min gaps) cut kinematic *consistency* error from 3.54/5.58 m to 0.06/0.09 m.
- **Extract:** constrained optimisation as a heavier alternative. **Caveat:** that's a
  *consistency* metric, **not** positional RMSE — and this is the paper whose **OBB jitter
  claims were refuted** (see §6). Read critically.

### "Extracting High-Precision Vehicle Motion Data… Various Weather Conditions" (2022)
**Remote Sensing** 14(21):5513. <https://www.mdpi.com/2072-4292/14/21/5513> — *medium confidence (403, abstract-corroborated).*
- **The one idea:** a full worked aerial pipeline (CLAHE + YOLOv5-OBB, SIFT+KNN
  stabilization, "SORT++"). **Why it matters:** a system-design template; reports no metric
  RMSE, so use it for architecture ideas, not benchmarks.

---

## 5. Benchmark targets — what "good" looks like (scene-specific, not universal)

| System | Reported accuracy | Read it as |
| --- | --- | --- |
| **highD** | <3 cm midpoint error vs manual; ~10 cm post-RTS | best-case flat highway nadir, known GSD |
| **Geo-trax** | 2.21 ± 1.99 px = 12.2 ± 10.9 cm BB-center | RTK-validated, realistic target |
| **pNEUMA / pNEUMA Vision** | ~16.5 cm/px GSD; image-space x,y+azimuth | annotations are image-space, not metric |
| **CitySim** | IOU only (0.53→0.76→0.976) | **not** a metric target |

> pNEUMA Vision (Kim et al., 2023, *TR-C* 147,
> <https://www.sciencedirect.com/science/article/pii/S0968090X22003795>) also proposes a
> regularized **anomaly-detection** denoiser for noisy vision trajectories — relevant prior
> art for your smoothing stage.

**Why it matters:** these tell you what error to *expect* and to *aim for*. They are **not
comparable across datasets** (different altitude/GSD/scene/ground-truth) — treat ~3 cm
(highD) / ~12 cm (Geo-trax) as scene-specific goals, not a universal SOTA you must beat.

---

## 6. Refuted claims — train your skepticism, do NOT cite these

The verification round killed three claims that *look* authoritative. Knowing what failed
is part of understanding the field:

- **OBB reduces jitter vs axis-aligned boxes by 15%/20%** — *refuted 1–2.*
- **OBB yields 0.068 m vs 0.611 m frame-to-frame noise** — *refuted 0–3.*
- **Geo-trax's 2.21 px is "sub-pixel"** — *refuted* (it's a few-pixel error; the number is
  fine, the adjective isn't).

**Takeaway:** oriented bounding boxes as a *jitter cure* is an **unconfirmed hypothesis**.
Don't switch to OBB on the strength of these papers. Good practice: notice when a number is
quoted without a comparable baseline.

---

## 7. Suggested reading order (a weekend's study)

1. **Punzo 2011** §abstract + jerk section — *why the problem exists.* (the keystone)
2. **highD 2018** §IV-D — *the fix you'll most likely implement.*
3. **Geo-trax 2025** stabilization + validation — *the ego-motion fix + a realistic accuracy target.*
4. **CitySim 2022** stabilization subsection — *corroboration; learn what IOU-only reporting hides.*
5. **Springer 2026** — *the smoother comparison + the "it depends" honesty.*
6. (only if needed) **Chen 2023 / Scientific Reports 2025** — *escalation when one motion model is too rigid.*

After these, you can explain to anyone: why acceleration is noisy, why you smooth position
not acceleration, why you mask vehicles during stabilization, and what accuracy is achievable.

---

## 8. What this implies for TraTrac (proposals — your call, not decided)

- **Smoothing:** **shipped** — `tratrac-postprocess` runs an RTS / constant-acceleration
  position-domain smoother over the track record before deriving kinematics (the in-pipeline
  EMA was removed in the export inversion). Offline zero-phase fits the batch export.
- **Stabilization:** **mask detections out of ORB keypoint extraction** in
  `OrbEgoMotionEstimator` to fix `orb-stabilization-traffic-hijack` without disabling
  stabilization on static clips. (This is the Geo-trax *technique*, owned by you.)
- **Geo-trax as a dependency:** only as the "Stabilo" library behind the `EgoMotionEstimator`
  port, and only after a **license check** (AGPL/distribution risk). Not as a whole-system
  replacement — it emits non-SSAM output and bypasses your architecture.

---

## 9. Open questions the literature did not answer

- Does OBB detection *actually* reduce center jitter vs axis-aligned boxes? (strong claims
  refuted — needs independent confirmation before any investment.)
- Smoothing-strength vs dynamic-fidelity (lag / over-smoothing real braking) trade-off for
  RTS-CA vs Savitzky-Golay vs adaptive KF on the *same* aerial dataset? No head-to-head exists.
- How does residual ~3–12 cm error split between detector / stabilization / GSD? No
  published error budget.
- Do SuperPoint+LightGlue (planned MVP2) beat SIFT/ORB+RANSAC *with exclusion masks*, and is
  the gain worth it once masking already kills the traffic-hijack failure? Not benchmarked directly.
