# Tracker Choice: BoT-SORT (not SORT), DINOv3 for ReID (not FastReID)

`docs/TECH_STACK.md`'s ideal stack names BoT-SORT, not plain SORT, deliberately — and this
choice was re-checked against current research (not just carried over from the original stack
pick), unlike the detector choice which didn't survive that check. Tracking held up; the ReID
plan for MVP5 didn't.

## Why BoT-SORT — confirmed against current benchmarks

`boxmot` (already TraTrac's dependency) supports six trackers: BoTSORT, HybridSORT, StrongSORT,
DeepOCSORT, ByteTrack, OCSORT. **BoT-SORT ranks highest among all of them on the MOT17
benchmark** (68.9, ahead of HybridSORT 68.2, StrongSORT 68.1, DeepOCSORT 67.8). It's also the
only one of the six with an appearance branch — which is what identity persistence through
occlusion (MVP5) actually needs; ByteTrack and OCSORT are motion-only and can't do that job
regardless of association quality.

One nuance worth being precise about: BoT-SORT's built-in camera-motion-compensation is a large
part of why it's generally recommended for drone footage, but TraTrac already handles ego-motion
itself via the keyframe-anchored ORB stabilizer (MVP1.9) and disables BoT-SORT's own CMC
(`cmc_method=None`) when that's active — so the value BoT-SORT adds to *this* pipeline
specifically is the appearance branch, not CMC.

## Why NOT plain SORT

SORT is motion-only and has poor long-term stability — exactly the failure mode this project
needs to avoid once occlusion recovery matters (MVP5).

## Current state (MVP1)

Shipped as **IoU-only** (`infrastructure/tracking/boxmot_bot_sort.py`, `boxmot.trackers.BotSort`) —
the appearance/ReID half of BoT-SORT isn't wired in yet, so today it behaves closer to SORT in
practice. ReID activation is MVP5's job; the adapter and the `Tracker` port don't need to change
for that, only the model/config BoT-SORT is constructed with.

When ego-motion coordinate stabilization is enabled (MVP1.9, see
`src/tratrac/infrastructure/video/EGO_MOTION.md`), BoT-SORT's own camera-motion-compensation is
disabled (`cmc_method=None`) so it doesn't double-correct already-stabilized detections.

**OBB tracking is natively supported**, confirmed directly against the `boxmot` project, not
assumed: it handles both axis-aligned and oriented-box detections from any model. This matters
because MVP1.5's replanned detector (`DETECTOR_CHOICE.md`) outputs oriented boxes — the tracker
doesn't need the orientation angle collapsed away before association, so it can survive into the
track for the exporter to use.

`boxmot` is **AGPL-3.0** — relevant if TraTrac is distributed (see `CLAUDE.md` Dependency Notes).

## ReID for MVP5: DINOv3 embeddings + motion-plausibility gating, not FastReID

`docs/TECH_STACK.md`'s original stack named **FastReID**. That doesn't survive scrutiny for this
specific use case, for a reason that's more fundamental than "wrong model": **the nadir viewpoint
itself discards most of what vehicle-ReID models are trained to discriminate on** — license
plates, side profile, grille/tail-light shape are largely invisible from directly overhead. This
isn't a TraTrac-specific problem; the literature is explicit about it, and a real ten-UAV,
twenty-intersection drone-traffic deployment (Songdo, South Korea) found pure vision-only ReID
insufficient from the air, resorting to fusing it with a temporal travel-time model built on
traffic-flow shockwave theory.

There's also no nadir-matched fine-tuning dataset the way UAV-OBB solved that problem for
detection: the best available aerial vehicle-ReID dataset, **VRAI** (137K images), spans
15–80m altitude with hovering/cruising/rotating viewpoint changes — mixed-angle, not
nadir-specific. FastReID's checkpoints (and `boxmot`'s own `clip_vehicleid.pt`) are trained on
ground-level/oblique surveillance datasets (VehicleID, VeRi-style) — vehicle-domain, which beats
a generic person-ReID checkpoint, but carrying the same oblique-vs-nadir gap.

**What actually generalizes better across this gap: DINOv3.** Self-supervised foundation-model
embeddings — not trained on any single fixed viewpoint — transfer across viewpoint shift
measurably better than a supervised ReID checkpoint. A DINOv3-pretrained backbone reaches 88.19
mAP on VeRi-Wild vehicle ReID from visual cues alone (matching the strongest metadata-dependent
baselines), and DINO-family ViTs are the most geometry-aware general-purpose vision transformers
available, retaining mIoU 0.766 under **90° angular separation** — effectively the
oblique-to-nadir gap this project faces. It's still measurably weaker under large viewpoint
shift than small ones — not immune to the problem — but it's the strongest available answer.

**Recommended: DINOv3 embeddings as BoT-SORT's appearance signal, combined with
motion-plausibility gating** — using the Kalman state `application/kalman.py` already maintains
to reject re-identification candidates that aren't a physically plausible reappearance (right
place, right time, right velocity for the elapsed gap). This mirrors the Songdo study's
appearance+temporal fusion in principle, using the motion infrastructure TraTrac already has
instead of a separate travel-time model. Cheap color/footprint-shape heuristics remain a
legitimate fallback given how little nadir-view visual signal survives at all — see
`docs/roadmap/mvp5.md` for the full MVP5 plan.

**Sources:**
- [BoxMOT — tracker list, MOT17 rankings, AABB + OBB support](https://github.com/mikel-brostrom/boxmot)
- [Tracker comparison — MOT benchmark results](https://trackers.roboflow.com/latest/trackers/comparison/)
- [DINOv3-based vehicle ReID — 88.19 mAP on VeRi-Wild](https://arxiv.org/html/2607.22068)
- [DINOv3 cross-viewpoint robustness under angular separation](https://www.alphaxiv.org/abs/2508.10104)
- [VRAI dataset — altitude/viewpoint composition](https://arxiv.org/pdf/1904.01400)
- [Drone traffic monitoring ReID + temporal/shockwave-theory fusion (Songdo study)](https://www.eurekalert.org/news-releases/1124023)
- [Oblique-vs-nadir vehicle-classification visibility tradeoff](https://doi.org/10.3390/rs17152653)
