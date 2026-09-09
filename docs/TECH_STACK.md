# Ideal Final Tech Stack

| Layer | Technology | Changed? |
| --- | --- | --- |
| Runtime | PyTorch | — |
| Video Decoding | TorchCodec (NVDEC) | ⚠️ was PyAV |
| Video Encoding | PyAV (libx264) | — (shipped) |
| Stabilization | SuperPoint + LightGlue | — (confirmed) |
| Detection | YOLO (v11/v26), **OBB task** | ⚠️ was RT-DETR |
| Segmentation | SAM 3 | ⚠️ was SAM2 |
| Tracking | BoT-SORT | — (confirmed) |
| ReID | DINOv3 embeddings + motion-plausibility gating | ⚠️ was FastReID |
| Motion Modeling | Constant-acceleration Kalman/RTS (shipped); KalmanNet-family adaptive noise (frontier) | ⚠️ was "Extended Kalman Filter" |
| Geometry | Multi-Homography (OpenCV); auto-calibration from road geometry where possible | 🟡 refined |
| Plane Assignment | Polygon-Based Plane Mapping | — |
| Topology Constraints | Lane Graph Model | — |
| Storage | Apache Parquet | — (shipped) |
| Visualization | FiftyOne | — (not re-researched) |
| Annotation | CVAT | — (not re-researched) |
| Deployment | Docker + CUDA | — (not re-researched) |
| Hardware | RTX 4090 / A100 | — |

The rows marked ⚠️ replace picks from an earlier version of this document that turned out to be
wrong once checked against current research — not preference, evidence. Each section below
shows the comparison and the sources, the same way `application/RESEARCH_NOTES.md` documents
the smoothing/stabilization literature. Rows marked — were checked and confirmed correct;
rows marked "not re-researched" simply haven't been audited yet.

---

## Detection: YOLO-OBB, not RT-DETR

**Why the original RT-DETR pick doesn't hold up for this project's actual use case.** RT-DETR's
real advantage over YOLO is transformer-based global reasoning in **dense, oblique, cluttered**
aerial scenes — many overlapping objects, varied camera angles. TraTrac's actual footage is
**consistently nadir/cenital** street video. That's a narrower, easier problem than the one
RT-DETR's advantage targets: top-down vehicles have far less perspective-driven occlusion
between them than oblique views do. Worse, the dataset any RT-DETR fine-tune would train on
(VisDrone) is itself a **mixed-oblique-angle** dataset — switching architecture while training
on the same non-nadir data doesn't actually target the nadir constraint at all.

**The bigger problem: RT-DETR can't do the thing that would actually help most.** TraTrac's
persistent weak point isn't raw detection — it's **orientation estimation** (`OrientationEstimator`'s
EMA-smoothed-heading hack, with a bbox-major-axis fallback that the validator specifically
catches flipping). A car viewed from directly overhead can face any heading; an axis-aligned box
around it is orientation-blind. **Oriented bounding box (OBB) detection reports the vehicle's
actual angle at detection time** — it solves orientation at the source instead of patching it
downstream with a smoothed heuristic. The standard Hugging Face RT-DETR implementation **does
not support OBB** — confirmed directly, not assumed: no rotation-angle output, axis-aligned only.
So RT-DETR is architecturally the wrong tool for the upgrade this pipeline actually needs, on top
of not fitting the nadir constraint it was originally chosen for.

**What to use instead: `ultralytics` YOLO's native OBB task** (`yolo26n/s/m-obb`, or `yolo11-obb`
as a more battle-tested fallback). This reuses the dependency TraTrac already has (`ultralytics`,
already an accepted AGPL trade-off) rather than building a from-scratch HuggingFace `transformers`
fine-tuning harness that doesn't exist in this repo today — a materially cheaper integration path,
independent of the accuracy argument.

**Fine-tuning dataset: UAV-OBB, not VisDrone.** UAV-OBB is captured specifically at 75–108m
altitude over urban roads — the same drone-survey altitude regime TraTrac's GSD calibration
already targets — with oriented annotations across six vehicle classes, already in YOLO-OBB
label format (zero conversion effort). It's small (1,617 images), so fine-tuning is realistic in
days, not weeks. **DroneVehicle** (28,439 RGB/IR image pairs, 5 vehicle classes, diverse
lighting/weather, also oriented) is the larger, more robust option to layer in if UAV-OBB alone
underperforms. A public project has already fine-tuned YOLOv11-OBB on a comparable UAV vehicle
dataset, so this combination isn't unprecedented.

**Honest trade-off this reopens, not resolves: licensing.** The original RT-DETR pick was partly
motivated by clearing the AGPL-3.0 `ultralytics` dependency (see `CLAUDE.md` Dependency Notes).
Staying on `ultralytics` for OBB does **not** clear that — it doubles down on it. This needs an
explicit decision, not a silent carry-over, if TraTrac is ever distributed.

**Downstream integration this requires, not a drop-in swap:** `VehicleState`/the orientation
pipeline needs to accept a detected angle (from the OBB) instead of, or blended with, the current
EMA estimate — this is where the actual payoff is, so it should be done properly. See
`src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` for the full MVP1.5 replan.

**Sources:**
- [UAV-OBB dataset (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2352340926002635)
- [UAV-OBB dataset (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092195/) — altitude/viewpoint/class detail
- [UAV-OBB on Mendeley Data](https://data.mendeley.com/datasets/6snrjwcpkh/1)
- [DroneVehicle dataset overview (HyperAI)](https://hyper.ai/en/datasets/32488)
- [Ultralytics OBB task docs](https://docs.ultralytics.com/tasks/obb)
- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- [RT-DETR docs (Hugging Face)](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr) — confirms no OBB support
- [DOTA dataset paper](https://arxiv.org/pdf/1711.10398) — for context on why DOTA-pretrained OBB weights are a domain-gapped *starting point*, not a substitute for drone-altitude fine-tuning (DOTA is satellite/very-high-altitude imagery)
- [UAV-DETR / RTUAV-YOLO / CF-YOLO — specialized aerial small-object detectors](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349633/) — beat both plain YOLO and plain RT-DETR on drone imagery, but as research models without the easy pretrained-checkpoint path YOLO-OBB has; a later escalation option, not the near-term pick

---

## Tracking algorithm: BoT-SORT — confirmed, no change

`boxmot` (already TraTrac's tracking dependency) supports six trackers: BoTSORT, HybridSORT,
StrongSORT, DeepOCSORT, ByteTrack, OCSORT. **BoT-SORT ranks highest among all of them on MOT17**
(68.9 vs. 68.2/68.1/67.8 for the next three). It's also the only one of these with an appearance
branch — which is what identity persistence through occlusion (MVP5) actually needs; ByteTrack
and OCSORT are motion-only and can't do that job regardless of how well they associate.

One nuance: BoT-SORT's camera-motion-compensation is a large part of why it's recommended for
drone footage generally, but TraTrac already handles that itself via the keyframe-anchored ORB
stabilizer (MVP1.9) and disables BoT-SORT's own CMC when that's active. So the value BoT-SORT
adds *to this pipeline specifically* isn't CMC — it's the appearance branch, i.e. the ReID model
below. **`boxmot` also natively supports OBB tracking, not just axis-aligned boxes** — confirmed,
not assumed — so the detection pivot above doesn't require collapsing OBB detections to an
axis-aligned box before tracking; the orientation angle can survive into the track.

See `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md` for the full writeup.

**Sources:**
- [BoxMOT — tracker list + MOT17 rankings, AABB + OBB support](https://github.com/mikel-brostrom/boxmot)
- [Tracker comparison — MOT benchmark results](https://trackers.roboflow.com/latest/trackers/comparison/)
- [BoT-SORT vs OC-SORT vs ByteTrack for drone/handheld footage](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort)

---

## ReID: DINOv3 embeddings + motion-plausibility gating, not FastReID

**The nadir-view appearance problem is real and documented, not hypothetical.** Vehicle-ReID
models are trained to discriminate on cues that mostly don't survive the drop to straight-down —
license plates, side profile, grille/tail-light shape. The literature is blunt about this:
"oblique perspective is technically advantageous for vehicle classification — it improves
visibility of body shape, height profiles, and vehicle-type cues that are difficult to
distinguish from standard nadir imagery." A real multi-week drone-traffic deployment (ten UAVs,
twenty intersections, Songdo, South Korea) hit this hard enough that pure vision-only ReID wasn't
sufficient — they had to fuse it with a temporal travel-time model grounded in traffic-flow
shockwave theory to make re-identification work at all. There's also no clean nadir-specific
vehicle-ReID *dataset* to fine-tune on the way UAV-OBB solved that problem for detection — the
best available aerial vehicle-ReID dataset (**VRAI**, 137K images) spans 15–80m altitude with
hovering/cruising/rotating viewpoint changes, mixed-angle like VisDrone was.

**FastReID's checkpoints, and `boxmot`'s own vehicle-ReID checkpoint (`clip_vehicleid.pt`), are
trained on ground-level/oblique surveillance footage (VehicleID, VeRi-style datasets)** — vehicle-domain,
which is better than a generic person-ReID checkpoint, but they carry the same oblique-vs-nadir
viewpoint gap as VisDrone-trained detectors do. Fine-tuning FastReID doesn't fix this — there's
no large nadir-vehicle-ReID dataset to fine-tune it *on*.

**What actually generalizes better across this viewpoint gap: DINOv3.** This is the one genuine
correction from earlier research passes in this investigation — self-supervised foundation-model
embeddings, not trained on any single fixed viewpoint, transfer across viewpoint shift much
better than a supervised ReID checkpoint does. Concretely: a DINOv3-pretrained backbone reaches
88.19 mAP on VeRi-Wild vehicle ReID from visual cues alone — matching the strongest
metadata-dependent baselines — and DINO-family self-supervised ViTs are **the most
geometry-aware general-purpose vision transformers available**, retaining mIoU of 0.766 (DINOv3)
under **90° angular separation** — effectively the oblique-to-nadir gap this project actually
faces. DINOv3 is still measurably weaker under large viewpoint shift than under small ones (it's
not immune to the problem), but it's the strongest available answer, not a shrug.

**Recommended combination, not either/or:** DINOv3 embeddings as the appearance signal BoT-SORT's
ReID branch consumes, **combined with motion-plausibility gating** — using the Kalman state
TraTrac's smoother already maintains to reject re-identification candidates that aren't a
physically plausible reappearance (right place, right time, right velocity), the same
belt-and-suspenders approach the Songdo study's temporal fusion validates in principle even
though the mechanism there was travel-time estimation rather than a Kalman state. Pure color/
footprint-shape heuristics remain a legitimate cheap fallback given how little nadir-view visual
signal survives at all.

**Sources:**
- [DINOv3-based vehicle ReID — 88.19 mAP on VeRi-Wild, foundation-model era rethink](https://arxiv.org/html/2607.22068)
- [DINOv3 cross-viewpoint robustness — mIoU under angular separation](https://www.alphaxiv.org/abs/2508.10104)
- [Vehicle Re-ID in Aerial Imagery (VRAI dataset paper, ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Vehicle_Re-Identification_in_Aerial_Imagery_Dataset_and_Approach_ICCV_2019_paper.pdf)
- [VRAI altitude/viewpoint composition (15–80m, mixed)](https://arxiv.org/pdf/1904.01400)
- [Drone traffic monitoring ReID + temporal/shockwave-theory fusion (Songdo study)](https://www.eurekalert.org/news-releases/1124023)
- [Oblique-vs-nadir vehicle-classification visibility tradeoff](https://doi.org/10.3390/rs17152653)
- [Trends in Vehicle Re-identification — comprehensive review](https://arxiv.org/pdf/2102.09744)

---

## Motion modeling: constant-acceleration Kalman/RTS (shipped, correct foundation); KalmanNet-family adaptive noise is the frontier upgrade

The original "Extended Kalman Filter" stack entry undersold what's already shipped and pointed
vaguely at what should come next. TraTrac's `application/kalman.py` (hand-rolled numpy,
constant-acceleration state, white-noise-jerk process model, RTS backward pass) is the correct
foundation — this isn't a pivot, it's already right, per `application/RESEARCH_NOTES.md` and the
highD/Punzo literature it cites.

What's genuinely open is that `--pos-noise`/`--jerk` are **fixed, hand-tuned hyperparameters**.
2025 research consensus has moved toward **hybrid classical+learned** filters that keep the
Kalman structure (interpretable, physically grounded — worth preserving) but learn the process/
measurement noise covariances from data instead: **KalmanNet** is the foundational named
approach, with **MAML-KalmanNet** (meta-learning, relevant here given TraTrac won't have a huge
labeled-trajectory dataset to train on) and **Recursive KalmanNet** (couples neural and analytic
covariance propagation to keep error estimates positive-definite and unbiased) as more targeted
descendants. This is a genuine frontier direction, not an off-the-shelf checkpoint swap the way
the detector pivot was — it needs its own design pass. See `src/tratrac/application/SMOOTHING.md`.

**Sources:**
- [KalmanNet — neural-aided Kalman filtering, foundational method](https://www.emergentmind.com/topics/neural-network-aided-kalman-filtering)
- [MAML-KalmanNet — meta-learning for low-data regimes (IEEE TSP 2025)](https://dl.acm.org/doi/abs/10.1109/TSP.2025.3540018)
- [Recursive KalmanNet — positive-definite/unbiased covariance propagation](https://onlinelibrary.wiley.com/doi/10.1002/acs.3982)
- [Latent-KalmanNet — learned Kalman filtering from high-dimensional signals](https://arxiv.org/pdf/2304.07827)
- [Differentiable Adaptive Kalman Filtering via Optimal Transport](https://arxiv.org/pdf/2508.07037)

---

## Geometry: multi-homography stays the right math; auto-calibration from road geometry is the workflow upgrade

No qualitatively better replacement exists for piecewise-planar road-surface projection than
homography-based inverse perspective mapping — this is confirmed, not assumed: it remains the
standard, interpretable, computationally cheap approach for this exact problem, and the
limitations TraTrac's own docs already flag (far-field sensitivity to calibration error,
single-plane assumption breaking for non-planar scenes — the reason MVP3 exists) are
independently corroborated in the current literature, not TraTrac-specific concerns.

What *has* moved is the calibration **workflow**. A May 2026 pipeline demonstrates deriving the
road-plane homography **automatically from visible road geometry** — lane markings, road
borders, crosswalks — instead of an operator manually clicking image↔world correspondence
points, which is what TraTrac's current Approach A requires and what URBAn's checklist already
flags as a UX gap (`URBAn/docs/PRODUCTION_MVP.md`). The same source's caveats match what
`application/WORLD_PROJECTION.md` already documents independently: far-field vehicles are most
sensitive to homography error, and manual validation currently outperforms fully automatic
calibration — so this is a real upgrade path for reducing operator burden, not a replacement for
operator validation. See `application/WORLD_PROJECTION.md`.

**Sources:**
- [Mobile Traffic Camera Calibration from Road Geometry for UAV-Based Traffic Surveillance (2026)](https://arxiv.org/abs/2605.11900)
- [Homography-based ground plane detection, classical approach](https://digital-library.theiet.org/doi/abs/10.1049/iet-its.2009.0073)

---

## Segmentation: SAM 3, not SAM2 — and reconsider how much is still needed

SAM2 is stale as a pick on two independent axes. First, it's been directly superseded: Meta
shipped **SAM 3** (November 2025), which adds text-prompted segmentation ("segment all cars")
and is the current model in the SAM family. Second, even setting that aside, SAM2-class models
are "significantly larger and slower" than current YOLO-seg variants for pure instance
segmentation — SAM's value proposition was always zero-shot generality, not efficiency.

More importantly: MVP4's stated purpose for segmentation was replacing coarse boxes with
"precise segmentation-derived geometry" — substantially an **orientation** problem. Once
detection itself reports orientation directly (the OBB pivot above), the orientation half of
that problem is already solved at the detector, before segmentation ever runs. What segmentation
still adds on top is a precise **occupancy mask/footprint** — useful for exact contact-point or
road-plane assignment — not orientation. Worth explicitly re-scoping MVP4 around "do we need
pixel-accurate footprints" rather than carrying forward the original "segmentation gives us
orientation" framing, which OBB now makes partially redundant. See `docs/roadmap/mvp4.md`.

**Sources:**
- [Best Computer Vision Models in 2026 — task-by-task guide, SAM2 vs YOLO-seg vs SAM3](https://blog.roboflow.com/best-computer-vision-models/)
- [Meta Segment Anything Model 2 (Ultralytics docs)](https://docs.ultralytics.com/models/sam-2)
- [Meta SAM 3 announcement](https://ai.meta.com/sam2/)

---

## Video decoding: TorchCodec, not PyAV

PyAV correctly fixed the **encode** side already (see `docs/BACKLOG.md` item 3) — that stays.
For **decode**, the qualitatively better current choice is **TorchCodec**: PyTorch's own
official media library, under active development specifically to replace ad-hoc PyAV/torchaudio
usage in PyTorch pipelines (PyTorch is consolidating decode/encode capability into it and
deprecating the older torchaudio video path). It supports NVDEC-accelerated GPU decode directly
and is reported as consistently the best-performing library for its designed use case — decoding
many videos as part of a data pipeline, which is structurally close to what `tratrac`'s
perception run needs. Given TraTrac's runtime is already PyTorch, this is a better final target
than PyAV for decode specifically, independent of the real integration risk already documented
in `docs/BACKLOG.md` item 3 (the `--process-fps` no-decode-skip behavior `cv2.grab()` currently
provides has no direct one-line equivalent in either PyAV or TorchCodec — that risk is unchanged
by this pick, not resolved by it).

**Sources:**
- [TorchCodec — PyTorch official media decoding/encoding library](https://github.com/meta-pytorch/torchcodec)
- [TorchCodec announcement — migration path from PyAV/torchaudio](https://pytorch.org/blog/torchcodec/)
- [Accelerated video decoding with NVDEC via TorchCodec/torchaudio](https://docs.pytorch.org/audio/2.8/tutorials/nvdec_tutorial.html)

---

## Not re-researched this round

Plane assignment, topology/lane-graph, FiftyOne, CVAT, and Docker/CUDA deployment weren't
audited in this investigation — the table above carries them forward unchanged from the
original stack, not because they were confirmed, but because they weren't in scope.
