# Production MVP Checklist

Everything TraTrac needs to be a finished production tool, ordered by value delivered to the
civil engineers who'll actually use it — not by what fits in any one deadline. The line below
marks where we realistically expect to be for TraTrac's civil engineering congress presentation
(~1 week out from 2026-09-09); everything above it is shipped or targeted for that date,
everything below stays open afterward as real, valued roadmap — not dropped.

Sibling checklists: `URBAn/docs/PRODUCTION_MVP.md`, `FloCo/docs/PRODUCTION_MVP.md`.

See [`ROADMAP.md`](ROADMAP.md) for the capability-ladder-vs-execution-order status table this
checklist draws its "shipped" checkmarks from, and [`BACKLOG.md`](BACKLOG.md) for the
"shipped cheaper now, upgrade later" items referenced below.

---

- [x] Detect vehicles + track identities frame-to-frame, export syntactically valid SSAM `.trj` (MVP1)
- [x] Physically real metric sizes & speeds from drone metadata — GSD calibration (MVP1.75)
- [x] Drone ego-motion removed from trajectories, so camera movement isn't mistaken for vehicle movement (MVP1.9)
- [x] Metric world-space coordinates via post-hoc homography projection for calibrated scenes (MVP2 Approach A)
- [ ] **Validated against real congress-site footage** — calibration, exclusion zones, smoother tuned until results are demonstrably clean (`validate_trj` compliance). First real-footage validation pass run (`docs/IMPLEMENTATION_PLAN.md` Group B1, ~15min 1920x1080@30fps intersection clip, YOLOv8-VisDrone baseline `conf=0.25`): 38.67%/40.10% Continuity compliance, 99.28% Orientation, 100%/98.81% Speed/Accel — **not yet clean**. 47.1% of the run's 1,545 track fragments last under 1 second, the signature of a low-confidence emergency-detector baseline producing many short/spurious tracks — real evidence the low Continuity score is dominantly a detector-quality problem (see `src/tratrac/application/REID_MERGE.md`'s real-footage findings, which ruled out occlusion/identity as the main cause), pointing at MVP1.5 (Group A2) as the higher-leverage next step once a GPU is available, not exclusion-zone/smoother tuning alone.
- [ ] **Aerial-robust detector** — fine-tuned YOLO-OBB replacing the YOLOv8 axis-aligned emergency adapter (MVP1.5, replanned from an earlier RT-DETR target — see `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md`), time-boxed with a fallback to a tuned YOLOv8 baseline if not clearly ahead by the mid-week checkpoint

**— 🏛️ realistic congress-day line — everything above is shipped or targeted for this window; everything below is real production-MVP work that stays open after —**

- [x] Multi-level road support — multi-homography + plane assignment for bridges, ramps, overpasses (MVP3; `MultiPlaneTransform` + `--plane-zones`, see `application/WORLD_PROJECTION.md`). Not composed with multi-anchor projection (a moving-drone shoot over a grade-separated site needs both landed together, which isn't done) and untested against real grade-separated footage.
- [ ] Long-term identity persistence through occlusion — ReID (MVP5); occlusion is routine in dense urban traffic, and conflict metrics on fragmented tracks aren't trustworthy. Partial: the motion-plausibility-gate + track-merge mechanism landed (`application/reid_merge.py`, `--reid-merge`), but it's not yet usable end-to-end — the DINOv3 embedding stage that would actually produce real appearance vectors from footage doesn't exist (needs a GPU), so there's no way to run this on a real occlusion today.
- [ ] Learned ego-motion stabilization — SuperPoint + LightGlue replacing ORB where the cheap estimator measurably struggles (Backlog #1)
- [x] Multi-anchor world projection for moving, wide-swept drone footage (MVP2 remainder; `PerAnchorTransform`, nearest-anchor by frame-index — see `application/WORLD_PROJECTION.md`). The remaining MVP2 gap is the SuperPoint+LightGlue stabilization upgrade (`docs/BACKLOG.md` #1), tracked separately below.
- [x] Lane-level assignment (MVP6; `LaneZone`/`--lane-zones`, same hand-drawn-polygon mechanism as Link ID — see `docs/roadmap/road_topology.md`). Lane-change *conflict classification* itself is a downstream SSAM-reader concern, not TraTrac's — out of this repo's scope.
- [ ] Segmentation-based precise occupancy footprint — SAM 3, not SAM2 (MVP4 remainder; orientation is no longer this item's job, MVP1.5's OBB detector already provides it). Partial: the footprint sidecar format + `--footprint` dimension-override integration landed (`application/FOOTPRINT.md`, reusing Group A's OBB slot so no new consumer code was needed), but the segmentation stage that would actually produce a footprint (`cli_segment.py`, SAM 3) needs a GPU and doesn't exist yet — nothing can populate this today.
- [x] Link-ID road-segment identity for multi-link scenes — single-plane classification (rest of MVP3; multi-homography/plane assignment for grade-separated sites is the item above, not this one)
- [ ] Faster, hardware-accelerated video decode — TorchCodec + NVDEC, not PyAV (Backlog #3; PyAV stays for encode, which already shipped)
- [ ] Production-scale platform: async pipelines (MVP7 remainder). FiftyOne visualization (`tratrac-fiftyone`, optional `fiftyone` extra) has landed and is verified end-to-end against real footage (`docs/roadmap/mvp7.md`); a Docker/CUDA multi-stage build has also landed but isn't build-tested (no `docker` in the environment it was written in). Async pipeline stays deliberately unbuilt pending a GPU timing re-profile.

## Resolved blocker

`src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` references `scripts/probe_detector.py`
for detector-quality validation; it was missing from the repo and has been restored (generalized
to probe either the `rt_detr` or any `ultralytics` `yolo` checkpoint, detect or OBB task, not just
the original RT-DETR-only version) — see `docs/IMPLEMENTATION_PLAN.md` Group A0. The MVP1.5
fine-tune attempt above still needs a trained YOLO-OBB checkpoint and a GPU to produce one
(neither exists yet) before a credible before/after comparison can actually be run.
