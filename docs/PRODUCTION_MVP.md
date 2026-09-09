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
- [ ] **Validated against real congress-site footage** — calibration, exclusion zones, smoother tuned until results are demonstrably clean (`validate_trj` compliance)
- [ ] **Aerial-robust detector** — fine-tuned RT-DETR replacing the YOLOv8 emergency adapter (MVP1.5), time-boxed with a fallback to a tuned YOLOv8 baseline if not clearly ahead by the mid-week checkpoint

**— 🏛️ realistic congress-day line — everything above is shipped or targeted for this window; everything below is real production-MVP work that stays open after —**

- [ ] Multi-level road support — multi-homography + plane assignment for bridges, ramps, overpasses (MVP3); without it, any site with grade separation produces false conflicts
- [ ] Long-term identity persistence through occlusion — ReID (MVP5); occlusion is routine in dense urban traffic, and conflict metrics on fragmented tracks aren't trustworthy
- [ ] Learned ego-motion stabilization — SuperPoint + LightGlue replacing ORB where the cheap estimator measurably struggles (Backlog #1)
- [ ] Multi-anchor / full world projection for moving, wide-swept drone footage (MVP2 remainder)
- [ ] Lane-level assignment + lane-change conflict classification (MVP6)
- [ ] Segmentation-based precise vehicle geometry & orientation — SAM2 (MVP4 remainder)
- [ ] Link-ID road-segment identity for multi-link scenes (rest of MVP3)
- [ ] Faster, hardware-accelerated video decode — PyAV + NVDEC (Backlog #3; encode side already shipped)
- [ ] Production-scale platform: FiftyOne visualization, async pipelines, Docker/CUDA deployment (MVP7 remainder)

## Known blocker

`src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` references `scripts/probe_detector.py`
for detector-quality validation, but that script does not exist in the repo. This blocks a
credible before/after comparison for the MVP1.5 fine-tune attempt above until it's restored or
replaced.
