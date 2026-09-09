# Tracker Choice: BoT-SORT (not SORT)

`docs/TECH_STACK.md`'s ideal stack names BoT-SORT, not plain SORT, deliberately.

## Why BoT-SORT

- combines motion tracking with appearance embeddings and camera-motion compensation
- the appearance/embedding side is what enables ReID-based long-term identity persistence
  (MVP5, `docs/roadmap/mvp5.md`) — plain SORT has no hook for it

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

`boxmot` is **AGPL-3.0** — relevant if TraTrac is distributed (see `CLAUDE.md` Dependency Notes).
