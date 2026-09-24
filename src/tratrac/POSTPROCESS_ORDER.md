# `cli_postprocess.py` pipeline stage order

Detection-orientation work (richer `TrackSample`), geometry (plane/link/lane assignment,
multi-anchor projection), and perception enrichment (ReID-merge, footprint) each add a step to
`postprocess`. This is the order they run in, and why it can't be reshuffled freely:

```
read record
  → apply --footprint               [obb_w/obb_h override — keyed by the pre-merge track_id]
  → apply --reid-merge              [track_id remap — before anything else track-lifetime-aware]
  → filter --exclusion-zones
  → assign plane / link / lane      [needed before projection knows which H to use]
  → project --calibration           [projector itself is plane/anchor-aware]
  → smooth (Kalman/RTS)
  → export .trj
```

`--footprint` runs ahead of `--reid-merge` even though ReID-merge is the track-lifetime-aware
step everything else keys off of: a footprint sidecar is itself keyed by the pre-merge
`track_id`, so it has to be applied before the remap for the same reason `--reid-merge` has to
run before exclusion/link/lane/projection — both are "resolve identity/shape before anything
that reasons about a track's full lifetime or position" constraints, footprint's constraint is
just one step earlier in the chain.
