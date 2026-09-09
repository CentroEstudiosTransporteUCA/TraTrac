# `infrastructure/video/` Design Docs

| Doc | Covers |
| --- | --- |
| [`EGO_MOTION.md`](EGO_MOTION.md) | MVP1.9 keyframe-anchored ORB ego-motion estimator: design decision, transform model, keyframe anchoring, ORB-vs-ECC, vehicle-masking, limitations |
| [`TIME_WINDOW.md`](TIME_WINDOW.md) | `--start`/`--end` analysis-window trimming |

Per-frame transform *persistence* (the `TransformSink` sidecar this estimator's output feeds)
is a separate doc at [`../transform/TRANSFORM_SINK.md`](../transform/TRANSFORM_SINK.md).

See [`../../../../docs/README.md`](../../../../docs/README.md) for the full repo documentation map.
