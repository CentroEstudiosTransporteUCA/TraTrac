# Roadmap: Not-Yet-Started Capabilities

These MVPs have no code yet, so — unlike the shipped/partial MVPs, whose docs now live next to
their implementation (see [`../README.md`](../README.md)) — they stay here as planning
documents until work on them begins. When one starts, extract its content into a doc next to
the new code and leave a stub here pointing at it.

| Doc | Capability | Status |
| --- | --- | --- |
| [`mvp3.md`](mvp3.md) | Multi-plane world trajectories (bridges/overpasses, multi-homography) | ❌ Not started |
| [`mvp4.md`](mvp4.md) | SAM 3 segmentation for occupancy footprint (orientation now comes from MVP1.5's OBB detector, not this MVP) | 🟡 Dual-export "B-first" pulled forward; segmentation not started |
| [`mvp5.md`](mvp5.md) | DINOv3 ReID + motion-plausibility gating for long-term identity persistence (not FastReID) | ❌ Not started |
| [`mvp6.md`](mvp6.md) | Lane-graph topology constraints | ❌ Not started |
| [`mvp7.md`](mvp7.md) | Production-grade analytics platform (FiftyOne, Docker/CUDA) | 🟡 Parquet storage pulled forward; rest not started |
| [`road_topology.md`](road_topology.md) | Where SSAM `Link ID` / `Lane ID` are sourced from, per MVP (Strategy A/B/C comparison) | Reference doc spanning MVP3–7 |

See [`../ROADMAP.md`](../ROADMAP.md) for the full capability-ladder-vs-execution-order status
table, including the shipped/partial MVPs whose docs have already moved next to their code.
