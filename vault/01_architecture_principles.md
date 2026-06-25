# Architecture Principles

---

## Fundamental Architectural Principle

The system should NEVER use:

- SSAM `.trj`
as:
- the internal canonical representation.

Instead:

### Internal Representation

Stores:

- polygons
- masks
- embeddings
- uncertainty
- topology metadata
- multi-plane information
- arbitrary analytics metadata

AND

### External Exporters

Generate:

- SSAM `.trj`
- internal analytics formats
- debug formats

This avoids crippling future capabilities.

---

## Export Strategy

### Dual Export Architecture

The system exports:

#### Exporter A — SSAM `.trj`

Compatible with:

- SSAM
- traffic safety analytics
- conflict analysis tools

AND

#### Exporter B — Extended Internal Format

Containing:

- segmentation polygons
- embeddings
- topology metadata
- uncertainty metrics
- plane metadata
- debugging information

#### Production order: B-first

The perception pipeline produces **only B** — the canonical internal record (raw
tracked measurements today; the richer fields above as later MVPs add them). **A
(the SSAM `.trj`) is derived post-hoc** from B by `tratrac-smooth`, which runs the
trajectory smoother and reconstructs kinematics. The pipeline never computes
kinematics inline and never re-ingests A. This keeps A a pure, lossy *export* and B
the single source of truth, and lets the `.trj` be re-derived/re-tuned offline with
no re-detection (see `22_smoothing.md`). Consequence: "valid SSAM at every MVP" is
satisfied in two steps (run → smooth), not one command.

---

### Why Dual Export Is Necessary

SSAM was designed for:

- traffic simulation
- simplified vehicle geometry

NOT:

- segmentation masks
- neural embeddings
- arbitrary CV metadata

Using SSAM as the internal format would severely limit:

- analytics
- debugging
- future ML improvements

---

## Canonical Internal Vehicle Representation

```text
VehicleState
    id
    timestamp

    position_world
    heading_vector

    length
    width

    front_point
    rear_point

    velocity
    acceleration

    segmentation_polygon
    reid_embedding
    road_plane
    lane_id

    uncertainty_metrics
```

---

### Why This Representation

This representation:

- decouples internal logic from SSAM
- preserves future flexibility
- enables advanced analytics
- simplifies exporters

---

### Mapping Internal Representation To SSAM

| Internal Representation | SSAM Field |
| --- | --- |
| front_point | Front X/Y/Z |
| rear_point | Rear X/Y/Z |
| velocity | Speed |
| acceleration | Acceleration |
| lane_id | Lane ID |
| length | Vehicle Length |
| width | Vehicle Width |

`velocity` is a 2D vector; SSAM **Speed** takes its magnitude (`speed = |velocity|`). `acceleration`,
by contrast, is stored as the **scalar longitudinal acceleration** — the rate of change of speed
(`d|v|/dt`) — and maps straight to SSAM **Acceleration**. It is *not* a vector projected onto the
heading: that projection conflated cornering with speeding-up/slowing-down, firing spuriously through
constant-speed turns as the smoothed heading lagged the rotating velocity. The lateral/centripetal
component of acceleration is intentionally not retained (SSAM cannot express it, and no analytics need
it yet); reintroduce a full acceleration vector only when an analytic consumer requires lateral g.
