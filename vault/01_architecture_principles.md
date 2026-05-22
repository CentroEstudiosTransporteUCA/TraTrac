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
