# Coordinate Systems and Geometry

---

## MVP1

Produces:

- syntactically valid SSAM `.trj`

BUT:

- coordinates are not yet physically meaningful

This MVP validates:

- pipeline correctness
- serialization
- tracking
- exporter architecture

---

## MVP2+

All SSAM exports use:

### world-space metric coordinates

This is mandatory because SSAM assumes:

- metric geometry
- real distances
- real speeds
- real accelerations

NOT:

- image pixels

---

### Why Image-Space Coordinates Break SSAM

Passing image-space coordinates into SSAM causes:

| Problem | Consequence |
| --- | --- |
| Pixel distance ≠ meters | Invalid TTC calculations |
| Perspective distortion | Invalid vehicle sizes |
| Camera motion | Fake accelerations |
| Multi-level roads overlap visually | False conflicts |
| Spatial scale varies | Invalid analytics |

Meaning:

- SSAM may still parse the file
- but the analytics become scientifically invalid

---

## Multi-Homography Geometry

### Why

Single homography assumes:

```text
all roads exist on the same plane
```

This breaks for:

- bridges
- ramps
- stacked highways
- overpasses

Multi-homography enables:

- multi-level road support
- physically correct trajectories

---

### Why NOT Full 3D Reconstruction

Full 3D:

- expensive
- operationally difficult
- unnecessary for road-relative analytics

Roads are:

- piecewise planar

not arbitrary 3D scenes.

---

### Cost

#### Pros

- Correct geometry for bridges/overpasses

#### Cons

- Requires calibration
- Requires road-plane annotations
