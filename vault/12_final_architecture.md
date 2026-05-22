# Final Ideal Architecture

```text
Aerial Video
    ↓
Hardware Decoding
    ↓
Video Stabilization
    ↓
Vehicle Detection
    ↓
Vehicle Segmentation
    ↓
Tracking
    ↓
ReID Matching
    ↓
Road Plane Assignment
    ↓
Multi-Homography Projection
    ↓
Topology Constraints
    ↓
EKF Motion Prediction
    ↓
Trajectory Smoothing
    ↓
Dual Export
        ├── SSAM .trj
        └── Internal Rich Format
    ↓
Persistent Storage
    ↓
Analytics Platform
```
