# Ideal Final Tech Stack

| Layer | Technology |
| --- | --- |
| Runtime | PyTorch |
| Video Decoding | NVIDIA NVDEC + PyAV |
| Stabilization | SuperPoint + LightGlue |
| Detection | RT-DETR |
| Segmentation | SAM2 |
| Tracking | BoT-SORT |
| ReID | FastReID |
| Motion Modeling | Extended Kalman Filter |
| Geometry | Multi-Homography OpenCV System |
| Plane Assignment | Polygon-Based Plane Mapping |
| Topology Constraints | Lane Graph Model |
| Storage | Apache Parquet |
| Visualization | FiftyOne |
| Annotation | CVAT |
| Deployment | Docker + CUDA |
| Hardware | RTX 4090 / A100 |

---

## RT-DETR vs YOLO (shipped decision)

Moved to `src/tratrac/infrastructure/detection/DETECTOR_CHOICE.md` — the detector adapters
(`rt_detr.py`, `yolov8_visdrone.py`) live in `src/tratrac/infrastructure/detection/`, and that
doc covers the full decision including the MVP1 YOLOv8-VisDrone emergency exception and the
open MVP1.5 fine-tuning work.

---

## BoT-SORT vs SORT (shipped decision)

Moved to `src/tratrac/infrastructure/tracking/TRACKER_CHOICE.md` — next to the tracker adapter
(`infrastructure/tracking/boxmot_bot_sort.py`).

---

## SAM2 (not yet adopted — MVP4)

- Required for:
  - precise occupancy masks
  - temporal segmentation consistency
  - dense traffic handling

---

### Why NOT Mask R-CNN

SAM2:

- much better temporal consistency
- better mask quality
- better occlusion handling

---

## FastReID (not yet adopted — MVP5)

- Used cause it provides:
  - vehicle-specific embeddings
  - long-term identity recovery
  - re-entry matching
