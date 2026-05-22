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

## RT-DETR

- Chosen cause it provides:
  - transformer-based reasoning
  - excellent aerial detection
  - strong dense-scene handling
  - fewer duplicate detections

---

### Why NOT YOLO

YOLO prioritizes:

- speed
- simplicity

RT-DETR prioritizes:

- aerial robustness
- dense-scene quality
- global scene reasoning

#### MVP1 exception

`05_mvp1.md` documents a temporary YOLOv8-VisDrone adapter wired in as the
default detector at MVP1 ship. The reason is purely operational, not
architectural: there was no GPU available in the MVP1 timebox to fine-tune
RT-DETR on aerial data, and the COCO-pretrained RT-DETR weights are
unusable on cenital views. The YOLO adapter is contained in a single file
behind the existing `Detector` port and is scheduled for removal in MVP1.5
once a fine-tuned RT-DETR checkpoint exists.

---

## SAM2

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

## BoT-SORT

- Chosen cause it combines:
  - motion tracking
  - appearance embeddings
  - camera motion compensation

---

### Why NOT SORT

SORT:

- motion only
- poor long-term stability

---

## FastReID

- Used cause it provides:
  - vehicle-specific embeddings
  - long-term identity recovery
  - re-entry matching
