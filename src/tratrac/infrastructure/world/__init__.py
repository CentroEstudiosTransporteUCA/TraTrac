"""World-projection adapters: the calibration sidecar reader and the cv2 homography fit.

The infrastructure side of MVP2 (``src/tratrac/application/WORLD_PROJECTION.md``): turns a ``calibration.json`` of
image↔world correspondences into a ``Calibration`` value object, and fits the 3x3
homography from them (cv2). The pure projection math lives in
``application/world_projection.py``.
"""
