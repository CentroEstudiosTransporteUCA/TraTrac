"""GSD calibration sidecar I/O: the scale-transform-file reader/writer.

Sibling of ``src/tratrac/calibration/`` (the pure GSD math — sensor+focal+altitude
formula, drone-model registry, SRT altitude parsing): this package is the
infrastructure side, turning a resolved ``meters_per_pixel`` into its own sidecar
file (``infrastructure/transform/records.py``'s shared transform-record schema).
"""
