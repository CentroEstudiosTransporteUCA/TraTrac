"""Domain layer: pure types and ports. No framework dependencies."""

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox, Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.ports import Detector, Tracker, TrajectoryExporter, VideoSource
from tratrac.domain.vehicle import VehicleState

__all__ = [
	"BoundingBox",
	"Detection",
	"Detector",
	"Dimensions",
	"Frame",
	"Heading",
	"Point2D",
	"TrackedDetection",
	"Tracker",
	"TrajectoryExporter",
	"Vector2D",
	"VehicleClass",
	"VehicleState",
	"VideoMetadata",
	"VideoSource",
]
