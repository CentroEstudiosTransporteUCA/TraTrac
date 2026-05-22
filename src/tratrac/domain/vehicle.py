"""VehicleState: canonical internal representation of a vehicle at one timestep."""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D


@dataclass(frozen=True, slots=True)
class VehicleState:
	"""
	A vehicle's state at a single timestep.

	Per vault/01_architecture_principles.md this is the canonical internal type.
	MVP1 carries only the fields required for SSAM v1.04 export; later MVPs add
	segmentation polygons, ReID embeddings, plane metadata, and uncertainty.

	``link_id`` and ``lane_id`` default to 0 (the SSAM "unknown" sentinel). The
	application layer populates them when a road graph is available — see
	``vault/13_road_topology.md`` for sourcing strategy per MVP. ``lane_id`` is
	a Byte in the SSAM record, so its range is validated here.
	"""

	vehicle_id: int
	timestamp_seconds: float
	centroid: Point2D
	heading: Heading
	dimensions: Dimensions
	velocity: Vector2D
	acceleration: Vector2D
	link_id: int = 0
	lane_id: int = 0

	def __post_init__(self) -> None:
		if self.link_id < 0:
			raise ValueError(f"link_id must be non-negative, got {self.link_id}.")
		if not 0 <= self.lane_id <= 255:
			raise ValueError(f"lane_id must be in [0, 255] (SSAM Byte field), got {self.lane_id}.")

	@property
	def speed(self) -> float:
		"""Magnitude of the velocity vector. SSAM Speed field."""
		return self.velocity.magnitude

	@property
	def forward_acceleration(self) -> float:
		"""Component of acceleration along the heading. SSAM Acceleration field."""
		return self.acceleration.dx * self.heading.dx + self.acceleration.dy * self.heading.dy

	@property
	def front_bumper(self) -> Point2D:
		half_length = self.dimensions.length / 2.0
		return self.centroid.translate_by(self.heading.as_vector_with_magnitude(half_length))

	@property
	def rear_bumper(self) -> Point2D:
		half_length = self.dimensions.length / 2.0
		return self.centroid.translate_by(
			self.heading.reversed().as_vector_with_magnitude(half_length)
		)
