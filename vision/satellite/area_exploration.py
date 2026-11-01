from dataclasses import dataclass, field
from typing import Any


@dataclass
class AreaExplorationResult:
    """
    Unified result returned by the Explore an Area capability.

    This keeps complex satellite/geospatial processing behind
    a simple result structure that the future UI can consume.
    """

    area_name: str
    center_latitude: float
    center_longitude: float
    land_cover: dict[str, Any] = field(default_factory=dict)
    geographic_analysis: dict[str, Any] = field(default_factory=dict)
    changes: dict[str, Any] = field(default_factory=dict)
    route_intelligence: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_name": self.area_name,
            "center": {
                "latitude": self.center_latitude,
                "longitude": self.center_longitude,
            },
            "land_cover": self.land_cover,
            "geographic_analysis": self.geographic_analysis,
            "changes": self.changes,
            "route_intelligence": self.route_intelligence,
            "reasoning": self.reasoning,
            "optimization": self.optimization,
        }
