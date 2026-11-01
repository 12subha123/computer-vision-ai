from dataclasses import dataclass
from typing import Any

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class SatelliteSpatialFeature:
    """
    Standard spatial feature produced by the satellite CV engine.
    """

    feature_id: str
    class_id: int
    class_name: str
    geometry: BaseGeometry
    area_square_meters: float
    properties: dict[str, Any]

    @property
    def centroid(self):
        return self.geometry.centroid

    @property
    def bounds(self):
        return self.geometry.bounds

    @property
    def geometry_type(self) -> str:
        return self.geometry.geom_type

    def to_dict(self):
        centroid = self.centroid

        return {
            "feature_id": self.feature_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "geometry_type": self.geometry_type,
            "area_square_meters": self.area_square_meters,
            "centroid": {
                "x": centroid.x,
                "y": centroid.y,
            },
            "bounds": self.bounds,
            "properties": self.properties,
        }
