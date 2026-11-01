from dataclasses import dataclass
from typing import Any

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GeographicFeature:
    """
    Geographic feature exposed by the Computer Vision AI
    to the geospatial and routing layers.
    """

    feature_id: str
    feature_type: str
    geometry: BaseGeometry
    properties: dict[str, Any]

    @property
    def centroid(self):
        return self.geometry.centroid

    @property
    def bounds(self):
        return self.geometry.bounds

    def to_dict(self):
        centroid = self.centroid

        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "geometry_type": self.geometry.geom_type,
            "centroid": {
                "x": centroid.x,
                "y": centroid.y,
            },
            "bounds": self.bounds,
            "properties": self.properties,
        }


class SatelliteGeospatialBridge:
    """
    Converts satellite CV spatial features into geographic
    features that can later be consumed by road/place/routing
    systems.
    """

    def land_cover_to_geographic(
        self,
        feature,
    ) -> GeographicFeature:

        return GeographicFeature(
            feature_id=feature.feature_id,
            feature_type="land_cover",
            geometry=feature.geometry,
            properties={
                **feature.properties,
                "class_id": feature.class_id,
                "class_name": feature.class_name,
                "area_square_meters": feature.area_square_meters,
            },
        )

    def convert_features(
        self,
        features,
    ):
        return [
            self.land_cover_to_geographic(feature)
            for feature in features
        ]
