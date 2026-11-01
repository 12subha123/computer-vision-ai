from vision.satellite.geometric_analysis import LandCoverRegion
from vision.satellite.spatial_feature import SatelliteSpatialFeature


CLASS_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
}


class SatelliteFeatureBuilder:
    """
    Converts geometric land-cover regions into standardized
    spatial features for the geospatial intelligence layer.
    """

    def __init__(self):
        self.class_names = CLASS_NAMES.copy()

    def from_region(
        self,
        region: LandCoverRegion,
        feature_id: str,
    ) -> SatelliteSpatialFeature:

        class_name = self.class_names.get(
            region.class_id,
            f"Unknown class {region.class_id}",
        )

        return SatelliteSpatialFeature(
            feature_id=feature_id,
            class_id=region.class_id,
            class_name=class_name,
            geometry=region.geometry,
            area_square_meters=region.area_square_meters,
            properties={
                "source": "satellite_cnn",
                "land_cover_class": region.class_id,
            },
        )

    def from_regions(
        self,
        regions,
        prefix="satellite",
    ):
        features = []

        for index, region in enumerate(regions):
            feature_id = f"{prefix}_{index:06d}"

            features.append(
                self.from_region(
                    region=region,
                    feature_id=feature_id,
                )
            )

        return features
