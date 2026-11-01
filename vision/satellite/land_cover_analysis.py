from pathlib import Path

import numpy as np
import rasterio


class LandCoverAnalysisService:
    """
    Converts a land-cover prediction raster into
    understandable class and area information.
    """

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

    def analyze(self, prediction_path: str | Path) -> dict:
        prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction raster not found: {prediction_path}"
            )

        with rasterio.open(prediction_path) as dataset:
            raster = dataset.read(1)

            pixel_width = abs(dataset.transform.a)
            pixel_height = abs(dataset.transform.e)
            pixel_area = pixel_width * pixel_height

            classes = {}

            for class_id, class_name in self.CLASS_NAMES.items():
                pixel_count = int(
                    np.count_nonzero(raster == class_id)
                )

                if pixel_count == 0:
                    continue

                area_square_meters = (
                    pixel_count * pixel_area
                )

                classes[str(class_id)] = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_count": pixel_count,
                    "area_square_meters": area_square_meters,
                    "area_hectares": (
                        area_square_meters / 10000.0
                    ),
                    "area_square_kilometers": (
                        area_square_meters / 1_000_000.0
                    ),
                }

            total_classified_pixels = sum(
                item["pixel_count"]
                for item in classes.values()
            )

            return {
                "prediction_path": str(prediction_path),
                "crs": (
                    str(dataset.crs)
                    if dataset.crs
                    else None
                ),
                "pixel_area_square_meters": pixel_area,
                "classified_pixel_count": (
                    total_classified_pixels
                ),
                "class_count": len(classes),
                "classes": classes,
            }
