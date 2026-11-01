from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


@dataclass(frozen=True)
class LandCoverRegion:
    class_id: int
    area_pixels: int
    area_square_meters: float
    geometry: object


class SatelliteGeometricAnalyzer:
    """
    Converts a land-cover prediction raster into connected
    spatial regions for geographic analysis.
    """

    def analyze_prediction_raster(
        self,
        prediction_path: str | Path,
        class_id: int,
    ):
        prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction raster not found: {prediction_path}"
            )

        class_id = int(class_id)

        with rasterio.open(prediction_path) as dataset:
            raster = dataset.read(1)
            transform = dataset.transform

            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)
            pixel_area = pixel_width * pixel_height

            mask = raster == class_id

            if not np.any(mask):
                return []

            regions = []

            for geometry_json, value in shapes(
                mask.astype(np.uint8),
                mask=mask,
                transform=transform,
            ):
                if int(value) != 1:
                    continue

                geometry = shape(geometry_json)

                if geometry.is_empty:
                    continue

                area_square_meters = float(
                    geometry.area
                )

                area_pixels = int(
                    round(
                        area_square_meters
                        / pixel_area
                    )
                )

                regions.append(
                    LandCoverRegion(
                        class_id=class_id,
                        area_pixels=area_pixels,
                        area_square_meters=area_square_meters,
                        geometry=geometry,
                    )
                )

            return regions
