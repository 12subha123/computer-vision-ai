from pathlib import Path

from vision.satellite.geometric_analysis import SatelliteGeometricAnalyzer
from vision.satellite.land_cover import SatelliteLandCoverAdapter
from vision.satellite.sentinel2_scene import Sentinel2Scene


class SatelliteVisionEngine:
    """
    Central satellite-imagery intelligence engine.

    Pipeline:

        Sentinel-2 B02/B03/B04/B08
                    ↓
        Real trained land-cover CNN
                    ↓
        Prediction + confidence rasters
                    ↓
        Spatial / geometric analysis

    The trained model itself is never modified.
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

    def __init__(
        self,
        checkpoint_path=None,
        statistics_path=None,
    ):
        self.land_cover = SatelliteLandCoverAdapter(
            checkpoint_path=checkpoint_path,
            statistics_path=statistics_path,
        )
        self.geometric_analyzer = SatelliteGeometricAnalyzer()

    def analyze_scene(
        self,
        scene: Sentinel2Scene,
        output_directory: str | Path,
        patch_size: int = 64,
        stride: int = 64,
    ):
        """
        Run land-cover inference on a complete Sentinel-2 scene.

        Returns paths and metadata for the generated prediction
        and confidence rasters.
        """

        if not isinstance(scene, Sentinel2Scene):
            raise TypeError(
                "scene must be a Sentinel2Scene instance."
            )

        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_path = (
            output_directory
            / f"{scene.scene_id}_land_cover.tif"
        )

        result = self.land_cover.predict_raster(
            band_paths=scene.band_paths,
            output_path=prediction_path,
            patch_size=patch_size,
            stride=stride,
        )

        return {
            "scene_id": scene.scene_id,
            "prediction_path": str(result["prediction_path"]),
            "confidence_path": str(result["confidence_path"]),
            "width": result["width"],
            "height": result["height"],
            "patch_size": result["patch_size"],
            "stride": result["stride"],
            "classes": dict(self.CLASS_NAMES),
        }

    def analyze_class(
        self,
        prediction_path: str | Path,
        class_id: int,
    ):
        """
        Extract spatial regions belonging to one land-cover class.
        """

        class_id = int(class_id)

        if class_id not in self.CLASS_NAMES:
            raise ValueError(
                f"Unsupported land-cover class: {class_id}"
            )

        regions = (
            self.geometric_analyzer.analyze_prediction_raster(
                prediction_path=prediction_path,
                class_id=class_id,
            )
        )

        return {
            "class_id": class_id,
            "class_name": self.CLASS_NAMES[class_id],
            "region_count": len(regions),
            "regions": [
                {
                    "class_id": region.class_id,
                    "class_name": self.CLASS_NAMES[
                        region.class_id
                    ],
                    "area_pixels": region.area_pixels,
                    "area_square_meters": (
                        region.area_square_meters
                    ),
                    "geometry": region.geometry,
                }
                for region in regions
            ],
        }

    def summarize_prediction(
        self,
        prediction_path: str | Path,
    ):
        """
        Return a simple class-area summary from a prediction raster.
        """

        prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction raster not found: {prediction_path}"
            )

        import numpy as np
        import rasterio

        with rasterio.open(prediction_path) as dataset:
            raster = dataset.read(1)
            transform = dataset.transform

            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)
            pixel_area = pixel_width * pixel_height

        summary = {}

        for class_id, class_name in self.CLASS_NAMES.items():
            pixel_count = int(
                np.count_nonzero(raster == class_id)
            )

            summary[class_id] = {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "area_square_meters": (
                    pixel_count * pixel_area
                ),
                "area_hectares": (
                    pixel_count * pixel_area / 10000.0
                ),
            }

        return {
            "prediction_path": str(prediction_path),
            "pixel_area_square_meters": pixel_area,
            "classes": summary,
        }
