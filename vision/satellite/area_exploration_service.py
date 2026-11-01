from pathlib import Path

from vision.satellite.area_exploration import AreaExplorationResult
from vision.satellite.reasoning import SatelliteReasoningService
from vision.satellite.satellite_vision_engine import SatelliteVisionEngine


class AreaExplorationService:
    def __init__(self):
        self.satellite_engine = SatelliteVisionEngine()
        self.reasoning_service = SatelliteReasoningService()

    def explore_prediction(
        self,
        area_name: str,
        center_latitude: float,
        center_longitude: float,
        prediction_path: str | Path,
    ) -> AreaExplorationResult:
        prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction raster not found: {prediction_path}"
            )

        prediction_summary = self.satellite_engine.summarize_prediction(
            prediction_path
        )

        land_cover = {}

        for class_id, details in prediction_summary["classes"].items():
            if details["pixel_count"] == 0:
                continue

            land_cover[str(class_id)] = details

        geographic_analysis = {
            "crs": None,
            "pixel_area_square_meters": prediction_summary[
                "pixel_area_square_meters"
            ],
            "predicted_classes": len(land_cover),
        }

        import rasterio

        with rasterio.open(prediction_path) as dataset:
            geographic_analysis["crs"] = (
                str(dataset.crs) if dataset.crs else None
            )
            geographic_analysis["width"] = dataset.width
            geographic_analysis["height"] = dataset.height
            geographic_analysis["bounds"] = {
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top,
            }

        result = AreaExplorationResult(
            area_name=area_name,
            center_latitude=float(center_latitude),
            center_longitude=float(center_longitude),
            land_cover=land_cover,
            geographic_analysis=geographic_analysis,
        )

        reasoning = self.reasoning_service.reason_about_area(
            result.to_dict()
        )

        result.reasoning = reasoning

        return result

    def reason_about_question(
        self,
        area_result: AreaExplorationResult,
        question: str,
    ) -> dict:
        if not isinstance(area_result, AreaExplorationResult):
            raise TypeError(
                "area_result must be an AreaExplorationResult."
            )

        return self.reasoning_service.reason_about_question(
            area_result.to_dict(),
            question,
        )
