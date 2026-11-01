from typing import Any

from vision.satellite.area_exploration_service import (
    AreaExplorationService,
)


class SatelliteIntelligenceService:
    """
    User-facing orchestration layer for Satellite Intelligence.

    Individual scientific and routing services remain separate.
    This class coordinates them so the API/UI does not need to
    understand the internal implementation details.
    """

    def __init__(self):
        self.exploration_service = AreaExplorationService()

    def explore_area(
        self,
        area_name: str,
        center_latitude: float,
        center_longitude: float,
        prediction_path: str,
    ) -> dict[str, Any]:
        result = self.exploration_service.explore_prediction(
            area_name=area_name,
            center_latitude=center_latitude,
            center_longitude=center_longitude,
            prediction_path=prediction_path,
        )

        return result.to_dict()

    def ask_about_area(
        self,
        area_result: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        from vision.satellite.area_exploration import (
            AreaExplorationResult,
        )

        result = AreaExplorationResult(
            area_name=area_result["area_name"],
            center_latitude=area_result["center"]["latitude"],
            center_longitude=area_result["center"]["longitude"],
            land_cover=area_result.get("land_cover", {}),
            geographic_analysis=area_result.get(
                "geographic_analysis",
                {},
            ),
            changes=area_result.get("changes", {}),
            route_intelligence=area_result.get(
                "route_intelligence",
                {},
            ),
            reasoning=area_result.get("reasoning", {}),
            optimization=area_result.get(
                "optimization",
                {},
            ),
        )

        return self.exploration_service.reason_about_question(
            area_result=result,
            question=question,
        )
