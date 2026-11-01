from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    distance_km: float
    travel_time_minutes: float | None = None
    metadata: dict[str, Any] | None = None


class RouteOptimizationService:
    """
    Selects the best route candidate according to a user objective.

    This layer does not calculate roads itself. It operates above the
    existing Route Intelligence service, so the current A-to-B routing
    workflow remains unchanged.
    """

    OBJECTIVES = {
        "shortest": "distance",
        "fastest": "travel_time",
        "balanced": "balanced",
    }

    def optimize(
        self,
        candidates: list[RouteCandidate],
        objective: str = "shortest",
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("At least one route candidate is required.")

        objective = objective.strip().lower()

        if objective not in self.OBJECTIVES:
            raise ValueError(
                f"Unsupported optimization objective: {objective}"
            )

        if objective == "shortest":
            selected = min(
                candidates,
                key=lambda route: route.distance_km,
            )

        elif objective == "fastest":
            available = [
                route
                for route in candidates
                if route.travel_time_minutes is not None
            ]

            if not available:
                raise ValueError(
                    "Fastest optimization requires travel-time estimates."
                )

            selected = min(
                available,
                key=lambda route: route.travel_time_minutes,
            )

        else:
            available = [
                route
                for route in candidates
                if route.travel_time_minutes is not None
            ]

            if not available:
                raise ValueError(
                    "Balanced optimization requires travel-time estimates."
                )

            max_distance = max(
                route.distance_km for route in available
            )
            max_time = max(
                route.travel_time_minutes
                for route in available
            )

            def balanced_score(route: RouteCandidate) -> float:
                distance_score = (
                    route.distance_km / max_distance
                    if max_distance > 0
                    else 0.0
                )
                time_score = (
                    route.travel_time_minutes / max_time
                    if max_time > 0
                    else 0.0
                )
                return (distance_score + time_score) / 2.0

            selected = min(
                available,
                key=balanced_score,
            )

        return {
            "objective": objective,
            "selected_route": {
                "route_id": selected.route_id,
                "distance_km": selected.distance_km,
                "travel_time_minutes": selected.travel_time_minutes,
                "metadata": selected.metadata or {},
            },
            "candidate_count": len(candidates),
        }
