from pathlib import Path
import json

from vision.routing.alternative_routes import AlternativeRouteEngine
from vision.routing.coverage_osm_extractor import CoverageOSMRoadExtractor
from vision.routing.routing_coverage import RoutingCoverage
from vision.routing.road_data import RoadNetworkLoader
from vision.routing.shortest_path import ShortestPathEngine
from vision.routing.safe_nearest_node import SafeNearestRoadNodeFinder
from vision.satellite.optimization import (
    RouteCandidate,
    RouteOptimizationService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OSM_PBF = PROJECT_ROOT / "data" / "osm" / "eastern-zone-latest.osm.pbf"

ROAD_NETWORK_DIR = PROJECT_ROOT / "data" / "osm" / "generated"


class SatelliteRoutingService:
    def __init__(self):
        self.loader = RoadNetworkLoader()
        self.path_engine = ShortestPathEngine()
        self.nearest_node = SafeNearestRoadNodeFinder()
        self.alternative_route_engine = AlternativeRouteEngine()
        self.optimization_service = RouteOptimizationService()

    def _load_network(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return self.loader.load(
            nodes=data["nodes"],
            edges=data["edges"],
        )

    def _get_network(
        self,
        source_latitude: float,
        source_longitude: float,
        target_latitude: float,
        target_longitude: float,
        margin_degrees: float,
    ):
        if not OSM_PBF.exists():
            raise FileNotFoundError(
                f"OSM data not found: {OSM_PBF}"
            )

        coverage = RoutingCoverage.from_points(
            source_latitude,
            source_longitude,
            target_latitude,
            target_longitude,
            margin_degrees=margin_degrees,
        )

        ROAD_NETWORK_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        network_name = (
            f"route_"
            f"{source_latitude:.5f}_"
            f"{source_longitude:.5f}_"
            f"{target_latitude:.5f}_"
            f"{target_longitude:.5f}.json"
        )

        network_path = ROAD_NETWORK_DIR / network_name

        if not network_path.exists():
            extractor = CoverageOSMRoadExtractor()

            extractor.extract(
                pbf_path=OSM_PBF,
                coverage=coverage,
                output_path=network_path,
            )

        graph = self._load_network(network_path)

        source = self.nearest_node.find(
            graph,
            source_latitude,
            source_longitude,
        )

        target = self.nearest_node.find(
            graph,
            target_latitude,
            target_longitude,
        )

        return (
            graph,
            source,
            target,
            coverage,
            network_path,
        )

    def route(
        self,
        source_latitude: float,
        source_longitude: float,
        target_latitude: float,
        target_longitude: float,
        margin_degrees: float = 0.02,
    ):
        (
            graph,
            source,
            target,
            coverage,
            network_path,
        ) = self._get_network(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
            margin_degrees=margin_degrees,
        )

        route = self.path_engine.find_route(
            graph=graph,
            source=source["node_id"],
            target=target["node_id"],
            cost_mode="distance",
        )

        coordinates = []

        for node_id in route.node_ids:
            node = graph.nodes[node_id]
            coordinates.append(
                {
                    "latitude": node.latitude,
                    "longitude": node.longitude,
                }
            )

        return {
            "source": source,
            "target": target,
            "route": {
                "node_ids": route.node_ids,
                "edge_ids": route.edge_ids,
                "distance_meters": route.total_cost,
                "distance_km": route.total_cost / 1000.0,
                "coordinates": coordinates,
            },
            "coverage": {
                "bounds": coverage.bounds,
            },
            "network": {
                "path": str(network_path),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
        }

    def optimize_route(
        self,
        source_latitude: float,
        source_longitude: float,
        target_latitude: float,
        target_longitude: float,
        objective: str = "shortest",
        max_routes: int = 3,
        minimum_difference_edges: int = 2,
        margin_degrees: float = 0.02,
    ):
        if max_routes < 1:
            raise ValueError(
                "max_routes must be at least 1."
            )

        (
            graph,
            source,
            target,
            coverage,
            network_path,
        ) = self._get_network(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
            margin_degrees=margin_degrees,
        )

        alternative_routes = (
            self.alternative_route_engine.generate(
                graph=graph,
                source=source["node_id"],
                target=target["node_id"],
                max_routes=max_routes,
                minimum_difference_edges=minimum_difference_edges,
            )
        )

        candidates = []

        for route in alternative_routes:
            candidates.append(
                RouteCandidate(
                    route_id=route.route_id,
                    distance_km=(
                        route.distance_meters / 1000.0
                    ),
                    travel_time_minutes=None,
                    metadata={
                        **route.metadata,
                        "node_ids": route.node_ids,
                        "edge_ids": route.edge_ids,
                    },
                )
            )

        optimization = self.optimization_service.optimize(
            candidates=candidates,
            objective=objective,
        )

        selected_route_id = optimization[
            "selected_route"
        ]["route_id"]

        route_results = []

        for candidate in candidates:
            route_coordinates = []

            for node_id in candidate.metadata["node_ids"]:
                node = graph.nodes[node_id]

                route_coordinates.append(
                    {
                        "latitude": node.latitude,
                        "longitude": node.longitude,
                    }
                )

            route_results.append(
                {
                    "route_id": candidate.route_id,
                    "distance_km": candidate.distance_km,
                    "travel_time_minutes": (
                        candidate.travel_time_minutes
                    ),
                    "selected": (
                        candidate.route_id
                        == selected_route_id
                    ),
                    "coordinates": route_coordinates,
                    "node_ids": candidate.metadata[
                        "node_ids"
                    ],
                    "edge_ids": candidate.metadata[
                        "edge_ids"
                    ],
                    "metadata": {
                        key: value
                        for key, value in candidate.metadata.items()
                        if key not in {
                            "node_ids",
                            "edge_ids",
                        }
                    },
                }
            )

        return {
            "objective": optimization["objective"],
            "selected_route": optimization[
                "selected_route"
            ],
            "candidates": route_results,
            "candidate_count": len(route_results),
            "source": source,
            "target": target,
            "coverage": {
                "bounds": coverage.bounds,
            },
            "network": {
                "path": str(network_path),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
            "limitations": {
                "travel_time_available": False,
                "message": (
                    "The current OSM road network does not "
                    "contain travel-time estimates. "
                    "Shortest optimization is based on "
                    "physical road distance. Fastest and "
                    "time-aware balanced optimization will "
                    "be enabled when travel-time prediction "
                    "is available."
                ),
            },
        }
