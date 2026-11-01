from pathlib import Path
import json

from vision.routing.coverage_osm_extractor import CoverageOSMRoadExtractor
from vision.routing.routing_coverage import RoutingCoverage
from vision.routing.road_data import RoadNetworkLoader
from vision.routing.shortest_path import ShortestPathEngine
from vision.routing.safe_nearest_node import SafeNearestRoadNodeFinder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OSM_PBF = PROJECT_ROOT / "data" / "osm" / "eastern-zone-latest.osm.pbf"

ROAD_NETWORK_DIR = PROJECT_ROOT / "data" / "osm" / "generated"


class SatelliteRoutingService:
    def __init__(self):
        self.loader = RoadNetworkLoader()
        self.path_engine = ShortestPathEngine()
        self.nearest_node = SafeNearestRoadNodeFinder()

    def _load_network(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return self.loader.load(
            nodes=data["nodes"],
            edges=data["edges"],
        )

    def route(
        self,
        source_latitude: float,
        source_longitude: float,
        target_latitude: float,
        target_longitude: float,
        margin_degrees: float = 0.02,
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
