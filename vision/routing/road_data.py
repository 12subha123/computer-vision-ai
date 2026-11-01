from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vision.routing.road_graph import RoadGraph, RoadNode, RoadEdge


@dataclass(frozen=True)
class RoadNetworkSource:
    """
    Describes a real-world road-network dataset.

    The actual dataset is intentionally not downloaded here.
    """

    name: str
    path: Path
    format: str

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Road-network source name cannot be empty.")

        if not self.format.strip():
            raise ValueError("Road-network format cannot be empty.")


class RoadNetworkLoader:
    """
    Converts verified road-network records into RoadGraph objects.

    Expected node record:
        {
            "node_id": "...",
            "latitude": ...,
            "longitude": ...
        }

    Expected edge record:
        {
            "edge_id": "...",
            "source": "...",
            "target": "...",
            "distance_meters": ...,
            "travel_time_seconds": optional
        }
    """

    def load(
        self,
        nodes: Iterable[dict],
        edges: Iterable[dict],
    ) -> RoadGraph:

        graph = RoadGraph()

        for record in nodes:
            node = RoadNode(
                node_id=str(record["node_id"]),
                latitude=float(record["latitude"]),
                longitude=float(record["longitude"]),
                properties=dict(
                    record.get("properties", {})
                ),
            )

            graph.add_node(node)

        for record in edges:
            edge = RoadEdge(
                edge_id=str(record["edge_id"]),
                source=str(record["source"]),
                target=str(record["target"]),
                distance_meters=float(
                    record["distance_meters"]
                ),
                travel_time_seconds=(
                    None
                    if record.get("travel_time_seconds") is None
                    else float(
                        record["travel_time_seconds"]
                    )
                ),
                properties=dict(
                    record.get("properties", {})
                ),
            )

            graph.add_edge(edge)

        return graph
