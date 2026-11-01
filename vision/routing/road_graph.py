from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoadNode:
    """
    Geographic node in the road graph.
    """

    node_id: str
    latitude: float
    longitude: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoadEdge:
    """
    Directed road segment connecting two road nodes.

    distance_meters:
        Physical road-segment distance.

    travel_time_seconds:
        Optional predicted/observed travel time.
        This will later be produced by the ML travel-time model.
    """

    edge_id: str
    source: str
    target: str
    distance_meters: float
    travel_time_seconds: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.distance_meters < 0:
            raise ValueError(
                "distance_meters cannot be negative."
            )

        if (
            self.travel_time_seconds is not None
            and self.travel_time_seconds < 0
        ):
            raise ValueError(
                "travel_time_seconds cannot be negative."
            )


@dataclass
class RoadGraph:
    """
    Lightweight road-network graph.

    ML predicts edge costs.
    Optimization algorithms choose the route.
    """

    nodes: dict[str, RoadNode] = field(
        default_factory=dict
    )

    edges: dict[str, RoadEdge] = field(
        default_factory=dict
    )

    adjacency: dict[str, list[str]] = field(
        default_factory=dict
    )

    def add_node(self, node: RoadNode):
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(
            node.node_id,
            [],
        )

    def add_edge(self, edge: RoadEdge):
        if edge.source not in self.nodes:
            raise ValueError(
                f"Source node not found: {edge.source}"
            )

        if edge.target not in self.nodes:
            raise ValueError(
                f"Target node not found: {edge.target}"
            )

        self.edges[edge.edge_id] = edge

        self.adjacency.setdefault(
            edge.source,
            [],
        ).append(edge.edge_id)

    def outgoing_edges(self, node_id: str):
        return [
            self.edges[edge_id]
            for edge_id in self.adjacency.get(
                node_id,
                [],
            )
        ]
