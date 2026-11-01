from dataclasses import dataclass
from typing import Any

from vision.routing.road_graph import RoadGraph
from vision.routing.shortest_path import ShortestPathEngine


@dataclass(frozen=True)
class AlternativeRoute:
    route_id: str
    node_ids: list[str]
    edge_ids: list[str]
    distance_meters: float
    metadata: dict[str, Any]


class AlternativeRouteEngine:
    """
    Generates genuinely different route candidates from a RoadGraph.

    The original ShortestPathEngine remains unchanged.

    Strategy:
        1. Find the shortest route.
        2. Temporarily penalize each edge of the selected route.
        3. Re-run Dijkstra.
        4. Keep routes that differ from previously discovered routes.
        5. Return the shortest unique alternatives.

    This produces graph-based alternatives rather than artificial
    distance/time variations.
    """

    def __init__(self):
        self.path_engine = ShortestPathEngine()

    def _route_signature(
        self,
        edge_ids: list[str],
    ) -> tuple[str, ...]:
        return tuple(edge_ids)

    def _edge_penalty(
        self,
        graph: RoadGraph,
        blocked_edge_ids: set[str],
    ) -> RoadGraph:
        penalized_graph = RoadGraph()

        for node in graph.nodes.values():
            penalized_graph.add_node(node)

        for edge in graph.edges.values():
            if edge.edge_id in blocked_edge_ids:
                continue

            penalized_graph.add_edge(edge)

        return penalized_graph

    def generate(
        self,
        graph: RoadGraph,
        source: str,
        target: str,
        max_routes: int = 3,
        minimum_difference_edges: int = 2,
    ) -> list[AlternativeRoute]:
        if max_routes < 1:
            raise ValueError(
                "max_routes must be at least 1."
            )

        if minimum_difference_edges < 1:
            raise ValueError(
                "minimum_difference_edges must be at least 1."
            )

        first_route = self.path_engine.find_route(
            graph=graph,
            source=source,
            target=target,
            cost_mode="distance",
        )

        candidates: list[AlternativeRoute] = [
            AlternativeRoute(
                route_id="route_1",
                node_ids=first_route.node_ids,
                edge_ids=first_route.edge_ids,
                distance_meters=first_route.total_cost,
                metadata={
                    "type": "shortest",
                    "alternative": False,
                },
            )
        ]

        if max_routes == 1:
            return candidates

        discovered_signatures = {
            self._route_signature(first_route.edge_ids)
        }

        for base_route in list(candidates):
            for edge_id in base_route.edge_ids:
                if len(candidates) >= max_routes:
                    break

                blocked = {edge_id}

                try:
                    graph_without_edge = self._edge_penalty(
                        graph,
                        blocked,
                    )

                    route = self.path_engine.find_route(
                        graph=graph_without_edge,
                        source=source,
                        target=target,
                        cost_mode="distance",
                    )
                except ValueError:
                    continue

                signature = self._route_signature(
                    route.edge_ids
                )

                if signature in discovered_signatures:
                    continue

                if self._edge_difference(
                    route.edge_ids,
                    first_route.edge_ids,
                ) < minimum_difference_edges:
                    continue

                discovered_signatures.add(signature)

                candidates.append(
                    AlternativeRoute(
                        route_id=f"route_{len(candidates) + 1}",
                        node_ids=route.node_ids,
                        edge_ids=route.edge_ids,
                        distance_meters=route.total_cost,
                        metadata={
                            "type": "alternative",
                            "alternative": True,
                            "blocked_edge": edge_id,
                        },
                    )
                )

        candidates.sort(
            key=lambda route: route.distance_meters
        )

        renamed_candidates = []

        for index, route in enumerate(
            candidates[:max_routes],
            start=1,
        ):
            renamed_candidates.append(
                AlternativeRoute(
                    route_id=f"route_{index}",
                    node_ids=route.node_ids,
                    edge_ids=route.edge_ids,
                    distance_meters=route.distance_meters,
                    metadata=route.metadata,
                )
            )

        return renamed_candidates

    def _edge_difference(
        self,
        first: list[str],
        second: list[str],
    ) -> int:
        first_set = set(first)
        second_set = set(second)

        return len(first_set.symmetric_difference(second_set))
