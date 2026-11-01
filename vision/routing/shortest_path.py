import heapq
from dataclasses import dataclass

from vision.routing.road_graph import RoadGraph


@dataclass(frozen=True)
class RouteResult:
    node_ids: list[str]
    edge_ids: list[str]
    total_cost: float


class ShortestPathEngine:
    """
    Dijkstra shortest-path engine.

    The graph supplies the network.
    The cost function determines whether we optimize:

        distance
        or
        predicted travel time
    """

    def find_route(
        self,
        graph: RoadGraph,
        source: str,
        target: str,
        cost_mode: str = "distance",
    ) -> RouteResult:

        if source not in graph.nodes:
            raise ValueError(
                f"Source node not found: {source}"
            )

        if target not in graph.nodes:
            raise ValueError(
                f"Target node not found: {target}"
            )

        if cost_mode not in {
            "distance",
            "travel_time",
        }:
            raise ValueError(
                "cost_mode must be 'distance' or 'travel_time'."
            )

        distances = {
            source: 0.0
        }

        previous = {}

        queue = [
            (0.0, source)
        ]

        while queue:
            current_cost, current_node = heapq.heappop(
                queue
            )

            if current_cost > distances.get(
                current_node,
                float("inf"),
            ):
                continue

            if current_node == target:
                break

            for edge in graph.outgoing_edges(
                current_node
            ):
                if cost_mode == "distance":
                    edge_cost = edge.distance_meters
                else:
                    if edge.travel_time_seconds is None:
                        raise ValueError(
                            "Travel-time cost requested, "
                            f"but edge {edge.edge_id} "
                            "has no travel_time_seconds."
                        )

                    edge_cost = (
                        edge.travel_time_seconds
                    )

                new_cost = (
                    current_cost
                    + edge_cost
                )

                if new_cost < distances.get(
                    edge.target,
                    float("inf"),
                ):
                    distances[edge.target] = new_cost

                    previous[edge.target] = (
                        current_node,
                        edge.edge_id,
                    )

                    heapq.heappush(
                        queue,
                        (
                            new_cost,
                            edge.target,
                        ),
                    )

        if target not in distances:
            raise ValueError(
                f"No route exists from {source} "
                f"to {target}."
            )

        node_ids = []
        edge_ids = []

        current = target

        while current != source:
            node_ids.append(current)

            previous_node, edge_id = previous[
                current
            ]

            edge_ids.append(edge_id)
            current = previous_node

        node_ids.append(source)

        node_ids.reverse()
        edge_ids.reverse()

        return RouteResult(
            node_ids=node_ids,
            edge_ids=edge_ids,
            total_cost=distances[target],
        )
