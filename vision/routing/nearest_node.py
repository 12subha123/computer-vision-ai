from math import radians, sin, cos, sqrt, atan2

from vision.routing.road_graph import RoadGraph


EARTH_RADIUS_METERS = 6_371_008.8


def haversine_distance_meters(
    latitude1,
    longitude1,
    latitude2,
    longitude2,
):
    lat1 = radians(latitude1)
    lat2 = radians(latitude2)

    dlat = radians(latitude2 - latitude1)
    dlon = radians(longitude2 - longitude1)

    value = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    angle = 2 * atan2(
        sqrt(value),
        sqrt(1 - value),
    )

    return EARTH_RADIUS_METERS * angle


class NearestRoadNodeFinder:
    def find(
        self,
        graph: RoadGraph,
        latitude: float,
        longitude: float,
    ):
        if not graph.nodes:
            raise ValueError("Road graph contains no nodes.")

        best_node = None
        best_distance = float("inf")

        for node in graph.nodes.values():
            distance = haversine_distance_meters(
                latitude,
                longitude,
                node.latitude,
                node.longitude,
            )

            if distance < best_distance:
                best_distance = distance
                best_node = node

        return {
            "node_id": best_node.node_id,
            "latitude": best_node.latitude,
            "longitude": best_node.longitude,
            "distance_meters": best_distance,
        }
