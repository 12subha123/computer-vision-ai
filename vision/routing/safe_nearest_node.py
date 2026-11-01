from vision.routing.nearest_node import NearestRoadNodeFinder


class SafeNearestRoadNodeFinder(NearestRoadNodeFinder):
    def find(
        self,
        graph,
        latitude,
        longitude,
        max_distance_meters=2000.0,
    ):
        result = super().find(
            graph=graph,
            latitude=latitude,
            longitude=longitude,
        )

        if result["distance_meters"] > max_distance_meters:
            raise ValueError(
                "Nearest road node is too far from the requested "
                f"location: {result['distance_meters']:.2f} meters. "
                f"Maximum allowed: {max_distance_meters:.2f} meters."
            )

        return result
