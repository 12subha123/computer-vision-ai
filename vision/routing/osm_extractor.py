from pathlib import Path
import json
from math import radians, sin, cos, sqrt, atan2

import osmium

from vision.routing.study_area import StudyArea


EARTH_RADIUS_METERS = 6_371_008.8

ROAD_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}


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


class OSMRoadHandler(osmium.SimpleHandler):
    """
    Extracts real OSM road segments from a PBF.

    PyOsmium provides node locations to ways through
    the locations=True processing mode.
    """

    def __init__(self, study_area):
        super().__init__()

        self.study_area = study_area

        self.nodes = {}
        self.edges = {}

        self.ways_seen = 0
        self.roads_seen = 0
        self.segments_seen = 0

    def way(self, way):

        highway = way.tags.get("highway")

        if highway not in ROAD_TYPES:
            return

        self.ways_seen += 1

        name = way.tags.get("name")
        oneway = way.tags.get("oneway")

        nodes = list(way.nodes)

        if len(nodes) < 2:
            return

        self.roads_seen += 1

        for index in range(len(nodes) - 1):

            first = nodes[index]
            second = nodes[index + 1]

            if not first.location.valid():
                continue

            if not second.location.valid():
                continue

            lat1 = float(first.location.lat)
            lon1 = float(first.location.lon)

            lat2 = float(second.location.lat)
            lon2 = float(second.location.lon)

            first_inside = self.study_area.contains(
                lat1,
                lon1,
            )

            second_inside = self.study_area.contains(
                lat2,
                lon2,
            )

            if not first_inside and not second_inside:
                continue

            source_id = str(first.ref)
            target_id = str(second.ref)

            self.nodes[source_id] = {
                "node_id": source_id,
                "latitude": lat1,
                "longitude": lon1,
            }

            self.nodes[target_id] = {
                "node_id": target_id,
                "latitude": lat2,
                "longitude": lon2,
            }

            distance = haversine_distance_meters(
                lat1,
                lon1,
                lat2,
                lon2,
            )

            edge_id = (
                f"way_{way.id}_"
                f"segment_{index}"
            )

            self.edges[edge_id] = {
                "edge_id": edge_id,
                "source": source_id,
                "target": target_id,
                "distance_meters": distance,
                "properties": {
                    "osm_way_id": int(way.id),
                    "highway": highway,
                    "name": name,
                    "oneway": oneway,
                },
            }

            self.segments_seen += 1

            if oneway not in {"yes", "true", "1"}:
                reverse_edge_id = (
                    f"way_{way.id}_"
                    f"segment_{index}_reverse"
                )

                self.edges[reverse_edge_id] = {
                    "edge_id": reverse_edge_id,
                    "source": target_id,
                    "target": source_id,
                    "distance_meters": distance,
                    "properties": {
                        "osm_way_id": int(way.id),
                        "highway": highway,
                        "name": name,
                        "oneway": oneway,
                    },
                }


class OSMRoadExtractor:
    """
    Converts a real OSM PBF into road-network records.
    """

    def extract(
        self,
        pbf_path,
        study_area,
        output_path,
    ):

        pbf_path = Path(pbf_path)
        output_path = Path(output_path)

        if not pbf_path.exists():
            raise FileNotFoundError(
                f"OSM PBF not found: {pbf_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handler = OSMRoadHandler(
            study_area
        )

        handler.apply_file(
            str(pbf_path),
            locations=True,
        )

        result = {
            "study_area": {
                "name": study_area.name,
                "bounds": study_area.bounds,
            },
            "nodes": list(
                handler.nodes.values()
            ),
            "edges": list(
                handler.edges.values()
            ),
            "statistics": {
                "ways_seen": handler.ways_seen,
                "roads_seen": handler.roads_seen,
                "segments_seen": handler.segments_seen,
                "nodes": len(handler.nodes),
                "edges": len(handler.edges),
            },
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        return output_path
