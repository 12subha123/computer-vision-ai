from pathlib import Path
import json
import time
import requests

from vision.routing.study_area import StudyArea


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class OSMRoadDownloader:
    """
    Downloads real OpenStreetMap road data in small geographic tiles.

    Large bounding-box queries are split into smaller requests to
    reduce Overpass timeout risk.
    """

    def __init__(
        self,
        output_directory: str | Path,
        endpoint: str = OVERPASS_URL,
    ):
        self.output_directory = Path(output_directory)
        self.endpoint = endpoint

    def build_query(
        self,
        study_area: StudyArea,
    ) -> str:

        return f"""
[out:json][timeout:120];
way["highway"](
    {study_area.min_latitude},
    {study_area.min_longitude},
    {study_area.max_latitude},
    {study_area.max_longitude}
);
out body;
>;
out skel qt;
""".strip()

    def split_area(
        self,
        study_area: StudyArea,
        rows: int = 2,
        columns: int = 2,
    ):
        if rows < 1 or columns < 1:
            raise ValueError(
                "rows and columns must be at least 1."
            )

        lat_step = (
            study_area.max_latitude
            - study_area.min_latitude
        ) / rows

        lon_step = (
            study_area.max_longitude
            - study_area.min_longitude
        ) / columns

        tiles = []

        for row in range(rows):
            for column in range(columns):
                min_lat = (
                    study_area.min_latitude
                    + row * lat_step
                )

                max_lat = (
                    study_area.min_latitude
                    + (row + 1) * lat_step
                )

                min_lon = (
                    study_area.min_longitude
                    + column * lon_step
                )

                max_lon = (
                    study_area.min_longitude
                    + (column + 1) * lon_step
                )

                tiles.append(
                    StudyArea(
                        name=(
                            f"{study_area.name} "
                            f"tile_{row}_{column}"
                        ),
                        min_latitude=min_lat,
                        min_longitude=min_lon,
                        max_latitude=max_lat,
                        max_longitude=max_lon,
                    )
                )

        return tiles

    def download_tile(
        self,
        study_area: StudyArea,
        tile_index: int,
    ) -> Path:

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / f"roads_tile_{tile_index:02d}.json"
        )

        if output_path.exists():
            print(
                f"Skipping existing tile: "
                f"{output_path.name}"
            )
            return output_path

        query = self.build_query(study_area)

        print()
        print(
            f"Downloading tile {tile_index}: "
            f"{study_area.bounds}"
        )

        response = requests.post(
            self.endpoint,
            data={"data": query},
            headers={
                "User-Agent": "SatelliteIntelligence/1.0"
            },
            timeout=180,
        )

        response.raise_for_status()

        data = response.json()

        if "elements" not in data:
            raise ValueError(
                "OSM response does not contain elements."
            )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

        print(
            f"Saved: {output_path}"
        )

        return output_path
