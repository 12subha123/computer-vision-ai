from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class GeocodedPlace:
    query: str
    display_name: str
    latitude: float
    longitude: float


class NominatimGeocoder:
    def __init__(
        self,
        base_url: str = "https://nominatim.openstreetmap.org/search",
        user_agent: str = "computer-vision-ai-satellite-intelligence/1.0",
    ):
        self.base_url = base_url
        self.user_agent = user_agent

    def geocode(
        self,
        query: str,
        countrycodes: Optional[str] = "in",
    ) -> GeocodedPlace:
        query = query.strip()

        if not query:
            raise ValueError("Place query cannot be empty.")

        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }

        if countrycodes:
            params["countrycodes"] = countrycodes

        response = requests.get(
            self.base_url,
            params=params,
            headers={
                "User-Agent": self.user_agent,
            },
            timeout=20,
        )

        response.raise_for_status()

        results = response.json()

        if not results:
            raise ValueError(
                f"No place found for query: {query}"
            )

        result = results[0]

        return GeocodedPlace(
            query=query,
            display_name=result["display_name"],
            latitude=float(result["lat"]),
            longitude=float(result["lon"]),
        )
