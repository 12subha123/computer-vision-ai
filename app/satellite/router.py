from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.satellite.routing_service import SatelliteRoutingService
from vision.routing.geocoder import NominatimGeocoder

router = APIRouter(
    prefix="/satellite",
    tags=["Satellite Intelligence"],
)

routing_service = SatelliteRoutingService()
geocoder = NominatimGeocoder()


class RouteRequest(BaseModel):
    source_latitude: float
    source_longitude: float
    target_latitude: float
    target_longitude: float


class PlaceRouteRequest(BaseModel):
    source: str
    target: str


@router.post("/route")
def calculate_route(request: RouteRequest):
    try:
        return routing_service.route(
            source_latitude=request.source_latitude,
            source_longitude=request.source_longitude,
            target_latitude=request.target_latitude,
            target_longitude=request.target_longitude,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/route-by-place")
def calculate_route_by_place(request: PlaceRouteRequest):
    try:
        source = geocoder.geocode(request.source)
        target = geocoder.geocode(request.target)

        result = routing_service.route(
            source_latitude=source.latitude,
            source_longitude=source.longitude,
            target_latitude=target.latitude,
            target_longitude=target.longitude,
        )

        route = result["route"]

        coordinates = route.get("coordinates", [])

        # Keep the response small enough for the browser UI
        # while preserving the overall route shape for a future map.
        max_points = 500

        if len(coordinates) > max_points:
            step = (len(coordinates) - 1) / (max_points - 1)

            simplified_coordinates = [
                coordinates[round(index * step)]
                for index in range(max_points)
            ]
        else:
            simplified_coordinates = coordinates

        return {
            "source": {
                "query": source.query,
                "display_name": source.display_name,
                "latitude": source.latitude,
                "longitude": source.longitude,
            },
            "target": {
                "query": target.query,
                "display_name": target.display_name,
                "latitude": target.latitude,
                "longitude": target.longitude,
            },
            "route": {
                "distance_meters": route["distance_meters"],
                "distance_km": route["distance_km"],
                "coordinates": simplified_coordinates,
                "coordinate_count": len(simplified_coordinates),
            },
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Route lookup failed: {error}",
        )
