from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.satellite.routing_service import SatelliteRoutingService
from app.satellite.satellite_intelligence_service import SatelliteIntelligenceService
from vision.routing.geocoder import NominatimGeocoder

router = APIRouter(
    prefix="/satellite",
    tags=["Satellite Intelligence"],
)

routing_service = SatelliteRoutingService()
intelligence_service = SatelliteIntelligenceService()
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
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from vision.satellite.area_exploration_service import AreaExplorationService

exploration_service = AreaExplorationService()


class ExplorePredictionRequest(BaseModel):
    area_name: str
    center_latitude: float
    center_longitude: float
    prediction_path: str


@router.post("/explore-prediction")
def explore_prediction(request: ExplorePredictionRequest):
    try:
        result = exploration_service.explore_prediction(
            area_name=request.area_name,
            center_latitude=request.center_latitude,
            center_longitude=request.center_longitude,
            prediction_path=request.prediction_path,
        )

        return result.to_dict()

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Area exploration failed: {error}",
        )

from vision.satellite.satellite_imagery_analysis import (
    SatelliteImageryAnalysisService,
)

imagery_analysis_service = SatelliteImageryAnalysisService()


class ImageryAnalysisRequest(BaseModel):
    imagery_path: str


@router.post("/analyze-imagery")
def analyze_imagery(request: ImageryAnalysisRequest):
    try:
        return imagery_analysis_service.analyze(
            request.imagery_path
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Satellite imagery analysis failed: {error}",
        )

from vision.satellite.land_cover_analysis import (
    LandCoverAnalysisService,
)

land_cover_service = LandCoverAnalysisService()


class LandCoverAnalysisRequest(BaseModel):
    prediction_path: str


@router.post("/analyze-land-cover")
def analyze_land_cover(request: LandCoverAnalysisRequest):
    try:
        return land_cover_service.analyze(
            request.prediction_path
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Land-cover analysis failed: {error}",
        )
from vision.satellite.geometric_analysis import SatelliteGeometricAnalyzer

geographic_analyzer = SatelliteGeometricAnalyzer()


class GeographicAnalysisRequest(BaseModel):
    prediction_path: str
    class_id: int


class ChangeDetectionRequest(BaseModel):
    earlier_prediction_path: str
    later_prediction_path: str


from vision.satellite.change_detection import ChangeDetectionService

change_detection_service = ChangeDetectionService()


from vision.satellite.optimization import (
    RouteCandidate,
    RouteOptimizationService,
)

route_optimization_service = RouteOptimizationService()


class RouteOptimizationRequest(BaseModel):
    candidates: list[dict]
    objective: str = "shortest"


@router.post("/optimize-route")
def optimize_route(request: RouteOptimizationRequest):
    try:
        candidates = [
            RouteCandidate(
                route_id=str(item["route_id"]),
                distance_km=float(item["distance_km"]),
                travel_time_minutes=(
                    float(item["travel_time_minutes"])
                    if item.get("travel_time_minutes") is not None
                    else None
                ),
                metadata=item.get("metadata", {}),
            )
            for item in request.candidates
        ]

        return route_optimization_service.optimize(
            candidates=candidates,
            objective=request.objective,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Missing route field: {error}",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Route optimization failed: {error}",
        )


@router.post("/change-detection")
def detect_change(request: ChangeDetectionRequest):
    try:
        return change_detection_service.compare(
            earlier_prediction=request.earlier_prediction_path,
            later_prediction=request.later_prediction_path,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Change detection failed: {error}",
        )


@router.post("/analyze-geography")
def analyze_geography(request: GeographicAnalysisRequest):
    try:
        regions = geographic_analyzer.analyze_prediction_raster(
            prediction_path=request.prediction_path,
            class_id=request.class_id,
        )

        class_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
        }

        return {
            "prediction_path": request.prediction_path,
            "class_id": request.class_id,
            "class_name": class_names.get(
                request.class_id,
                f"Class {request.class_id}",
            ),
            "region_count": len(regions),
            "regions": [
                {
                    "region_id": index + 1,
                    "area_pixels": region.area_pixels,
                    "area_square_meters": region.area_square_meters,
                    "area_hectares": region.area_square_meters / 10000.0,
                    "geometry": region.geometry.__geo_interface__,
                }
                for index, region in enumerate(regions)
            ],
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Geographic analysis failed: {error}",
        )
from pathlib import Path
import shutil
import tempfile

from fastapi import File, Form, UploadFile

@router.post("/user/analyze-imagery")
async def user_analyze_imagery(
    imagery: UploadFile = File(...),
):
    suffix = Path(imagery.filename or "").suffix or ".tif"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(imagery.file, buffer)

        return imagery_analysis_service.analyze(str(temp_path))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Satellite imagery analysis failed: {error}",
        )

    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/user/analyze-land-cover")
async def user_analyze_land_cover(
    prediction: UploadFile = File(...),
):
    suffix = Path(prediction.filename or "").suffix or ".tif"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(prediction.file, buffer)

        return land_cover_service.analyze(str(temp_path))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Land-cover analysis failed: {error}",
        )

    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/user/change-detection")
async def user_change_detection(
    earlier: UploadFile = File(...),
    later: UploadFile = File(...),
):
    earlier_suffix = Path(earlier.filename or "").suffix or ".tif"
    later_suffix = Path(later.filename or "").suffix or ".tif"

    earlier_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=earlier_suffix,
    )
    later_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=later_suffix,
    )

    earlier_path = Path(earlier_temp.name)
    later_path = Path(later_temp.name)

    earlier_temp.close()
    later_temp.close()

    try:
        with earlier_path.open("wb") as buffer:
            shutil.copyfileobj(earlier.file, buffer)

        with later_path.open("wb") as buffer:
            shutil.copyfileobj(later.file, buffer)

        return change_detection_service.compare(
            earlier_prediction=str(earlier_path),
            later_prediction=str(later_path),
        )

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Change detection failed: {error}",
        )

    finally:
        earlier_path.unlink(missing_ok=True)
        later_path.unlink(missing_ok=True)


@router.post("/user/analyze-land-cover-real")
def user_analyze_land_cover_real():
    try:
        if not USER_SATELLITE_PREDICTION.exists():
            raise FileNotFoundError(
                f"Satellite prediction raster not found: {USER_SATELLITE_PREDICTION}"
            )

        result = land_cover_service.analyze(
            str(USER_SATELLITE_PREDICTION)
        )

        if hasattr(result, "to_dict"):
            return result.to_dict()

        return result

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Land-cover analysis failed: {error}",
        )

@router.post("/user/analyze-geography-real")
def user_analyze_geography_real(
    class_id: int = 10,
):
    try:
        if not USER_SATELLITE_PREDICTION.exists():
            raise FileNotFoundError(
                f"Satellite prediction raster not found: {USER_SATELLITE_PREDICTION}"
            )

        regions = geographic_analyzer.analyze_prediction_raster(
            prediction_path=str(USER_SATELLITE_PREDICTION),
            class_id=class_id,
        )

        class_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
        }

        return {
            "class_id": class_id,
            "class_name": class_names.get(
                class_id,
                f"Class {class_id}",
            ),
            "region_count": len(regions),
            "regions": [
                {
                    "region_id": index + 1,
                    "area_pixels": region.area_pixels,
                    "area_square_meters": region.area_square_meters,
                    "area_hectares": region.area_square_meters / 10000.0,
                    "geometry": region.geometry.__geo_interface__,
                }
                for index, region in enumerate(regions)
            ],
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Geographic analysis failed: {error}",
        )

@router.post("/user/analyze-geography")
async def user_analyze_geography(
    prediction: UploadFile = File(...),
    class_id: int = Form(...),
):
    suffix = Path(prediction.filename or "").suffix or ".tif"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(prediction.file, buffer)

        regions = geographic_analyzer.analyze_prediction_raster(
            prediction_path=str(temp_path),
            class_id=class_id,
        )

        class_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
        }

        return {
            "class_id": class_id,
            "class_name": class_names.get(
                class_id,
                f"Class {class_id}",
            ),
            "region_count": len(regions),
            "regions": [
                {
                    "region_id": index + 1,
                    "area_pixels": region.area_pixels,
                    "area_square_meters": region.area_square_meters,
                    "area_hectares": region.area_square_meters / 10000.0,
                    "geometry": region.geometry.__geo_interface__,
                }
                for index, region in enumerate(regions)
            ],
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Geographic analysis failed: {error}",
        )

    finally:
        temp_path.unlink(missing_ok=True)
from fastapi import HTTPException

USER_SATELLITE_PREDICTION = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "satellite_test"
    / "T45QWG_20211225T044209_land_cover.tif"
)


@router.post("/user/explore-area")
def user_explore_area(request: PlaceRouteRequest):
    try:
        place = geocoder.geocode(request.source)

        latitude = float(place.latitude)
        longitude = float(place.longitude)

        if not (
            23.4179763362762 <= latitude <= 24.413427907095365
            and 86.99980272924681 <= longitude <= 88.08274340663948
        ):
            raise ValueError(
                "This location is outside the currently available "
                "satellite coverage."
            )

        result = exploration_service.explore_prediction(
            area_name=place.display_name,
            center_latitude=latitude,
            center_longitude=longitude,
            prediction_path=str(USER_SATELLITE_PREDICTION),
        )

        return result.to_dict()

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not explore this area: {error}",
        )



class AreaQuestionRequest(BaseModel):
    source: str
    question: str


@router.post("/user/ask-area")
def user_ask_area(request: AreaQuestionRequest):
    try:
        question = request.question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        place = geocoder.geocode(request.source)

        latitude = float(place.latitude)
        longitude = float(place.longitude)

        if not (
            23.4179763362762 <= latitude <= 24.413427907095365
            and 86.99980272924681 <= longitude <= 88.08274340663948
        ):
            raise ValueError(
                "This location is outside the currently available "
                "satellite coverage."
            )

        area_result = intelligence_service.explore_area(
            area_name=place.display_name,
            center_latitude=latitude,
            center_longitude=longitude,
            prediction_path=str(USER_SATELLITE_PREDICTION),
        )

        answer = intelligence_service.ask_about_area(
            area_result=area_result,
            question=question,
        )

        return {
            "area": area_result,
            "question": question,
            "answer": answer,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "display_name": place.display_name,
            },
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not answer the satellite question: {error}",
        )


class OptimizeRouteRequest(BaseModel):
    source_latitude: float
    source_longitude: float
    target_latitude: float
    target_longitude: float
    objective: str = "shortest"
    max_routes: int = 3
    minimum_difference_edges: int = 2


@router.post("/optimize-route-real")
def optimize_route_real(request: OptimizeRouteRequest):
    try:
        return routing_service.optimize_route(
            source_latitude=request.source_latitude,
            source_longitude=request.source_longitude,
            target_latitude=request.target_latitude,
            target_longitude=request.target_longitude,
            objective=request.objective,
            max_routes=request.max_routes,
            minimum_difference_edges=request.minimum_difference_edges,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Real route optimization failed: {error}",
        )

