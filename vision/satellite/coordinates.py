from dataclasses import dataclass

from pyproj import Transformer


@dataclass(frozen=True)
class GeographicCoordinate:
    latitude: float
    longitude: float


class CoordinateTransformer:
    """
    Converts projected satellite coordinates into
    geographic latitude/longitude coordinates.
    """

    def __init__(self, source_crs, target_crs="EPSG:4326"):
        self.source_crs = source_crs
        self.target_crs = target_crs

        self.transformer = Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
        )

    def transform(
        self,
        x: float,
        y: float,
    ) -> GeographicCoordinate:

        longitude, latitude = self.transformer.transform(
            x,
            y,
        )

        return GeographicCoordinate(
            latitude=float(latitude),
            longitude=float(longitude),
        )
