from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingCoverage:
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float

    @classmethod
    def from_points(
        cls,
        latitude1: float,
        longitude1: float,
        latitude2: float,
        longitude2: float,
        margin_degrees: float = 0.02,
    ):
        if margin_degrees <= 0:
            raise ValueError(
                "margin_degrees must be greater than zero."
            )

        min_latitude = min(
            latitude1,
            latitude2,
        ) - margin_degrees

        max_latitude = max(
            latitude1,
            latitude2,
        ) + margin_degrees

        min_longitude = min(
            longitude1,
            longitude2,
        ) - margin_degrees

        max_longitude = max(
            longitude1,
            longitude2,
        ) + margin_degrees

        if min_latitude < -90 or max_latitude > 90:
            raise ValueError(
                "Calculated latitude coverage is outside valid bounds."
            )

        if min_longitude < -180 or max_longitude > 180:
            raise ValueError(
                "Calculated longitude coverage is outside valid bounds."
            )

        return cls(
            min_latitude=min_latitude,
            min_longitude=min_longitude,
            max_latitude=max_latitude,
            max_longitude=max_longitude,
        )

    @property
    def bounds(self):
        return (
            self.min_latitude,
            self.min_longitude,
            self.max_latitude,
            self.max_longitude,
        )
