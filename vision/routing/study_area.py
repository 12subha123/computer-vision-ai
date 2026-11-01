from dataclasses import dataclass


@dataclass(frozen=True)
class StudyArea:
    """
    Geographic area used for real road-network acquisition.
    """

    name: str
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float

    def __post_init__(self):
        if self.min_latitude >= self.max_latitude:
            raise ValueError(
                "min_latitude must be smaller than max_latitude."
            )

        if self.min_longitude >= self.max_longitude:
            raise ValueError(
                "min_longitude must be smaller than max_longitude."
            )

        if not (
            -90.0 <= self.min_latitude <= 90.0
            and -90.0 <= self.max_latitude <= 90.0
        ):
            raise ValueError(
                "Latitude must be between -90 and 90 degrees."
            )

        if not (
            -180.0 <= self.min_longitude <= 180.0
            and -180.0 <= self.max_longitude <= 180.0
        ):
            raise ValueError(
                "Longitude must be between -180 and 180 degrees."
            )

    @property
    def bounds(self):
        return (
            self.min_latitude,
            self.min_longitude,
            self.max_latitude,
            self.max_longitude,
        )

    def contains(
        self,
        latitude: float,
        longitude: float,
    ) -> bool:
        return (
            self.min_latitude
            <= latitude
            <= self.max_latitude
            and self.min_longitude
            <= longitude
            <= self.max_longitude
        )
