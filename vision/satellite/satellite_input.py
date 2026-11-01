from dataclasses import dataclass
from pathlib import Path


SATELLITE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".jp2",
    ".jp2k",
}


@dataclass(frozen=True)
class SatelliteInput:
    path: Path

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(
                f"Satellite input not found: {self.path}"
            )

        if self.path.suffix.lower() not in SATELLITE_EXTENSIONS:
            raise ValueError(
                f"Unsupported satellite format: {self.path.suffix}"
            )

    @property
    def format(self) -> str:
        return self.path.suffix.lower().lstrip(".")


def is_satellite_input(path: str | Path) -> bool:
    path = Path(path)

    return (
        path.suffix.lower()
        in SATELLITE_EXTENSIONS
    )


def create_satellite_input(path: str | Path) -> SatelliteInput:
    return SatelliteInput(
        path=Path(path)
    )
