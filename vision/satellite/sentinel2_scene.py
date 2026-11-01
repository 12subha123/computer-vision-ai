from dataclasses import dataclass
from pathlib import Path


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")


@dataclass(frozen=True)
class Sentinel2Scene:
    """
    A complete Sentinel-2 four-band scene for the trained
    land-cover CNN.
    """

    b02: Path
    b03: Path
    b04: Path
    b08: Path

    def __post_init__(self):
        for band_name, path in self.band_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{band_name} file not found: {path}"
                )

            if path.suffix.lower() not in {".jp2", ".tif", ".tiff"}:
                raise ValueError(
                    f"Unsupported {band_name} format: {path.suffix}"
                )

    @property
    def band_paths(self):
        return {
            "B02": self.b02,
            "B03": self.b03,
            "B04": self.b04,
            "B08": self.b08,
        }

    @property
    def scene_id(self) -> str:
        return self.b02.stem.replace("_B02_10m", "")


def create_sentinel2_scene(
    b02: str | Path,
    b03: str | Path,
    b04: str | Path,
    b08: str | Path,
) -> Sentinel2Scene:
    return Sentinel2Scene(
        b02=Path(b02),
        b03=Path(b03),
        b04=Path(b04),
        b08=Path(b08),
    )
