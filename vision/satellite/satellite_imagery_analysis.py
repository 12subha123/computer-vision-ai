from pathlib import Path

import rasterio


class SatelliteImageryAnalysisService:
    """
    High-level satellite imagery inspection service.

    This service focuses on understanding the source imagery itself:
    dimensions, coordinate system, spatial bounds, resolution,
    band information, and basic raster statistics.

    It does not modify the imagery or the routing system.
    """

    def analyze(self, imagery_path: str | Path) -> dict:
        imagery_path = Path(imagery_path)

        if not imagery_path.exists():
            raise FileNotFoundError(
                f"Satellite imagery not found: {imagery_path}"
            )

        with rasterio.open(imagery_path) as dataset:
            statistics = []

            for band_index in range(1, dataset.count + 1):
                band = dataset.read(band_index)

                statistics.append(
                    {
                        "band": band_index,
                        "min": float(band.min()),
                        "max": float(band.max()),
                        "mean": float(band.mean()),
                    }
                )

            return {
                "path": str(imagery_path),
                "format": imagery_path.suffix.lower(),
                "width": dataset.width,
                "height": dataset.height,
                "band_count": dataset.count,
                "crs": (
                    str(dataset.crs)
                    if dataset.crs
                    else None
                ),
                "resolution": {
                    "x": float(dataset.res[0]),
                    "y": float(dataset.res[1]),
                },
                "bounds": {
                    "left": float(dataset.bounds.left),
                    "bottom": float(dataset.bounds.bottom),
                    "right": float(dataset.bounds.right),
                    "top": float(dataset.bounds.top),
                },
                "statistics": statistics,
            }
