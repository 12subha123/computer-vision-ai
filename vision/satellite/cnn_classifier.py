import numpy as np
import rasterio

from vision.satellite.land_cover import SatelliteLandCoverAdapter
from vision.satellite.sentinel2_scene import Sentinel2Scene


class SatelliteCNNClassifier:
    """
    Connects the unified Computer Vision AI satellite branch
    to the trained Sentinel-2 land-cover CNN.
    """

    def __init__(self):
        self.land_cover = SatelliteLandCoverAdapter()

    def predict_patch(
        self,
        scene: Sentinel2Scene,
        row: int = 0,
        col: int = 0,
        patch_size: int = 64,
    ):
        if patch_size != 64:
            raise ValueError(
                "The trained satellite CNN requires 64x64 patches."
            )

        arrays = []

        datasets = []

        try:
            for band in ("B02", "B03", "B04", "B08"):
                dataset = rasterio.open(
                    scene.band_paths[band]
                )
                datasets.append(dataset)

            reference = datasets[0]

            for dataset in datasets[1:]:
                if (
                    dataset.width != reference.width
                    or dataset.height != reference.height
                    or dataset.transform != reference.transform
                    or dataset.crs != reference.crs
                ):
                    raise ValueError(
                        "Sentinel-2 bands are not spatially aligned."
                    )

            if (
                row < 0
                or col < 0
                or row + patch_size > reference.height
                or col + patch_size > reference.width
            ):
                raise ValueError(
                    "Requested patch is outside the satellite scene."
                )

            for dataset in datasets:
                arrays.append(
                    dataset.read(
                        1,
                        window=rasterio.windows.Window(
                            col,
                            row,
                            patch_size,
                            patch_size,
                        ),
                    )
                )

        finally:
            for dataset in datasets:
                dataset.close()

        patch = np.stack(arrays).astype(
            np.float32,
            copy=False,
        )

        result = self.land_cover.predict_patch(
            patch
        )

        return {
            "scene_id": scene.scene_id,
            "row": row,
            "col": col,
            "patch_size": patch_size,
            "prediction": result,
        }
