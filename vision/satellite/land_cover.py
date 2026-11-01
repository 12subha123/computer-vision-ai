from pathlib import Path
import sys

SATELLITE_PROJECT_ROOT = Path(r"D:\satellite-intelligence")

if not SATELLITE_PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Satellite Intelligence project not found: {SATELLITE_PROJECT_ROOT}"
    )

if str(SATELLITE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SATELLITE_PROJECT_ROOT))

from backend.vision.real_satellite_inference import (
    RealSatelliteLandCoverInference,
)


class SatelliteLandCoverAdapter:
    """
    Adapter connecting the Computer Vision AI satellite branch
    to the trained Satellite Intelligence Sentinel-2 land-cover model.
    """

    def __init__(
        self,
        checkpoint_path=None,
        statistics_path=None,
    ):
        if checkpoint_path is None:
            checkpoint_path = (
                SATELLITE_PROJECT_ROOT
                / "models"
                / "land_cover_real"
                / "best_model.pt"
            )

        if statistics_path is None:
            statistics_path = (
                SATELLITE_PROJECT_ROOT
                / "data"
                / "real"
                / "sentinel2"
                / "processed"
                / "training_patches"
                / "training_band_statistics.json"
            )

        self.inference = RealSatelliteLandCoverInference(
            checkpoint_path=checkpoint_path,
            statistics_path=statistics_path,
            device="cpu",
        )

    def predict_patch(self, patch):
        """
        Predict land cover for one 4-band 64x64 Sentinel-2 patch.
        """
        return self.inference.predict_patch(patch)

    def predict_raster(
        self,
        band_paths,
        output_path,
        patch_size=64,
        stride=64,
    ):
        """
        Run the trained Sentinel-2 model over four-band satellite imagery.
        """
        return self.inference.predict_raster(
            band_paths=band_paths,
            output_path=output_path,
            patch_size=patch_size,
            stride=stride,
        )
