from pathlib import Path

import numpy as np
import rasterio


class ChangeDetectionService:
    """
    Detects land-cover changes between two prediction rasters.

    The rasters must represent the same geographic area and
    have matching dimensions and spatial transforms.
    """

    CLASS_NAMES = {
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

    def compare(
        self,
        earlier_prediction: str | Path,
        later_prediction: str | Path,
    ) -> dict:
        earlier_prediction = Path(earlier_prediction)
        later_prediction = Path(later_prediction)

        if not earlier_prediction.exists():
            raise FileNotFoundError(
                f"Earlier prediction not found: "
                f"{earlier_prediction}"
            )

        if not later_prediction.exists():
            raise FileNotFoundError(
                f"Later prediction not found: "
                f"{later_prediction}"
            )

        with rasterio.open(earlier_prediction) as earlier:
            with rasterio.open(later_prediction) as later:
                if earlier.width != later.width:
                    raise ValueError(
                        "Prediction rasters have different widths."
                    )

                if earlier.height != later.height:
                    raise ValueError(
                        "Prediction rasters have different heights."
                    )

                if earlier.transform != later.transform:
                    raise ValueError(
                        "Prediction rasters have different spatial transforms."
                    )

                if earlier.crs != later.crs:
                    raise ValueError(
                        "Prediction rasters have different coordinate systems."
                    )

                before = earlier.read(1)
                after = later.read(1)

                changed = (
                    (before != after)
                    & (before != 0)
                    & (after != 0)
                )

                pixel_area = (
                    abs(earlier.transform.a)
                    * abs(earlier.transform.e)
                )

                transitions = {}

                before_values = before[changed]
                after_values = after[changed]

                for before_class, after_class in zip(
                    before_values,
                    after_values,
                ):
                    key = f"{int(before_class)}->{int(after_class)}"

                    if key not in transitions:
                        transitions[key] = {
                            "from_class_id": int(before_class),
                            "from_class_name": self.CLASS_NAMES.get(
                                int(before_class),
                                "Unknown",
                            ),
                            "to_class_id": int(after_class),
                            "to_class_name": self.CLASS_NAMES.get(
                                int(after_class),
                                "Unknown",
                            ),
                            "pixel_count": 0,
                            "area_square_meters": 0.0,
                            "area_hectares": 0.0,
                            "area_square_kilometers": 0.0,
                        }

                    transitions[key]["pixel_count"] += 1

                for transition in transitions.values():
                    area = (
                        transition["pixel_count"]
                        * pixel_area
                    )

                    transition["area_square_meters"] = area
                    transition["area_hectares"] = area / 10000.0
                    transition["area_square_kilometers"] = (
                        area / 1_000_000.0
                    )

                return {
                    "earlier_prediction": str(
                        earlier_prediction
                    ),
                    "later_prediction": str(
                        later_prediction
                    ),
                    "crs": (
                        str(earlier.crs)
                        if earlier.crs
                        else None
                    ),
                    "pixel_area_square_meters": pixel_area,
                    "changed_pixel_count": int(
                        np.count_nonzero(changed)
                    ),
                    "changed_area_square_meters": (
                        int(np.count_nonzero(changed))
                        * pixel_area
                    ),
                    "changed_area_hectares": (
                        int(np.count_nonzero(changed))
                        * pixel_area
                        / 10000.0
                    ),
                    "changed_area_square_kilometers": (
                        int(np.count_nonzero(changed))
                        * pixel_area
                        / 1_000_000.0
                    ),
                    "transition_count": len(transitions),
                    "transitions": transitions,
                }
