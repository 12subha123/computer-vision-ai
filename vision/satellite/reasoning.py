from typing import Any


class SatelliteReasoningService:
    """
    Converts satellite-analysis results into human-readable reasoning.

    The reasoning layer is deterministic and grounded entirely in the
    measured outputs of the satellite/geospatial pipeline.
    """

    PRIORITY_CLASSES = {
        "Built-up",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Mangroves",
    }

    ENVIRONMENTAL_CLASSES = {
        "Permanent water bodies",
        "Herbaceous wetland",
        "Mangroves",
    }

    def reason_about_area(
        self,
        area_result: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(area_result, dict):
            raise TypeError("area_result must be a dictionary.")

        area_name = str(area_result.get("area_name", "Unknown area"))
        land_cover = area_result.get("land_cover", {})
        geographic = area_result.get("geographic_analysis", {})

        if not isinstance(land_cover, dict):
            raise ValueError("land_cover must be a dictionary.")

        observations = []
        priorities = []
        total_area_m2 = 0.0

        for details in land_cover.values():
            if not isinstance(details, dict):
                continue

            class_name = str(details.get("class_name", "Unknown"))
            area_m2 = float(details.get("area_square_meters", 0.0))
            area_ha = float(
                details.get(
                    "area_hectares",
                    area_m2 / 10000.0,
                )
            )

            if area_m2 <= 0:
                continue

            total_area_m2 += area_m2

            observations.append(
                {
                    "class_name": class_name,
                    "area_square_meters": area_m2,
                    "area_hectares": area_ha,
                }
            )

            if class_name in self.PRIORITY_CLASSES:
                priorities.append(class_name)

        observations.sort(
            key=lambda item: item["area_square_meters"],
            reverse=True,
        )

        dominant = observations[0] if observations else None

        if dominant:
            conclusion = (
                f"{area_name} is dominated by "
                f"{dominant['class_name'].lower()} "
                f"within the classified area."
            )
        else:
            conclusion = (
                f"No classified land-cover area was available for {area_name}."
            )

        if priorities:
            priority_text = (
                "The analysis identifies environmentally or "
                "developmentally important land-cover categories: "
                + ", ".join(sorted(set(priorities)))
                + "."
            )
        else:
            priority_text = (
                "No priority environmental or built-up category "
                "was identified in the available classification."
            )

        return {
            "area_name": area_name,
            "conclusion": conclusion,
            "observations": observations,
            "dominant_class": dominant,
            "priority_categories": sorted(set(priorities)),
            "priority_summary": priority_text,
            "classified_area_square_meters": total_area_m2,
            "classified_area_hectares": total_area_m2 / 10000.0,
            "geographic_context": {
                "crs": geographic.get("crs"),
                "width": geographic.get("width"),
                "height": geographic.get("height"),
            },
        }

    def reason_about_question(
        self,
        area_result: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """
        Answer a user question using only the available satellite
        observations for the analyzed area.
        """
        if not isinstance(area_result, dict):
            raise TypeError("area_result must be a dictionary.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        analysis = self.reason_about_area(area_result)

        normalized_question = " ".join(question.strip().lower().split())
        area_name = analysis["area_name"]
        observations = analysis["observations"]
        dominant = analysis["dominant_class"]
        priorities = analysis["priority_categories"]

        def find_class(class_name: str) -> dict[str, Any] | None:
            target = class_name.lower()
            return next(
                (
                    item
                    for item in observations
                    if item["class_name"].lower() == target
                ),
                None,
            )

        def contains_any(*terms: str) -> bool:
            return any(term in normalized_question for term in terms)

        if not observations:
            answer = (
                f"The available satellite classification does not contain "
                f"enough land-cover observations to answer that question "
                f"for {area_name}."
            )
            basis = "No positive-area land-cover classes were available."

        # ---------------------------------------------------------
        # 1. Dominant / largest class
        # ---------------------------------------------------------
        elif contains_any("dominant", "largest", "most common"):
            if dominant:
                answer = (
                    f"The dominant classified land-cover category in "
                    f"{area_name} is {dominant['class_name']}, covering "
                    f"{dominant['area_hectares']:.2f} hectares."
                )
                basis = (
                    "The answer is based on the largest classified "
                    "land-cover area."
                )
            else:
                answer = (
                    f"No dominant land-cover category could be determined "
                    f"for {area_name}."
                )
                basis = "No classified observations were available."

        # ---------------------------------------------------------
        # 2. Specific land-cover area questions
        # ---------------------------------------------------------
        elif contains_any(
            "cropland",
            "crop land",
            "farmland",
            "farm land",
        ):
            item = find_class("Cropland")

            if item:
                answer = (
                    f"The satellite classification identifies "
                    f"{item['area_hectares']:.2f} hectares of cropland "
                    f"in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Cropland "
                    "land-cover category."
                )
            else:
                answer = (
                    f"No cropland was identified in the available "
                    f"classification for {area_name}."
                )
                basis = (
                    "The Cropland class was not present with positive "
                    "classified area."
                )

        elif contains_any(
            "grassland",
            "grass land",
        ):
            item = find_class("Grassland")

            if item:
                answer = (
                    f"The satellite classification identifies "
                    f"{item['area_hectares']:.2f} hectares of grassland "
                    f"in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Grassland "
                    "land-cover category."
                )
            else:
                answer = (
                    f"No grassland was identified in the available "
                    f"classification for {area_name}."
                )
                basis = (
                    "The Grassland class was not present with positive "
                    "classified area."
                )

        elif contains_any(
            "tree cover",
            "forest",
            "trees",
        ):
            item = find_class("Tree cover")

            if item:
                answer = (
                    f"The satellite classification identifies "
                    f"{item['area_hectares']:.2f} hectares of tree cover "
                    f"in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Tree cover "
                    "land-cover category."
                )
            else:
                answer = (
                    f"No tree cover was identified in the available "
                    f"classification for {area_name}."
                )
                basis = (
                    "The Tree cover class was not present with positive "
                    "classified area."
                )

        # ---------------------------------------------------------
        # 3. Direct water-body questions
        # ---------------------------------------------------------
        elif contains_any(
            "water bodies",
            "water body",
            "permanent water",
            "surface water",
        ):
            item = find_class("Permanent water bodies")

            if item:
                answer = (
                    f"Yes. The satellite classification identifies "
                    f"{item['area_hectares']:.2f} hectares of permanent "
                    f"water bodies in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Permanent "
                    "water bodies category."
                )
            else:
                answer = (
                    f"No permanent water bodies were identified in the "
                    f"available classification for {area_name}."
                )
                basis = (
                    "The Permanent water bodies class was not present "
                    "with positive classified area."
                )

        # ---------------------------------------------------------
        # 4. Wetland questions
        # ---------------------------------------------------------
        elif contains_any(
            "wetland",
            "wetlands",
        ):
            item = find_class("Herbaceous wetland")

            if item:
                answer = (
                    f"Yes. The satellite classification identifies "
                    f"{item['area_hectares']:.2f} hectares of herbaceous "
                    f"wetland in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Herbaceous "
                    "wetland category."
                )
            else:
                answer = (
                    f"No herbaceous wetland was identified in the "
                    f"available classification for {area_name}."
                )
                basis = (
                    "The Herbaceous wetland class was not present with "
                    "positive classified area."
                )

        # ---------------------------------------------------------
        # 5. Environmental-feature questions
        # ---------------------------------------------------------
        elif contains_any(
            "environmental features",
            "environmentally important",
            "environmental importance",
        ):
            environmental = [
                item
                for item in observations
                if item["class_name"] in self.ENVIRONMENTAL_CLASSES
            ]

            if environmental:
                answer = (
                    f"The environmentally important classified features "
                    f"in {area_name} are: "
                    + ", ".join(
                        item["class_name"]
                        for item in environmental
                    )
                    + "."
                )
                basis = (
                    "The answer is based on the classified water, "
                    "wetland, and mangrove categories."
                )
            else:
                answer = (
                    f"No permanent water, herbaceous wetland, or mangrove "
                    f"classification was identified for {area_name}."
                )
                basis = (
                    "Those environmental classes were not present with "
                    "positive classified area."
                )

        # ---------------------------------------------------------
        # 6. Built-up / development questions
        # ---------------------------------------------------------
        elif contains_any(
            "built",
            "urban",
            "development",
            "developed",
        ):
            built_up = find_class("Built-up")

            if built_up:
                answer = (
                    f"The satellite classification identifies "
                    f"{built_up['area_hectares']:.2f} hectares of built-up "
                    f"land in {area_name}."
                )
                basis = (
                    "The answer is based on the classified Built-up "
                    "land-cover category."
                )
            else:
                answer = (
                    f"No built-up area was identified in the available "
                    f"classification for {area_name}."
                )
                basis = (
                    "The Built-up class was not present with positive "
                    "classified area."
                )

        # ---------------------------------------------------------
        # 7. Priority questions
        # ---------------------------------------------------------
        elif contains_any(
            "priority",
            "important categories",
        ):
            if priorities:
                answer = (
                    f"The priority categories identified in {area_name} "
                    "are: "
                    + ", ".join(priorities)
                    + "."
                )
                basis = (
                    "Priority categories are determined from the existing "
                    "satellite reasoning rules."
                )
            else:
                answer = (
                    f"No predefined priority category was identified "
                    f"in {area_name}."
                )
                basis = (
                    "None of the configured priority classes had positive "
                    "classified area."
                )

        # ---------------------------------------------------------
        # 8. Total classified area
        # ---------------------------------------------------------
        elif contains_any(
            "total area",
            "classified area",
            "area of this region",
            "size of this area",
        ):
            answer = (
                f"The classified satellite area for {area_name} is "
                f"{analysis['classified_area_hectares']:.2f} hectares."
            )
            basis = (
                "The value is calculated from the positive-area "
                "land-cover classifications."
            )

        # ---------------------------------------------------------
        # 9. Safe fallback
        # ---------------------------------------------------------
        else:
            answer = (
                f"For {area_name}, the satellite analysis finds "
                f"{len(observations)} classified land-cover categories. "
                f"The dominant category is "
                f"{dominant['class_name'] if dominant else 'not available'}."
            )
            basis = (
                "The question did not match a specialized reasoning "
                "category, so the answer summarizes the available "
                "satellite observations instead of inventing unsupported facts."
            )

        return {
            "question": question.strip(),
            "area_name": area_name,
            "answer": answer,
            "basis": basis,
            "observations": observations,
            "dominant_class": dominant,
            "priority_categories": priorities,
        }
