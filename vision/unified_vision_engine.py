from pathlib import Path

from vision.vision_engine import VisionEngine
from vision.object_detection import ObjectDetector
from vision.satellite.satellite_input import is_satellite_input
from vision.satellite.land_cover import SatelliteLandCoverAdapter


class UnifiedVisionEngine:
    def __init__(self):
        self.vision_engine = VisionEngine()
        self.object_detector = ObjectDetector(confidence=0.25)

        self.satellite_land_cover = SatelliteLandCoverAdapter()

    def process_image(self, image_path):
        image_path = Path(image_path)

        if is_satellite_input(image_path):
            return self.process_satellite(image_path)

        import cv2

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        vision_results = self.vision_engine.process(frame)
        object_results = self.object_detector.detect(image_path)

        return {
            "input_type": "ordinary_image",
            "faces": vision_results.get("faces"),
            "hands": vision_results.get("hands"),
            "poses": vision_results.get("poses"),
            "objects": object_results,
        }

    def process_satellite(self, satellite_path):
        satellite_path = Path(satellite_path)

        if not satellite_path.exists():
            raise FileNotFoundError(
                f"Satellite input not found: {satellite_path}"
            )

        return {
            "input_type": "satellite",
            "path": str(satellite_path),
            "status": "satellite_input_detected",
        }

    def _count_faces(self, value):
        if value is None:
            return 0

        if hasattr(value, "face_landmarks"):
            return len(value.face_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def _count_hands(self, value):
        if value is None:
            return 0

        if hasattr(value, "hand_landmarks"):
            return len(value.hand_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def _count_poses(self, value):
        if value is None:
            return 0

        if hasattr(value, "pose_landmarks"):
            return len(value.pose_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def get_summary(self, results):
        if results.get("input_type") == "satellite":
            return {
                "input_type": "satellite",
                "path": results["path"],
                "status": results["status"],
            }

        return {
            "input_type": "ordinary_image",
            "faces": self._count_faces(results["faces"]),
            "hands": self._count_hands(results["hands"]),
            "poses": self._count_poses(results["poses"]),
            "objects": len(results["objects"]),
        }

    def close(self):
        self.vision_engine.close()
