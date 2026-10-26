from pathlib import Path

from vision.vision_engine import VisionEngine
from vision.object_detection import ObjectDetector


class UnifiedVisionEngine:
    def __init__(self):
        self.vision_engine = VisionEngine()
        self.object_detector = ObjectDetector(confidence=0.25)

    def process_image(self, image_path):
        import cv2

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        vision_results = self.vision_engine.process(frame)
        object_results = self.object_detector.detect(image_path)

        return {
            "faces": vision_results.get("faces"),
            "hands": vision_results.get("hands"),
            "poses": vision_results.get("poses"),
            "objects": object_results,
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
        return {
            "faces": self._count_faces(results["faces"]),
            "hands": self._count_hands(results["hands"]),
            "poses": self._count_poses(results["poses"]),
            "objects": len(results["objects"]),
        }

    def close(self):
        self.vision_engine.close()
