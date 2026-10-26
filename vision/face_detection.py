import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=self._get_model_path()
        )

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def _get_model_path(self):
        from pathlib import Path

        model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "blaze_face_short_range.tflite"

        if not model_path.exists():
            raise FileNotFoundError(
                "Face detection model is missing: "
                f"{model_path}"
            )

        return str(model_path)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.detector.detect(mp_image)

    def draw(self, frame, results):
        if results.detections:
            height, width = frame.shape[:2]

            for detection in results.detections:
                bbox = detection.bounding_box

                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                w = bbox.width
                h = bbox.height

                x2 = min(width, x + w)
                y2 = min(height, y + h)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

        return frame

    def close(self):
        self.detector.close()
