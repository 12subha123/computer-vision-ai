from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarker:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "face_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.landmarker.detect(mp_image)

    def draw(self, frame, results):
        if not results.face_landmarks:
            return frame

        height, width = frame.shape[:2]

        for face_landmarks in results.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1,
                    )

        return frame

    def close(self):
        self.landmarker.close()
