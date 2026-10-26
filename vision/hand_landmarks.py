from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandLandmarker:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "hand_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
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
        if not results.hand_landmarks:
            return frame

        height, width = frame.shape[:2]

        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1,
                    )

            for connection in connections:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                x1 = int(start.x * width)
                y1 = int(start.y * height)
                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

        return frame

    def close(self):
        self.landmarker.close()
