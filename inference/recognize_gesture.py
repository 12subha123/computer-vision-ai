from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.hand_landmarks import HandLandmarker
from vision.gesture_recognition import GestureRecognizer


IMAGE_PATH = PROJECT_ROOT / "inference" / "hand_test.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    landmarker = HandLandmarker()
    recognizer = GestureRecognizer()

    results = landmarker.detect(image)

    hand_count = len(results.hand_landmarks)

    print(f"Hands detected: {hand_count}")

    for index, landmarks in enumerate(results.hand_landmarks):
        handedness = "Unknown"

        if results.handedness and index < len(results.handedness):
            handedness = results.handedness[index][0].category_name

        gesture = recognizer.recognize(
            landmarks,
            handedness,
        )

        print(
            f"Hand {index + 1}: "
            f"{handedness} - {gesture}"
        )

    landmarker.close()


if __name__ == "__main__":
    main()
