from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.hand_landmarks import HandLandmarker


IMAGE_PATH = PROJECT_ROOT / "inference" / "hand_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "hand_landmarks_result.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    landmarker = HandLandmarker()
    results = landmarker.detect(image)
    result = landmarker.draw(image, results)
    landmarker.close()

    cv2.imwrite(str(OUTPUT_PATH), result)

    hand_count = len(results.hand_landmarks)

    print(f"Hands detected: {hand_count}")

    if hand_count:
        print(f"Landmarks on first hand: {len(results.hand_landmarks[0])}")

    print(f"Result saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
