from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.pose_landmarks import PoseLandmarker


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "pose_landmarks_result.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    landmarker = PoseLandmarker()
    results = landmarker.detect(image)
    result = landmarker.draw(image, results)
    landmarker.close()

    cv2.imwrite(str(OUTPUT_PATH), result)

    pose_count = len(results.pose_landmarks)

    print(f"Poses detected: {pose_count}")

    if pose_count:
        print(
            f"Landmarks on first pose: "
            f"{len(results.pose_landmarks[0])}"
        )

    print(f"Result saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
