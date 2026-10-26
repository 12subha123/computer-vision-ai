from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.face_landmarks import FaceLandmarker


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "face_landmarks_result.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    landmarker = FaceLandmarker()
    results = landmarker.detect(image)
    result = landmarker.draw(image, results)
    landmarker.close()

    cv2.imwrite(str(OUTPUT_PATH), result)

    face_count = len(results.face_landmarks)

    print(f"Faces with landmarks: {face_count}")
    print(f"Result saved: {OUTPUT_PATH}")

    if face_count:
        print(f"Landmarks on first face: {len(results.face_landmarks[0])}")


if __name__ == "__main__":
    main()
