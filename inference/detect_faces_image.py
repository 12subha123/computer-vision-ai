from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.face_detection import FaceDetector


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "face_detection_result.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    detector = FaceDetector()
    results = detector.detect(image)
    result = detector.draw(image, results)
    detector.close()

    cv2.imwrite(str(OUTPUT_PATH), result)

    count = len(results.detections) if results.detections else 0

    print(f"Faces detected: {count}")
    print(f"Result saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
