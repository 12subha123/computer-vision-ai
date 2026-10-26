from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.object_detection import ObjectDetector


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "object_detection_result.jpg"

detector = ObjectDetector(confidence=0.25)
detections = detector.detect(IMAGE_PATH)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

for detection in detections:
    x1, y1, x2, y2 = map(int, detection["box"])

    label = detection["label"]
    confidence = detection["confidence"]

    text = f"{label}: {confidence * 100:.1f}%"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

cv2.imwrite(str(OUTPUT_PATH), image)

print()
print("Object Detection Visualization")
print("------------------------------")
print(f"Objects detected: {len(detections)}")
print(f"Result saved: {OUTPUT_PATH}")
