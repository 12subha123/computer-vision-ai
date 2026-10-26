from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.object_detection import ObjectDetector


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

detector = ObjectDetector(confidence=0.25)
detections = detector.detect(IMAGE_PATH)

print()
print("Object Detection")
print("-----------------")
print(f"Objects detected: {len(detections)}")

for index, detection in enumerate(detections, start=1):
    print(
        f"{index}. "
        f"{detection['label']} "
        f"({detection['confidence'] * 100:.2f}%) "
        f"Box: {detection['box']}"
    )
