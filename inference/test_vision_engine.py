from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.vision_engine import VisionEngine


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "inference" / "vision_engine_result.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print(f"ERROR: Could not load image: {IMAGE_PATH}")
        return

    engine = VisionEngine()

    results = engine.process(image)
    summary = engine.get_summary(results)

    result = engine.draw(image, results)

    engine.close()

    cv2.imwrite(str(OUTPUT_PATH), result)

    print("Vision Engine Results")
    print(f"Faces: {summary['faces']}")
    print(f"Hands: {summary['hands']}")
    print(f"Poses: {summary['poses']}")
    print(f"Result saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
