from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.unified_vision_engine import UnifiedVisionEngine


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

engine = UnifiedVisionEngine()

try:
    results = engine.process_image(IMAGE_PATH)
    summary = engine.get_summary(results)

    print()
    print("Unified Vision Engine")
    print("---------------------")
    print(f"Faces:   {summary['faces']}")
    print(f"Hands:   {summary['hands']}")
    print(f"Poses:   {summary['poses']}")
    print(f"Objects: {summary['objects']}")

    if results["objects"]:
        print()
        print("Detected objects:")

        for obj in results["objects"]:
            print(
                f"- {obj['label']}: "
                f"{obj['confidence'] * 100:.2f}%"
            )

finally:
    engine.close()
