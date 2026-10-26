from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.unified_vision_engine import UnifiedVisionEngine
from vision.vision_ai import VisionAI


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

engine = UnifiedVisionEngine()
vision_ai = VisionAI()

try:
    results = engine.process_image(IMAGE_PATH)
    understanding = vision_ai.understand(IMAGE_PATH, results)

    print()
    print("Vision AI")
    print("---------")
    print("Scene description:")
    print(understanding["scene_description"])
    print()
    print("Image caption:")
    print(understanding["caption"])
    print()
    print(f"Object count: {understanding['object_count']}")

finally:
    engine.close()
