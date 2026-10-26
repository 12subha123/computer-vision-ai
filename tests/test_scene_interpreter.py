from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.unified_vision_engine import UnifiedVisionEngine
from vision.scene_interpreter import SceneInterpreter


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

engine = UnifiedVisionEngine()
interpreter = SceneInterpreter()

try:
    results = engine.process_image(IMAGE_PATH)
    summary = engine.get_summary(results)
    scene = interpreter.summarize(results)

    print()
    print("AI Vision Scene Understanding")
    print("-----------------------------")
    print(f"Faces:   {summary['faces']}")
    print(f"Hands:   {summary['hands']}")
    print(f"Poses:   {summary['poses']}")
    print(f"Objects: {summary['objects']}")
    print()
    print("Scene description:")
    print(scene["description"])

finally:
    engine.close()
