from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.vision_engine import VisionEngine
from vision.fps_counter import FPSCounter


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
TOTAL_FRAMES = 20

engine = VisionEngine()
fps_counter = FPSCounter()

import cv2

frame = cv2.imread(str(IMAGE_PATH))

if frame is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

start_time = time.perf_counter()

for _ in range(TOTAL_FRAMES):
    results = engine.process(frame)
    fps_counter.update()

elapsed = time.perf_counter() - start_time
average_fps = TOTAL_FRAMES / elapsed if elapsed > 0 else 0.0

engine.close()

print()
print("Vision Engine FPS Test")
print("----------------------")
print(f"Frames processed: {TOTAL_FRAMES}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Average FPS: {average_fps:.2f}")
print(f"Final FPS counter: {fps_counter.get_fps():.2f}")
