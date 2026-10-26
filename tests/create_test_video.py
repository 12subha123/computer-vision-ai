from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"
VIDEO_PATH = PROJECT_ROOT / "inference" / "vision_test.mp4"

frame = cv2.imread(str(IMAGE_PATH))

if frame is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

height, width = frame.shape[:2]

fps = 10
duration_seconds = 5
total_frames = fps * duration_seconds

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(VIDEO_PATH),
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    raise RuntimeError("Could not create video file.")

for _ in range(total_frames):
    writer.write(frame)

writer.release()

print(f"Video created: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Frames: {total_frames}")
