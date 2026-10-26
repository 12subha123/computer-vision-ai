from pathlib import Path
import cv2
from vision.unified_vision_engine import UnifiedVisionEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VideoVisionProcessor:
    def __init__(self):
        self.engine = UnifiedVisionEngine()

    def process(self, video_path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = 0
        results = []

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                frame_results = self.engine.vision_engine.process(frame)
                summary = self.engine.vision_engine.get_summary(frame_results)

                results.append({
                    "frame": frame_count,
                    "summary": summary,
                })

                frame_count += 1

        finally:
            capture.release()

        return {
            "video": str(video_path),
            "frames_processed": frame_count,
            "results": results,
        }

    def close(self):
        self.engine.close()
