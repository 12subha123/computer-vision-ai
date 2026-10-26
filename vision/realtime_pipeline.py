from pathlib import Path
import cv2

from vision.vision_engine import VisionEngine
from vision.fps_counter import FPSCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealtimeVisionPipeline:
    def __init__(self, source):
        self.source = source
        self.engine = VisionEngine()
        self.fps_counter = FPSCounter()

    def run(self):
        cap = cv2.VideoCapture(str(self.source))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                results = self.engine.process(frame)
                output = self.engine.draw(frame, results)

                fps = self.fps_counter.update()
                summary = self.engine.get_summary(results)

                cv2.putText(
                    output,
                    f"FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    output,
                    (
                        f"Faces: {summary['faces']} | "
                        f"Hands: {summary['hands']} | "
                        f"Poses: {summary['poses']}"
                    ),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Computer Vision AI", output)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.engine.close()


if __name__ == "__main__":
    video_path = PROJECT_ROOT / "inference" / "vision_test.mp4"

    pipeline = RealtimeVisionPipeline(video_path)
    pipeline.run()
