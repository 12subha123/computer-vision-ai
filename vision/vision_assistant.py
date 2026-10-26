from pathlib import Path
import uuid

import cv2

from vision.unified_vision_engine import UnifiedVisionEngine
from vision.vision_ai import VisionAI
from vision.resnet_classifier import ResNet18Classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DIR = PROJECT_ROOT / "app" / "static" / "annotated"

ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class VisionAssistant:
    def __init__(self):
        self.engine = UnifiedVisionEngine()
        self.vision_ai = VisionAI()
        self.classifier = ResNet18Classifier()

    def _create_annotated_image(self, image_path, results):
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        # UnifiedVisionEngine uses plural keys.
        # VisionEngine.draw expects singular keys.
        draw_results = {
            "face": results["faces"],
            "hand": results["hands"],
            "pose": results["poses"],
        }

        output = self.engine.vision_engine.draw(
            frame,
            draw_results,
        )

        # Draw YOLO object detections.
        for obj in results["objects"]:
            x1, y1, x2, y2 = [
                int(value)
                for value in obj["box"]
            ]

            label = (
                f'{obj["label"]} '
                f'{obj["confidence"] * 100:.1f}%'
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        filename = f"vision_{uuid.uuid4().hex}.jpg"
        output_path = ANNOTATED_DIR / filename

        success = cv2.imwrite(
            str(output_path),
            output,
        )

        if not success:
            raise RuntimeError(
                "Failed to save annotated image."
            )

        return filename

    def analyze(self, image_path, question=None):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.engine.process_image(
            image_path
        )

        summary = self.engine.get_summary(
            results
        )

        understanding = self.vision_ai.understand(
            image_path,
            results,
        )

        classification = self.classifier.predict(
            image_path,
            top_k=5,
        )

        annotated_filename = (
            self._create_annotated_image(
                image_path,
                results,
            )
        )

        response = {
            "summary": summary,
            "scene_description": (
                understanding["scene_description"]
            ),
            "caption": understanding["caption"],
            "classification": classification,
            "annotated_image": annotated_filename,
        }

        if question:
            response["question"] = question
            response["answer"] = (
                self.vision_ai.answer_question(
                    image_path,
                    question,
                )
            )

        return response

    def close(self):
        self.engine.close()
