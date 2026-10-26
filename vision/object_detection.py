from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"


class ObjectDetector:
    def __init__(self, confidence=0.25):
        self.confidence = confidence

        if MODEL_PATH.exists():
            self.model = YOLO(str(MODEL_PATH))
        else:
            self.model = YOLO("yolo11n.pt")

            if not MODEL_PATH.exists():
                source = Path("yolo11n.pt")

                if source.exists():
                    source.replace(MODEL_PATH)

    def detect(self, image_path):
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class_id": class_id,
                        "label": self.model.names[class_id],
                        "confidence": confidence,
                        "box": coordinates,
                    }
                )

        return detections
