from pathlib import Path

from vision.vision_assistant import VisionAssistant
from vision.vision_reasoning_pipeline import VisionReasoningPipeline


class IntelligentVisionAssistant:
    def __init__(self):
        self.vision_assistant = VisionAssistant()
        self.reasoning_pipeline = VisionReasoningPipeline()

    def analyze(self, image_path, question=None):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        vision_result = self.vision_assistant.analyze(
            image_path,
            question=None,
        )

        context = vision_result.get("context", {})
        vision_results = vision_result.get("vision_results", {})

        caption = vision_result.get("caption")
        classification = vision_result.get("classification")

        answer = None

        if question:
            reasoning_result = self.reasoning_pipeline.answer(
                vision_results=vision_results,
                caption=caption,
                classification=classification,
                question=question,
            )
            answer = reasoning_result["answer"]

        return {
            "vision": vision_result,
            "question": question,
            "answer": answer,
            "context": context,
        }

    def close(self):
        self.vision_assistant.close()
