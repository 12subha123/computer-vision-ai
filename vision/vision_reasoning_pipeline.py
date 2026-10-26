from vision.visual_context import VisualContextBuilder
from vision.gemini_reasoner import GeminiVisionReasoner


class VisionReasoningPipeline:
    def __init__(self):
        self.context_builder = VisualContextBuilder()
        self.reasoner = GeminiVisionReasoner()

    def build_context(
        self,
        vision_results,
        caption=None,
        classification=None,
        question=None,
    ):
        return self.context_builder.build(
            vision_results=vision_results,
            caption=caption,
            classification=classification,
            question=question,
        )

    def build_prompt(
        self,
        vision_results,
        caption=None,
        classification=None,
        question=None,
    ):
        context = self.build_context(
            vision_results=vision_results,
            caption=caption,
            classification=classification,
            question=question,
        )

        visual_prompt = self.context_builder.to_prompt(context)

        return (
            "You are a visual AI assistant.\n"
            "Answer the user's question using only the available visual "
            "information.\n"
            "Do not invent objects, people, actions, or details.\n"
            "If the information is insufficient, say so clearly.\n"
            "Give a concise, natural answer.\n\n"
            f"{visual_prompt}\n\n"
            "Answer:"
        )

    def answer(
        self,
        vision_results,
        caption=None,
        classification=None,
        question=None,
    ):
        prompt = self.build_prompt(
            vision_results=vision_results,
            caption=caption,
            classification=classification,
            question=question,
        )

        try:
            answer = self.reasoner.generate(prompt)

            return {
                "answer": answer,
                "prompt": prompt,
                "status": "success",
            }

        except Exception as error:
            error_text = str(error)

            if "503" in error_text or "UNAVAILABLE" in error_text:
                return {
                    "answer": None,
                    "prompt": prompt,
                    "status": "gemini_temporarily_unavailable",
                    "message": (
                        "Gemini is temporarily unavailable. "
                        "The visual analysis was completed, "
                        "but AI reasoning could not be completed."
                    ),
                }

            return {
                "answer": None,
                "prompt": prompt,
                "status": "reasoning_error",
                "message": error_text,
            }
