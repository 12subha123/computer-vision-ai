from vision.visual_context import VisualContextBuilder


class VisionReasoner:
    def __init__(self, llm=None):
        self.llm = llm
        self.context_builder = VisualContextBuilder()

    def build_prompt(self, context):
        visual_prompt = self.context_builder.to_prompt(context)

        return (
            "You are a visual AI assistant.\n"
            "Use the visual information below to answer the user's question.\n"
            "Do not invent objects, people, actions, or details that are not "
            "supported by the visual information.\n"
            "If the available information is insufficient, say so clearly.\n"
            "Give a concise, natural answer.\n\n"
            f"{visual_prompt}\n\n"
            "Answer the user's question based only on the available "
            "visual information."
        )

    def reason(self, context):
        prompt = self.build_prompt(context)

        if self.llm is None:
            return {
                "prompt": prompt,
                "answer": None,
                "status": "llm_not_configured",
            }

        answer = self.llm(prompt)

        return {
            "prompt": prompt,
            "answer": answer,
            "status": "success",
        }
