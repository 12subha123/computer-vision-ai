from google import genai


class GeminiVisionReasoner:
    def __init__(self, api_key=None, model="gemini-3.8-flash"):
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

        self.model = model

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text
