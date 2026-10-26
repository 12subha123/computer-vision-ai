from vision.scene_interpreter import SceneInterpreter
from vision.image_captioning import ImageCaptioner
from vision.vqa import VisualQuestionAnswering


class VisionAI:
    def __init__(self):
        self.scene_interpreter = SceneInterpreter()
        self.captioner = ImageCaptioner()
        self.vqa = VisualQuestionAnswering()

    def understand(self, image_path, results):
        scene = self.scene_interpreter.summarize(results)
        caption = self.captioner.caption(image_path)

        return {
            "scene_description": scene["description"],
            "caption": caption,
            "object_count": scene["object_count"],
        }

    def answer_question(self, image_path, question):
        return self.vqa.answer(
            image_path,
            question,
        )
