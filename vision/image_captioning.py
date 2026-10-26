from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Salesforce/blip-image-captioning-base"


class ImageCaptioner:
    def __init__(self):
        self.device = torch.device("cpu")

        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)

        self.model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

    def caption(self, image_path, max_new_tokens=40):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        return caption.strip()
