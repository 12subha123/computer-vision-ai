from pathlib import Path

from PIL import Image


class ImageClassifier:
    """Base image-classification engine."""

    def load_image(self, image_path: str | Path) -> Image.Image:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        return image

    def get_image_info(self, image_path: str | Path) -> dict:
        image = self.load_image(image_path)

        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
