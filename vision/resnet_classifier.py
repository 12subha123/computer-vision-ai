from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


class ResNet18Classifier:
    """Image classifier based on pretrained ResNet-18."""

    def __init__(self):
        self.device = torch.device("cpu")

        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        self.model.eval()
        self.model.to(self.device)

        self.transform = self.weights.transforms()

        self.categories = self.weights.meta["categories"]

    def prepare_image(self, image_path: str | Path) -> torch.Tensor:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)

        return tensor.unsqueeze(0)

    def predict(self, image_path: str | Path, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        tensor = self.prepare_image(image_path)

        with torch.no_grad():
            output = self.model(tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        values, indices = torch.topk(
            probabilities,
            min(top_k, len(self.categories)),
        )

        results = []

        for probability, index in zip(values, indices):
            results.append({
                "label": self.categories[index.item()],
                "confidence": float(probability.item()),
            })

        return results
