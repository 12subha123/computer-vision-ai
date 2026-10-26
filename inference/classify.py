import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.resnet_classifier import ResNet18Classifier


def main():
    parser = argparse.ArgumentParser(
        description="Classify an image with pretrained ResNet-18."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    classifier = ResNet18Classifier()
    results = classifier.predict(args.image, top_k=args.top_k)

    print()
    print("Image Classification")
    print("====================")
    print(f"Image: {args.image}")
    print()

    for number, result in enumerate(results, 1):
        percentage = result["confidence"] * 100
        print(f"{number}. {result['label']} - {percentage:.2f}%")


if __name__ == "__main__":
    main()
