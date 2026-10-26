from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.vision_ai import VisionAI


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

vision_ai = VisionAI()

questions = [
    "Is there a person in the image?",
    "What is the person wearing?",
]

print()
print("Vision AI - Visual Question Answering")
print("--------------------------------------")

for question in questions:
    answer = vision_ai.answer_question(
        IMAGE_PATH,
        question,
    )

    print()
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
