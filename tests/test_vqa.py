from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.vqa import VisualQuestionAnswering


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

vqa = VisualQuestionAnswering()

questions = [
    "What is in the image?",
    "Is there a person in the image?",
    "What is the person doing?",
]

print()
print("Visual Question Answering")
print("-------------------------")

for question in questions:
    answer = vqa.answer(IMAGE_PATH, question)

    print()
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
