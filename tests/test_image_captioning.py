from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.image_captioning import ImageCaptioner


IMAGE_PATH = PROJECT_ROOT / "inference" / "face_test.jpg"

captioner = ImageCaptioner()
caption = captioner.caption(IMAGE_PATH)

print()
print("Image Captioning")
print("----------------")
print(f"Image: {IMAGE_PATH.name}")
print(f"Caption: {caption}")
