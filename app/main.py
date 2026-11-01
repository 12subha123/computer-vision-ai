from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vision.vision_assistant import VisionAssistant
from vision.video_processor import VideoVisionProcessor
from vision.intelligent_vision_assistant import IntelligentVisionAssistant
from app.satellite.router import router as satellite_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"

app = FastAPI(
    title="Computer Vision AI",
    description="Vision Assistant API for image and video understanding.",
    version="1.2.0",
)

assistant = VisionAssistant()
intelligent_assistant = IntelligentVisionAssistant()
video_processor = VideoVisionProcessor()

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

app.include_router(satellite_router)


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/vision/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str | None = Form(default=None),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    suffix = Path(image.filename or "").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = assistant.analyze(
            temp_path,
            question,
        )

        return {
            "filename": image.filename,
            "question": question,
            "result": result,
        }

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/vision/intelligent")
async def intelligent_analyze(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a question.",
        )

    suffix = Path(image.filename or "").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = intelligent_assistant.analyze(
            temp_path,
            question.strip(),
        )

        return {
            "filename": image.filename,
            "question": question.strip(),
            "result": result,
        }

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/vision/video")
async def analyze_video(video: UploadFile = File(...)):
    if not (
        video.content_type
        and video.content_type.startswith("video/")
    ) and Path(video.filename or "").suffix.lower() not in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file.",
        )

    suffix = Path(video.filename or "").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = video_processor.process(temp_path)

        return {
            "filename": video.filename,
            "result": result,
        }

    finally:
        temp_path.unlink(missing_ok=True)


@app.on_event("shutdown")
def shutdown():
    assistant.close()
    intelligent_assistant.close()
    video_processor.close()
