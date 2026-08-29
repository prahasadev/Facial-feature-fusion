from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import tempfile
import shutil
from pathlib import Path
from main import build_v9_splicer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.post("/api/generate-composite")
async def generate_composite(
    base_image: UploadFile = File(...),
    nose_source: UploadFile = File(...),
    mouth_source: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    base_path = temp_path / "base.jpg"
    nose_path = temp_path / "nose.jpg"
    mouth_path = temp_path / "mouth.jpg"
    output_path = temp_path / "v8_final.jpg"

    try:
        with open(base_path, "wb") as buffer:
            shutil.copyfileobj(base_image.file, buffer)

        with open(nose_path, "wb") as buffer:
            shutil.copyfileobj(nose_source.file, buffer)

        with open(mouth_path, "wb") as buffer:
            shutil.copyfileobj(mouth_source.file, buffer)

        build_v9_splicer(
            str(base_path),
            str(nose_path),
            str(mouth_path),
            str(output_path))

        if not output_path.exists():
            raise FileNotFoundError("V8 did not create the output image.")

        return FileResponse(
            str(output_path),
            media_type="image/jpeg",
            filename="v8_final.jpg",
            background=BackgroundTask(
                shutil.rmtree,
                temp_dir,
                ignore_errors=True))

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
