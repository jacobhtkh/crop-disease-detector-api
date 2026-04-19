import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# HF tokenizers: avoid extra multiprocessing (fork-safety / parallelism warnings).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# Merge key/value pairs from .env into os.environ so os.getenv reads them (does not override existing vars).
load_dotenv(Path(__file__).resolve().parent / ".env")


def _pipeline_device() -> int | str:
    """HF_DEVICE → transformers `pipeline(..., device=...)`.

    | HF_DEVICE    | Passed through as | Meaning (typical)        |
    |--------------|-------------------|--------------------------|
    | (unset)      | "cpu"             | CPU (default in code)    |
    | cpu          | "cpu"             | CPU                      |
    | (empty) / -1 | -1                | CPU (legacy HF style)    |
    | 0, 1, …      | 0, 1, …           | CUDA GPU by index        |
    | mps          | "mps"             | Apple Silicon GPU        |
    | cuda:0       | "cuda:0"          | Explicit CUDA device str |
    """
    raw = os.environ.get("HF_DEVICE", "cpu").strip()
    if raw in ("", "-1"):
        return -1
    if raw.isdigit():
        return int(raw)
    return raw


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.environ.get(
        "HF_IMAGE_CLASSIFICATION_MODEL",
        "microsoft/resnet-50",
    )
    print(f"Loading model: {model_id}")
    app.state.classifier = pipeline(
        "image-classification",
        model=model_id,
        device=_pipeline_device(),
    )
    yield
    del app.state.classifier


app = FastAPI(lifespan=lifespan)

@app.post("/classify")
async def classify_images(
    request: Request,
    files: List[UploadFile] = File(),
    top_k: int = 5,
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20",
        )
    classifier = request.app.state.classifier
    results: list[dict] = []
    for file in files:
        contents = await file.read()
        name = Path(file.filename or "image.bin").name
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail=f"Not a valid image: {name}",
            )
        predictions = await asyncio.to_thread(
            classifier,
            image,
            top_k=top_k,
        )
        results.append({"filename": name, "predictions": predictions})
    return {"results": results}