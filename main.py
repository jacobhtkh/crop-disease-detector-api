import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

# HF tokenizers: avoid extra multiprocessing (fork-safety / parallelism warnings).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
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


SUPPORTED_CROPS = frozenset({"corn", "potato", "rice", "wheat"})


class Prediction(BaseModel):
    label: str
    score: float


class ImageResult(BaseModel):
    filename: str
    cropInImage: str | None
    predictions: list[Prediction]


class ClassifyResponse(BaseModel):
    results: list[ImageResult]


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "limit": str(exc.limit.limit)},  # type: ignore
    )


app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit("20/minute")
async def classify_images(
    request: Request,
    files: list[UploadFile] = File(),
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
    results: list[ImageResult] = []
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

        crop_in_image = next(
            (c for c in SUPPORTED_CROPS if c.lower() in name.lower()), None
        )
        if crop_in_image is not None:
            predictions = [
                p for p in predictions if crop_in_image.lower() in p["label"].lower()
            ]
        results.append(
            ImageResult(
                filename=name,
                cropInImage=crop_in_image,
                predictions=predictions,
            )
        )
    return {"results": results}
