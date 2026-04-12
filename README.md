## About

This is a project to create an API for frontends that can use AI to detect if a crop is diseased or not.

The server is a **FastAPI** app (`main.py`) that can **store uploaded images** and **run image classification** using Hugging Face [Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ImageClassificationPipeline). The default model is a general ImageNet classifier; for real crop-disease use, point `HF_IMAGE_CLASSIFICATION_MODEL` at a **fine-tuned** checkpoint such as (LishaV01/agriculture-crop-disease-detection) whose labels match your problem.

## Architecture

The current tech stack is:

**Languages**

- Python 3.12+

**Backend**

- FastAPI (HTTP API, automatic OpenAPI docs)
- [Transformers](https://huggingface.co/docs/transformers) `image-classification` pipeline (PyTorch + `torch`)
- Pillow for decoding uploaded image bytes
- `python-dotenv` to load a local `.env` into the process environment

**Deployment**

- Railway

## How it works

### Startup (`lifespan`)

When the process starts, FastAPI runs a **lifespan** context once:

1. Read `HF_IMAGE_CLASSIFICATION_MODEL` from the environment (see below). If unset, it defaults to `microsoft/resnet-50`.
2. Build a single **`pipeline("image-classification", ...)`** and store it on **`app.state.classifier`**.
3. All `/classify` requests reuse that pipeline so the weights are not reloaded per request.

The first run may **download** the model from the Hugging Face Hub; later runs use the cache.

### Configuration (`.env`)

Variables are read from **`os.environ`**. A `.env` file next to `main.py` is loaded at import time via `load_dotenv` (values in `.env` do not override variables already set in the shell).

| Variable                        | Purpose                                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `HF_IMAGE_CLASSIFICATION_MODEL` | Hugging Face model id for `image-classification` (default: `microsoft/resnet-50`).                                                  |
| `HF_DEVICE`                     | Device passed to the pipeline (`cpu`, `mps`, `cuda:0`, `0` for GPU index, `-1` for CPU legacy style, etc.). Default in code: `cpu`. |
| `HF_TOKEN`                      | Compulsory. Hub token for private or gated models (Transformers reads this from the environment).                                   |

### Endpoints

Interactive docs: **`http://127.0.0.1:8000/docs`** (when running locally).

#### `POST /upload`

Accepts **multipart form-data** with one or more parts named **`files`**.

- Saves each file under **`uploads/`** (directory is created if missing).
- Response: `{"filenames": ["...", "..."]}`.
- Duplicate basenames overwrite earlier files in `uploads/`.

#### `POST /classify`

Accepts the same **`files`** multipart field. Optional query parameter **`top_k`** (integer, **1–20**, default **5**): number of top labels/scores per image.

For each file:

1. Read bytes asynchronously.
2. Decode with Pillow and convert to **RGB**.
3. Run the classifier in a **thread pool** (`asyncio.to_thread`) so slow inference does not block the async event loop.
4. Append `{"filename": "...", "predictions": [...]}` to the response list.

Response shape: `{"results": [...]}`. Each `predictions` entry is typically a list of objects with **`label`** and **`score`** (exact shape depends on the model).

Invalid or non-image data returns **400** with a short message.

### Calling from Postman or clients

- **URL:** `http://127.0.0.1:8000/classify` (single slash before `classify`; a typo like `//classify` returns **404**).
- **Body:** `form-data`, key **`files`**, type **File** (repeat the key for multiple images).
- **Query (optional):** `top_k=5`.

## Running locally

From the project root (with [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
uv run fastapi dev
```

The dev server prints the local URL (usually `http://127.0.0.1:8000`).

## Project layout (essentials)

- `main.py` — FastAPI app, lifespan, `/upload`, `/classify`.
- `.env` — local secrets and model/device overrides (not committed; keep it gitignored).
- `uploads/` — default directory for saved uploads (created on demand).
- `pyproject.toml` / `uv.lock` — dependencies and lockfile.
