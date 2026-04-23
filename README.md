## About

This project is a small **HTTP API** for frontends that use AI to estimate whether a crop image shows disease, depending on the model you configure.

The server is a **FastAPI** app in `main.py`. It uses a Hugging Face model for image classification, loaded via the [Transformers](https://huggingface.co/docs/transformers) [`image-classification` pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ImageClassificationPipeline). The model is controlled by the `HF_IMAGE_CLASSIFICATION_MODEL` environment variable. If that variable is not set, the API falls back to `microsoft/resnet-50` (a general ImageNet classifier). For this crop disease classification the hugging face model used is [`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit).

### Supported crops and diseases

| Crop   | Disease / Label |
| ------ | --------------- |
| Corn   | Common Rust     |
| Corn   | Gray Leaf Spot  |
| Corn   | Healthy         |
| Corn   | Leaf Blight     |
| Potato | Early Blight    |
| Potato | Healthy         |
| Potato | Late Blight     |
| Rice   | Brown Spot      |
| Rice   | Healthy         |
| Rice   | Leaf Blast      |
| Wheat  | Brown Rust      |
| Wheat  | Healthy         |
| Wheat  | Yellow Rust     |

As you can see it's mostly leaf diseases covered.

Images that do not match a supported crop return an `Invalid` label.

## Architecture

**Languages**

- Python 3.12+

**Backend**

- FastAPI (HTTP API, automatic OpenAPI docs)
- Transformers `image-classification` pipeline (PyTorch via `torch`)
- Pillow for decoding uploaded image bytes
- `python-dotenv` to merge a local `.env` into the process environment (without overriding variables already set in the shell)
- `slowapi` for per-IP rate limiting (20 requests/minute)

**Deployment**

- Railway (or any host that can run a Python ASGI app)
- Deployed at: https://crop-disease-detector-api-production.up.railway.app

## How it works

### Startup (lifespan)

On startup, FastAPI runs a **lifespan** context once:

1. Read `HF_IMAGE_CLASSIFICATION_MODEL` from the environment. If unset, it defaults to `microsoft/resnet-50`.
2. Construct a single `pipeline("image-classification", model=..., device=...)` and attach it to **`app.state.classifier`**.
3. `/classify` reuses that pipeline so weights are not reloaded per request.

The first run may **download** weights from the Hugging Face Hub; later runs use the local cache.

### Hugging Face tokenizers

Before importing Transformers, `main.py` sets `TOKENIZERS_PARALLELISM` to `false` by default (via `os.environ.setdefault`), which avoids extra tokenizer multiprocessing. You can override it in the environment if you need different behavior.

### Configuration (`.env`)

Values are read from **`os.environ`**. The `.env` file beside `main.py` is loaded at import time with `load_dotenv` (entries in `.env` do **not** override variables already exported in your shell).

| Variable                        | Purpose                                                                                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HF_IMAGE_CLASSIFICATION_MODEL` | Hugging Face model id for `image-classification` (default: `microsoft/resnet-50`).                                                                                  |
| `HF_DEVICE`                     | Device for the pipeline: `cpu`, `mps`, `cuda:0`, a numeric GPU index (`0`, `1`, …), or `-1` / empty for CPU in Hugging Face’s legacy style. Default in code: `cpu`. |
| `HF_TOKEN`                      | **Optional** for public models. Use a Hub token for **private** or **gated** models (Transformers / Hub clients read this from the environment).                    |
| `TOKENIZERS_PARALLELISM`        | Optional override for tokenizer thread/process behavior (see above).                                                                                                |
| `ALLOWED_ORIGINS`               | Comma-separated list of allowed CORS origins (default: `http://localhost:5173`). Example: `https://myapp.com,https://www.myapp.com`.                                |

### Endpoints

Interactive docs: **`http://127.0.0.1:8000/docs`** when running locally on the default port.

#### `GET /supported-crops`

Returns the list of supported crops and their conditions, each with a brief description.

**Example Response:**

```json
{
  "crops": [
    {
      "crop": "Corn",
      "conditions": [
        {
          "name": "Common Rust",
          "description": "Caused by Puccinia sorghi, producing brick-red pustules on both leaf surfaces. Spores spread northward via wind each summer and are favored by cool, humid conditions."
        },
        {
          "name": "Gray Leaf Spot",
          "description": "Caused by Cercospora zeae-maydis, producing rectangular gray-brown lesions that run parallel between leaf veins. Thrives in warm, humid conditions and overwinters in corn debris."
        },
        {
          "name": "Leaf Blight",
          "description": "Northern Leaf Blight, caused by Exserohilum turcicum, produces long elliptical gray-green lesions on leaves that turn tan. Severe infections can reduce grain yields by 40–70%."
        },
        {
          "name": "Healthy",
          "description": "Crop shows no signs of disease or infection."
        }
      ]
    },
    {
      "crop": "Potato",
      "conditions": [
        {
          "name": "Early Blight",
          "description": "Caused by Alternaria solani, producing dark circular lesions with a concentric ring (target-like) pattern on older, lower leaves. Favored by alternating wet and dry periods."
        },
        {
          "name": "Late Blight",
          "description": "Caused by Phytophthora infestans, rapidly spreading dark blotches on leaves and stems that collapse quickly. Historically responsible for the 1845 Irish Potato Famine."
        },
        {
          "name": "Healthy",
          "description": "Crop shows no signs of disease or infection."
        }
      ]
    },
    {
      "crop": "Rice",
      "conditions": [
        {
          "name": "Brown Spot",
          "description": "Caused by Cochliobolus miyabeanus, producing oval brown spots with gray centers on leaves. Common in nutrient-deficient soils; can cause up to 45% yield loss."
        },
        {
          "name": "Leaf Blast",
          "description": "Caused by Magnaporthe oryzae, producing spindle-shaped whitish-gray lesions with brown borders on leaves. One of the most destructive rice diseases, responsible for 10–30% global yield loss."
        },
        {
          "name": "Healthy",
          "description": "Crop shows no signs of disease or infection."
        }
      ]
    },
    {
      "crop": "Wheat",
      "conditions": [
        {
          "name": "Brown Rust",
          "description": "Caused by Puccinia triticina, producing small round orange-brown pustules scattered across leaf surfaces. Wind-dispersed; high humidity and mild temperatures favor spread."
        },
        {
          "name": "Yellow Rust",
          "description": "Caused by Puccinia striiformis, producing yellow-orange spores arranged in stripes along leaves. A cool-season disease where yield losses can exceed 70% in severe epidemics."
        },
        {
          "name": "Healthy",
          "description": "Crop shows no signs of disease or infection."
        }
      ]
    }
  ]
}
```

#### `POST /classify`

- **Body:** `multipart/form-data` with one or more parts named **`files`**.
- **Body (optional):** `selected_crop_names_for_images` — a repeated form field, one value per image, explicitly naming the crop (e.g. `corn`, `wheat`). When provided and the crop is supported, it takes precedence over filename-based detection.
- **Query:** optional **`top_k`** (integer, **1–20**, default **5**): how many top labels/scores to return per image.

For each file the handler reads bytes, decodes with Pillow as **RGB**, runs the classifier in a **thread pool** (`asyncio.to_thread`) so inference does not block the event loop, and appends the result. The crop used to filter predictions is resolved in order: explicit `selected_crop_names_for_images` value → crop name found in the filename → `null` (no filtering).

**Example Response:**

```json
{
  "results": [
    {
      "filename": "green-wheat.jpeg",
      "cropInImage": "wheat",
      "predictions": [
        {
          "label": "Wheat___Healthy",
          "score": 0.9750185012817383
        },
        {
          "label": "Wheat___Yellow_Rust",
          "score": 0.005910073406994343
        }
      ]
    }
  ]
}
```

`cropInImage` is the detected crop name from the filename, or `null` if none matched. Predictions are filtered to only include labels matching the detected crop.

Invalid or non-image input returns **400** with a short message.

### Calling from HTTP clients

- **URL:** `http://127.0.0.1:8000/classify`.
- **Body:** `form-data`, key **`files`**, type **File** (repeat the key for multiple images).
- **Body (optional):** `form-data`, key **`selected_crop_names_for_images`**, type **Text** (repeat once per image in the same order as `files`).
- **Query (optional):** `top_k=5`.

## Running locally

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### Development (auto-reload)

```bash
uv run fastapi dev
```

### Production-style (no reload)

```bash
uv run fastapi run
```

## Running tests

```bash
uv run pytest
```

## Project layout (essentials)

- `main.py` — FastAPI app, lifespan, `/supported-crops`, `/classify`.
- `supported-crops.json` — crop and condition data (names and descriptions) served by `/supported-crops`.
- `railway.toml` — Railway deployment config (start command with `$PORT` binding).
- `.env` — local secrets and model/device overrides (gitignored; not committed).
- `pyproject.toml` / `uv.lock` — dependencies and lockfile.

## Citation

Hugging Face model used:

```bibtex
@misc{kinyua2024smartfarming,
  title={Smart Farming Disease Detection Transformer},
  author={Wambugu Kinyua},
  year={2024},
  publisher={Hugging Face},
}
```
