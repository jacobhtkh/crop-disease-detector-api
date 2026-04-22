import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import SUPPORTED_CROPS, app

# Fake classifier output covering multiple crops
MOCK_PREDICTIONS = [
    {"label": "Corn___Common_Rust", "score": 0.91},
    {"label": "Corn___Healthy", "score": 0.05},
    {"label": "Potato___Early_Blight", "score": 0.03},
    {"label": "Rice___Brown_Spot", "score": 0.01},
]


def make_image_bytes(fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(100, 150, 200)).save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def client():
    mock_classifier = MagicMock(return_value=MOCK_PREDICTIONS)
    with patch("main.pipeline", return_value=mock_classifier):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /supported-crops
# ---------------------------------------------------------------------------


def test_supported_crops_returns_200(client):
    response = client.get("/supported-crops")
    assert response.status_code == 200


def test_supported_crops_response_shape(client):
    body = client.get("/supported-crops").json()
    assert "crops" in body
    for crop in body["crops"]:
        assert "crop" in crop
        assert "conditions" in crop
        for condition in crop["conditions"]:
            assert "name" in condition
            assert "description" in condition


def test_supported_crops_contains_expected_crops(client):
    body = client.get("/supported-crops").json()
    crop_names = {c["crop"].lower() for c in body["crops"]}
    assert crop_names == SUPPORTED_CROPS


def test_supported_crops_conditions_are_non_empty(client):
    body = client.get("/supported-crops").json()
    for crop in body["crops"]:
        assert len(crop["conditions"]) > 0
        for condition in crop["conditions"]:
            assert condition["name"]
            assert condition["description"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_classify_valid_image_returns_200(client):
    response = client.post(
        "/classify",
        files=[("files", ("leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 1


def test_classify_response_shape(client):
    response = client.post(
        "/classify",
        files=[("files", ("leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    result = response.json()["results"][0]
    assert "filename" in result
    assert "cropInImage" in result
    assert "predictions" in result
    assert all("label" in p and "score" in p for p in result["predictions"])


def test_classify_multiple_images(client):
    files = [
        ("files", ("a.jpg", make_image_bytes(), "image/jpeg")),
        ("files", ("b.jpg", make_image_bytes(), "image/jpeg")),
    ]
    response = client.post("/classify", files=files)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_classify_png_image(client):
    response = client.post(
        "/classify",
        files=[("files", ("leaf.png", make_image_bytes("PNG"), "image/png"))],
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Crop detection from filename
# ---------------------------------------------------------------------------


def test_crop_in_image_detected_from_filename(client):
    response = client.post(
        "/classify",
        files=[("files", ("corn_leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    result = response.json()["results"][0]
    assert result["cropInImage"] == "corn"


def test_crop_filtering_limits_predictions_to_matched_crop(client):
    response = client.post(
        "/classify",
        files=[("files", ("corn_leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    predictions = response.json()["results"][0]["predictions"]
    assert all("corn" in p["label"].lower() for p in predictions)


def test_no_crop_in_filename_returns_null_crop_and_all_predictions(client):
    response = client.post(
        "/classify",
        files=[("files", ("random_image.jpg", make_image_bytes(), "image/jpeg"))],
    )
    result = response.json()["results"][0]
    assert result["cropInImage"] is None
    assert len(result["predictions"]) == len(MOCK_PREDICTIONS)


# ---------------------------------------------------------------------------
# top_k parameter
# ---------------------------------------------------------------------------


def test_top_k_is_forwarded_to_classifier(client):
    client.post(
        "/classify",
        params={"top_k": 3},
        files=[("files", ("leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    classifier = app.state.classifier
    _, kwargs = classifier.call_args
    assert kwargs["top_k"] == 3


def test_top_k_below_range_returns_400(client):
    response = client.post(
        "/classify",
        params={"top_k": 0},
        files=[("files", ("leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 400


def test_top_k_above_range_returns_400(client):
    response = client.post(
        "/classify",
        params={"top_k": 21},
        files=[("files", ("leaf.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_no_files_returns_422(client):
    # FastAPI validates the required `files` field before the handler runs
    response = client.post("/classify")
    assert response.status_code == 422


def test_non_image_file_returns_400(client):
    response = client.post(
        "/classify",
        files=[("files", ("note.txt", b"not an image", "text/plain"))],
    )
    assert response.status_code == 400
