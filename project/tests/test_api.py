"""Smoke tests for the FastAPI surface."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    # Disable DB tables auto-create — these tests don't require Postgres.
    import os
    os.environ["AUTO_CREATE_TABLES"] = "0"
    from backend.server.main import app
    with TestClient(app) as c:
        yield c


def _png_bytes(size: int = 32, color=(200, 100, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "insole" in r.json().get("service", "")


def test_health_responds(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "model_loaded", "device", "db_connected"):
        assert key in body


def test_classify_requires_at_least_one_image(client):
    r = client.post("/api/classify", files={}, data={})
    assert r.status_code in (400, 422, 503)


def test_classify_with_one_image_returns_prediction(client):
    img = _png_bytes()
    files = {"lateral": ("lat.png", img, "image/png")}
    r = client.post("/api/classify", files=files)
    # Either inference is wired (200) or model never loaded (503) — both are
    # acceptable signals that the route is alive.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "predicted_class" in body
        assert "class_probabilities" in body
        assert isinstance(body["class_probabilities"], dict)


def test_data_summary(client):
    r = client.get("/api/data/summary")
    assert r.status_code == 200
    body = r.json()
    assert "data_dir" in body and "exists" in body
