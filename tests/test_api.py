"""Integration tests for the FastAPI application and WebSocket inference."""

import base64
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _blank_jpeg_data_uri() -> str:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_ready"], bool)


def test_model_info(client):
    res = client.get("/api/model/info")
    assert res.status_code == 200
    body = res.json()
    assert "model_ready" in body
    assert "labels" in body
    if body["model_ready"]:
        assert body["feature_dim"] == 63


def test_web_root_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Sign Language Detection" in res.text


def test_static_assets(client):
    assert client.get("/css/style.css").status_code == 200
    assert client.get("/js/app.js").status_code == 200


def test_unknown_route_not_found(client):
    res = client.get("/definitely/not/here")
    assert res.status_code == 404


def test_websocket_inference_blank_frame(client):
    with client.websocket_connect("/ws/inference") as ws:
        ws.send_text(json.dumps({"type": "frame", "image": _blank_jpeg_data_uri()}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "prediction"
        assert "label" in msg
        assert "confidence" in msg
        assert "annotated" in msg
        assert len(msg["annotated"]) > 0  # annotated JPEG returned
        assert isinstance(msg["fps"], (int, float))


def test_websocket_rejects_malformed_json(client):
    with client.websocket_connect("/ws/inference") as ws:
        ws.send_text("not json at all")
        # The server must stay alive and respond to valid frames afterwards.
        ws.send_text(json.dumps({"type": "frame", "image": _blank_jpeg_data_uri()}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "prediction"


def test_websocket_rejects_nonframe_type(client):
    with client.websocket_connect("/ws/inference") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        ws.send_text(json.dumps({"type": "frame", "image": _blank_jpeg_data_uri()}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "prediction"


def test_websocket_binary_jpeg_ok(client):
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    with client.websocket_connect("/ws/inference") as ws:
        ws.send_bytes(buf.tobytes())
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "prediction"