"""Tests for application configuration validation and vocabulary loading."""

import json

import pytest

from app.core.config import AppSettings, load_vocabulary


def test_default_settings_validate():
    settings = AppSettings()
    settings.validate()


@pytest.mark.parametrize(
    "patch, expected",
    [
        ({"app_env": "production", "cors_origins": []}, "CORS_ORIGINS must be set"),
        ({"app_env": "production", "cors_origins": ["*"]}, "Wildcard CORS"),
        ({"app_env": "production", "cors_origins": ["http://a", "http://b"], "app_debug": True}, "APP_DEBUG"),
    ],
)
def test_production_validation_failures(patch, expected):
    settings = AppSettings(**patch)
    with pytest.raises(ValueError, match=expected):
        settings.validate()


def test_production_with_explicit_origins_ok():
    settings = AppSettings(
        app_env="production",
        cors_origins=["https://demo.example.com"],
        app_debug=False,
    )
    settings.validate()


def test_invalid_numeric_ranges_rejected():
    with pytest.raises(ValueError):
        AppSettings(confidence_threshold=0.0).validate()
    with pytest.raises(ValueError):
        AppSettings(confidence_threshold=1.5).validate()
    with pytest.raises(ValueError):
        AppSettings(smoothing_window_size=0).validate()
    with pytest.raises(ValueError):
        AppSettings(smoothing_min_stability=1.5).validate()
    with pytest.raises(ValueError):
        AppSettings(smoothing_cooldown_frames=-3).validate()


def test_load_vocabulary_from_file(tmp_path):
    vocab = {"gestures": ["A", "B", "HELLO"]}
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    assert load_vocabulary(p) == ["A", "B", "HELLO"]


def test_load_vocabulary_missing_file(tmp_path):
    assert load_vocabulary(tmp_path / "nope.json") == []


def test_load_vocabulary_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_vocabulary(p) == []


def test_env_loading_via_dotenv(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        "APP_PORT=9000\nCORS_ORIGINS=http://a.com, http://b.com\n"
        "CONFIDENCE_THRESHOLD=0.55\n",
        encoding="utf-8",
    )
    for key in ("APP_PORT", "CORS_ORIGINS", "CONFIDENCE_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)
    settings = AppSettings.from_env(dotenv_path=p)
    assert settings.app_port == 9000
    assert settings.cors_origins == ["http://a.com", "http://b.com"]
    assert settings.confidence_threshold == pytest.approx(0.55)