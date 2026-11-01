# Security and Privacy Design

This document describes the security and privacy properties of the
Real-Time Sign Language Detection and Translation System.

## Threat Model

The application runs on a **local development machine** (a student laptop,
typically Windows/macOS/Linux). The primary risks considered are:

1. Accidental leakage of personal camera data.
2. Secrets or credentials accidentally committed to Git.
3. Malformed input crashing the server or causing unsafe behavior.
4. Path traversal or arbitrary file writes via user-controlled paths.
5. A remote party interacting with a publicly exposed instance.

The system is **not** designed to be exposed to the public internet without a
reverse proxy, authentication, and TLS. This is stated in the README.

## Privacy

- **Camera frames never leave the machine.** The browser sends JPEG frames
  over a localhost WebSocket to the FastAPI backend for classification.
  Nothing is sent to any external service.
- **No webcam frames are stored.** During inference, frames exist only as
  in-memory buffers (NumPy arrays) and are discarded after processing. No
  code path writes frame data to disk.
- **Collected datasets contain landmarks, not video.** `scripts/collect_data.py`
  stores 21 hand landmarks per sample (numbers), never images. This is both
  a performance and a privacy decision.
- **Logging never includes image data.** Log records contain counts,
  connection events, and configuration summaries; never pixel values or
  landmark vectors. See `app/core/logging_conf.py`.

## Secrets

- No API keys are required and no secrets exist in the codebase.
- `.env` is ignored by Git (see `.gitignore`) and only the placeholder
  `.env.example` is committed.
- Configuration is read at runtime via `python-dotenv`
  (`app/core/config.py`).

## Input Validation

| Area | Control |
| --- | --- |
| Gesture labels | Must match `^[A-Z0-9_]{1,24}$` and be part of `configs/vocabulary.json` when configured (`scripts/collect_data.py`). |
| Sample counts | Integer between 1 and 5000. |
| Dataset split | Class coverage guards (>= 2 classes, >= 3 samples per class). |
| WebSocket frames | Maximum payload of 3 MB; base64-decoding failures are caught and skipped; malformed JSON is ignored without crashing the connection. |
| Model paths | Configured via environment variables with project-relative defaults; the predictor refuses to load missing files with a clear error. |

## Path Traversal

No endpoint accepts arbitrary user-controlled filesystem paths. Dataset
paths come from scripts' `--raw/--out/--labels` arguments or environment
variables (developer-controlled), never from a client request. WebSocket
messages carry image payloads only.

## CORS

- Default (development): origins are the localhost port the app runs on.
- Production: `APP_ENV=production` **requires** an explicit `CORS_ORIGINS`
  allow-list and rejects `*` and debug mode (`app/core/config.py::validate`).

## Error Handling

- API errors return structured JSON; full exception details are logged
  server-side (without image data) rather than echoed to clients.
- Model-missing states produce a clear, actionable message
  (train first), not a stack trace.

## Dependency Hygiene

- Dependency versions are pinned in `requirements.txt`.
- Only well-known, maintained open-source packages are used
  (OpenCV, MediaPipe, TensorFlow, FastAPI, Uvicorn, scikit-learn, pytest).

## Operational Notes

- Run with `APP_ENV=production` and `APP_DEBUG=false` for any real
  deployment, and place the app behind an authenticated reverse proxy with TLS.
- The WebSocket endpoint is unauthenticated by design because the app is
  intended strictly for localhost use. Do not expose it directly on a
  public network.