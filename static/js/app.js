"use strict";

/* Real-Time Sign Language Detection & Translation — frontend application.
 *
 * Responsibilities:
 *  - webcam capture (getUserMedia)  ->  JPEG frames
 *  - WebSocket to /ws/inference     ->  annotated frame + predictions
 *  - sentence building (add/space/delete/clear)
 *  - browser SpeechSynthesis "Speak"
 *  - model/connection status chips, FPS, keyboard shortcuts
 */

const VIDEO_FRAME_INTERVAL_MS = 66; // ~15 fps over the WebSocket (local only)
const JPEG_QUALITY = 0.7;

/* ------------------------------------------------ Sentence builder */

class SentenceBuilder {
  constructor(onChange) {
    this.tokens = [];
    this.onChange = onChange;
  }

  addToken(label) {
    if (!label) return;
    this.tokens.push(label);
    this.onChange(this.text());
  }

  addSpace() {
    if (!this.tokens.length) return;
    if (this.tokens[this.tokens.length - 1] !== " ") this.tokens.push(" ");
    this.onChange(this.text());
  }

  deleteLast() {
    if (!this.tokens.length) return;
    this.tokens.pop();
    this.onChange(this.text());
  }

  clear() {
    this.tokens = [];
    this.onChange(this.text());
  }

  text() {
    return this.tokens.join("");
  }
}

/* ------------------------------------------------ Speech */

class Speaker {
  constructor(canSpeak, statusEl) {
    this.supported = canSpeak && "speechSynthesis" in window;
    this.statusEl = statusEl;
  }

  speak(text) {
    if (!text.trim()) return;
    if (!this.supported) {
      this.statusEl.textContent = "Speech not supported by this browser.";
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  }

  stop() {
    if (this.supported) window.speechSynthesis.cancel();
  }
}

/* ------------------------------------------------ App */

class SignApp {
  constructor() {
    this.dom = {
      video: document.getElementById("video"),
      canvas: document.getElementById("annotated"),
      cameraEmpty: document.getElementById("camera-empty"),
      startBtn: document.getElementById("start-btn"),
      stopBtn: document.getElementById("stop-btn"),
      fps: document.getElementById("fps"),
      cameraError: document.getElementById("camera-error"),
      modelStatus: document.getElementById("model-status"),
      connectionStatus: document.getElementById("connection-status"),
      predictionLabel: document.getElementById("prediction-label"),
      predictionConfidence: document.getElementById("prediction-confidence"),
      confidenceFill: document.getElementById("confidence-fill"),
      sentenceOut: document.getElementById("sentence"),
      addBtn: document.getElementById("add-btn"),
      spaceBtn: document.getElementById("space-btn"),
      deleteBtn: document.getElementById("delete-btn"),
      clearBtn: document.getElementById("clear-btn"),
      speakBtn: document.getElementById("speak-btn"),
      cancelSpeakBtn: document.getElementById("cancel-speak-btn"),
    };

    this.ctx = this.dom.canvas.getContext("2d");
    this.stream = null;
    this.ws = null;
    this.sendTimer = null;
    this.lastPrediction = null;
    this.lastAcceptedLabel = null;
    this.sentence = new SentenceBuilder((text) => {
      this.dom.sentenceOut.textContent = text || " ";
    });
    this.speaker = new Speaker(true, null);

    this.lastFrameAcknowledged = true;
    this.fpsSamples = [];
  }

  /* ---- status helpers ---- */

  setModelStatus(ready, detail) {
    const el = this.dom.modelStatus;
    if (ready) {
      el.textContent = `Model: ready (${detail})`;
      el.className = "status-chip status-ok";
    } else {
      el.textContent = `Model: ${detail}`;
      el.className = "status-chip status-warn";
    }
  }

  setConnectionStatus(state, detail) {
    const el = this.dom.connectionStatus;
    let cls = "status-unknown";
    if (state === "open") cls = "status-ok";
    if (state === "closed" || state === "error") cls = "status-error";
    el.textContent = `Connection: ${detail || state}`;
    el.className = `status-chip ${cls}`;
  }

  /* ---- camera ---- */

  async startCamera() {
    this.hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.showError("Camera access is not supported in this browser.");
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      const detail =
        err && err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access and retry."
          : err && err.name === "NotFoundError"
          ? "No webcam was found on this device."
          : err && err.name === "NotReadableError"
          ? "The webcam is in use by another application."
          : "Could not start the camera: " + (err && err.message ? err.message : err);
      this.showError(detail);
      return;
    }
    this.dom.video.srcObject = this.stream;
    this.dom.cameraEmpty.hidden = true;
    this.dom.canvas.hidden = false;
    this.dom.startBtn.disabled = true;
    this.dom.stopBtn.disabled = false;
    this.dom.addBtn.disabled = false;
    this.startSending();
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.stopSending();
    this.dom.video.srcObject = null;
    this.dom.cameraEmpty.hidden = false;
    this.dom.canvas.hidden = true;
    this.dom.startBtn.disabled = false;
    this.dom.stopBtn.disabled = true;
    this.dom.fps.textContent = "FPS: —";
    this.setConnectionStatus("closed", "idle");
  }

  /* ---- frame sending over WebSocket ---- */

  startSending() {
    this.fpsSamples = [];
    this.sendTimer = setInterval(() => this.sendFrame(), VIDEO_FRAME_INTERVAL_MS);
  }

  stopSending() {
    if (this.sendTimer) {
      clearInterval(this.sendTimer);
      this.sendTimer = null;
    }
  }

  sendFrame() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (!this.stream || this.dom.video.readyState < 2) return;

    const width = this.dom.video.videoWidth || 640;
    const height = this.dom.video.videoHeight || 480;
    this.dom.canvas.width = width;
    this.dom.canvas.height = height;
    this.ctx.drawImage(this.dom.video, 0, 0, width, height);
    const jpeg = this.dom.canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    this.ws.send(JSON.stringify({ type: "frame", image: jpeg }));
  }

  /* ---- websocket ---- */

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/inference`);
    this.setConnectionStatus("connecting", "connecting…");

    this.ws.onopen = () => this.setConnectionStatus("open", "connected");
    this.ws.onclose = () => this.setConnectionStatus("closed", "closed");
    this.ws.onerror = () => this.setConnectionStatus("error", "error");
    this.ws.onmessage = (evt) => this.handleMessage(evt.data);
  }

  handleMessage(raw) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      return;
    }
    if (msg.type === "prediction") {
      this.renderPrediction(msg);
      if (msg.annotated) {
        const img = new Image();
        img.onload = () => {
          this.ctx.drawImage(img, 0, 0, this.dom.canvas.width, this.dom.canvas.height);
        };
        img.src = "data:image/jpeg;base64," + msg.annotated;
      }
      this.dom.fps.textContent = `FPS: ${msg.fps}`;
    } else if (msg.type === "error") {
      this.setModelStatus(false, "not trained");
      this.showError(msg.message || "Inference error.");
    }
  }

  renderPrediction(msg) {
    if (msg.label) {
      this.dom.predictionLabel.textContent = msg.label;
      this.dom.predictionConfidence.textContent =
        `Confidence: ${Math.round(msg.confidence * 100)}%`;
      this.dom.confidenceFill.style.width = `${Math.round(msg.confidence * 100)}%`;
    } else {
      this.dom.predictionLabel.textContent = "No sign / low confidence";
      this.dom.predictionConfidence.textContent = "";
      this.dom.confidenceFill.style.width = "0%";
    }

    this.lastPrediction = msg.label || null;

    if (msg.accepted_label) {
      this.lastAcceptedLabel = msg.accepted_label;
      this.flashAccepted(msg.accepted_label);
    }
  }

  flashAccepted(label) {
    const el = this.dom.predictionLabel;
    const prev = el.textContent;
    el.textContent = `✔ ${label}`;
    setTimeout(() => {
      if (this.dom.predictionLabel.textContent === `✔ ${label}`) {
        this.dom.predictionLabel.textContent = prev;
      }
    }, 900);
  }

  /* ---- fetch model info ---- */

  async loadModelInfo() {
    try {
      const res = await fetch("/api/model/info");
      const info = await res.json();
      if (info.model_ready) {
        this.setModelStatus(true, `${info.num_classes} signs`);
        this.trainedLabels = info.labels || [];
      } else {
        this.setModelStatus(false, "not trained");
        if (info.error && info.error.length > 0) {
          console.warn("[sign-app] Model info:", info.error);
        }
      }
    } catch (err) {
      this.setModelStatus(false, "unavailable");
    }
  }

  /* ---- sentence controls ---- */

  addCurrentPrediction() {
    const label = this.lastAcceptedLabel || this.lastPrediction;
    if (!label) {
      this.showError("No stable sign detected yet. Hold a gesture and wait for it to stabilize.");
      setTimeout(() => this.hideError(), 2500);
      return;
    }
    this.sentence.addToken(label);
    this.lastAcceptedLabel = null;
  }

  /* ---- errors ---- */

  showError(text) {
    this.dom.cameraError.textContent = text;
    this.dom.cameraError.hidden = false;
  }

  hideError() {
    this.dom.cameraError.textContent = "";
    this.dom.cameraError.hidden = true;
  }

  /* ---- wiring ---- */

  bind() {
    this.dom.startBtn.addEventListener("click", () => this.startCamera());
    this.dom.stopBtn.addEventListener("click", () => this.stopCamera());
    this.dom.addBtn.addEventListener("click", () => this.addCurrentPrediction());
    this.dom.spaceBtn.addEventListener("click", () => this.sentence.addSpace());
    this.dom.deleteBtn.addEventListener("click", () => this.sentence.deleteLast());
    this.dom.clearBtn.addEventListener("click", () => this.sentence.clear());
    this.dom.speakBtn.addEventListener("click", () =>
      this.speaker.speak(this.sentence.text())
    );
    this.dom.cancelSpeakBtn.addEventListener("click", () => this.speaker.stop());

    document.addEventListener("keydown", (evt) => {
      const target = evt.target;
      const isTyping =
        target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
      if (isTyping) return;

      switch (evt.key) {
        case "a":
        case "A":
          if (!this.lastPrediction) return;
          evt.preventDefault();
          this.addCurrentPrediction();
          break;
        case " ":
          evt.preventDefault();
          this.sentence.addSpace();
          break;
        case "Backspace":
          evt.preventDefault();
          this.sentence.deleteLast();
          break;
        case "Escape":
          this.sentence.clear();
          break;
      }
    });
  }

  init() {
    this.bind();
    this.connect();
    this.loadModelInfo();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const app = new SignApp();
  window.__signApp = app;
  app.init();
});