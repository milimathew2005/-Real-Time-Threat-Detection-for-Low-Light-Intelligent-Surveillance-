"""
app.py — Unified Flask Backend (Multi-Feed)
AIVIS Platform v4.0 — REAL TIME CRIME SURVEILLANCE

Integrates:
  • YOLOv8 weapon / person detection
  • EfficientNet + Bidirectional LSTM behaviour classification
  • Multiple simultaneous RTSP/IP-cam feeds with per-feed classification

Run:
    pip install -r requirements.txt
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
from ultralytics import YOLO
import cv2, threading, time, os, uuid, base64, numpy as np
from datetime import datetime
from collections import deque, Counter

# ── Optional deep-learning imports (LSTM engine) ──────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image
    from model import build_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import (
    YOLO_MODEL_PATH, LSTM_CHECKPOINT_PATH,
    CONF_BASE, CONF_GENERIC_WEAPON, CONF_ARMED_PERSON, MIN_BOX_AREA_RATIO,
    WEAPON_CLASSES, ARMED_PERSON_CLASSES, PERSON_CLASSES,
    CLASSES, NUM_CLASSES, IDX_TO_CLASS, NORMAL_IDX, CLASS_SEVERITY,
    CNN_BACKBONE, CNN_FEATURE_DIM, IMG_SIZE,
    SEQ_LEN, OVERLAP, LSTM_ALERT_THRESHOLD,
    CAMERA_LOCATION,
)

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Global models (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
yolo_model        = None
lstm_model        = None
cnn_extractor     = None
lstm_device       = None
lstm_loaded       = False

# ─────────────────────────────────────────────────────────────────────────────
# Multi-feed state
# ─────────────────────────────────────────────────────────────────────────────
feeds      = {}        # feed_id -> FeedState
feeds_lock = threading.Lock()
feed_counter = 0       # auto-increment for feed IDs

# Global alert history (across all feeds)
alert_history = deque(maxlen=100)
total_alerts  = 0

# ─────────────────────────────────────────────────────────────────────────────
# Global video-analysis job registry
# ─────────────────────────────────────────────────────────────────────────────
analysis_jobs = {}
jobs_lock     = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
#  Model loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_yolo():
    """Load the custom YOLOv8 weapon detector."""
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[✓] YOLO model loaded — classes: {list(yolo_model.names.values())}")
    except Exception:
        print("[!] Custom YOLO model not found — falling back to yolov8n.pt")
        yolo_model = YOLO("yolov8n.pt")


_IMG_TRANSFORM = None

def _get_transform():
    global _IMG_TRANSFORM
    if _IMG_TRANSFORM is None:
        _IMG_TRANSFORM = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _IMG_TRANSFORM


def _build_cnn(backbone: str, device):
    """Build CNN backbone with feature extraction head matching LSTM input dim."""
    import torch.nn as nn
    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()  # outputs 2048-dim
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, CNN_FEATURE_DIM),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m.eval().to(device)


def load_lstm():
    """Load CNN backbone + BiLSTM checkpoint (if available)."""
    global lstm_model, cnn_extractor, lstm_device, lstm_loaded
    if not TORCH_AVAILABLE:
        print("[!] PyTorch not available — LSTM engine disabled")
        return
    if torch.cuda.is_available():
        lstm_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        lstm_device = torch.device("mps")
    else:
        lstm_device = torch.device("cpu")
        
    print(f"[*] LSTM device: {lstm_device}")

    try:
        cnn_extractor = _build_cnn(CNN_BACKBONE, lstm_device)
        print(f"[✓] CNN backbone ({CNN_BACKBONE}) loaded")
    except Exception as e:
        print(f"[!] CNN backbone load failed: {e}")
        return

    if not os.path.isfile(LSTM_CHECKPOINT_PATH):
        print(f"[!] LSTM checkpoint not found at '{LSTM_CHECKPOINT_PATH}' — LSTM disabled")
        return

    try:
        lstm_model = build_model(num_classes=NUM_CLASSES).to(lstm_device)
        ckpt = torch.load(LSTM_CHECKPOINT_PATH, map_location=lstm_device)
        lstm_model.load_state_dict(ckpt["model_state"])
        lstm_model.eval()
        lstm_loaded = True
        print(f"[✓] LSTM model loaded (epoch {ckpt.get('epoch','?')}) — classes: {CLASSES}")
    except Exception as e:
        print(f"[!] LSTM checkpoint load failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  Inference helpers — YOLO
# ═════════════════════════════════════════════════════════════════════════════

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + Bilateral filtering for low-light noise reduction."""
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
    return denoised


def is_valid_detection(label, conf, x1, y1, x2, y2, frame_h, frame_w) -> bool:
    """Filter detections by size, aspect-ratio and per-class confidence."""
    box_area       = (x2 - x1) * (y2 - y1)
    frame_area     = max(frame_h * frame_w, 1)
    box_area_ratio = box_area / frame_area

    if box_area_ratio < MIN_BOX_AREA_RATIO:
        return False
    if label in WEAPON_CLASSES and box_area_ratio > 0.20:
        return False
    if label in {"rifle", "shotgun"}:
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        if 0.6 < w / h < 1.7:
            return False
    if label == "weapon" and conf < CONF_GENERIC_WEAPON:
        return False
    if label == "person hold weapon" and conf < CONF_ARMED_PERSON:
        return False
    return True


def run_yolo(frame, h, w):
    """Run YOLO on one frame; return (persons, weapons, dets)."""
    results = yolo_model.predict(frame, conf=CONF_BASE, iou=0.50,
                                 verbose=False, device="cpu")[0]
    persons = weapons = 0
    dets    = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = results.names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if not is_valid_detection(label, conf, x1, y1, x2, y2, h, w):
            continue
        if label in PERSON_CLASSES:
            persons += 1; cat = "person"
        elif label in WEAPON_CLASSES:
            weapons += 1; cat = "weapon"
        elif label in ARMED_PERSON_CLASSES:
            weapons += 1; cat = "weapon"
        else:
            continue
        col = (0, 220, 0) if cat == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        lbl_text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, lbl_text, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        dets.append({"label": label, "confidence": round(conf, 3), "category": cat})
    return persons, weapons, dets


# ═════════════════════════════════════════════════════════════════════════════
#  LSTMEngine — background feature-buffer + sliding-window inference
# ═════════════════════════════════════════════════════════════════════════════

class LSTMEngine:
    """
    Maintains a buffer of CNN features extracted from recent frames.
    Produces predictions by sliding-window after SEQ_LEN.
    """

    def __init__(self):
        self._buf    = []
        self._lock   = threading.Lock()
        self._stride = SEQ_LEN - OVERLAP
        self.label      = "—"
        self.confidence = 0.0
        self.severity   = "CLEAR"
        self.probs      = []

    @torch.no_grad()
    def _extract_feat(self, frame_bgr: np.ndarray) -> np.ndarray:
        img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t    = _get_transform()(img).unsqueeze(0).to(lstm_device)
        feat = cnn_extractor(t).cpu().numpy()[0]
        return feat

    @torch.no_grad()
    def _predict(self, window: np.ndarray):
        t      = torch.from_numpy(window).unsqueeze(0).to(lstm_device)
        logits, attn = lstm_model(t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        if len(probs) > 1 and float(probs[1]) >= LSTM_ALERT_THRESHOLD:
            cls = 1
            conf = float(probs[1])
        else:
            cls = 0
            conf = float(probs[0])
        return cls, conf, probs.tolist()

    def push(self, frame_bgr: np.ndarray):
        """Extract feature from frame and run LSTM when enough frames are buffered."""
        if not lstm_loaded:
            return
        feat = self._extract_feat(frame_bgr)
        with self._lock:
            self._buf.append(feat)
            buf_len = len(self._buf)

            # Determine if we should predict
            should_predict = False
            if buf_len >= SEQ_LEN:
                # Full window: use sliding window
                window = np.stack(self._buf[-SEQ_LEN:])
                should_predict = True
                # Slide the buffer
                self._buf = self._buf[self._stride:]

            if should_predict:
                cls, conf, probs = self._predict(window)
                label    = IDX_TO_CLASS.get(cls, "Unknown")
                severity = CLASS_SEVERITY.get(label, "MEDIUM")
                self.label      = label
                self.confidence = conf
                self.severity   = severity
                self.probs      = probs

    def reset(self):
        with self._lock:
            self._buf       = []
            self.label      = "—"
            self.confidence = 0.0
            self.severity   = "CLEAR"
            self.probs      = []


# ═════════════════════════════════════════════════════════════════════════════
#  FeedState — per-feed state container
# ═════════════════════════════════════════════════════════════════════════════

class FeedState:
    """Holds all state for a single RTSP/camera feed."""
    def __init__(self, feed_id: str, url: str, is_video: bool = False):
        self.feed_id   = feed_id
        self.url       = url
        self.is_video  = is_video
        self.thread    = None
        self.stop_flag = False
        self.running   = False
        self.lstm_engine = LSTMEngine()
        self.frame_b64 = None
        self.frame_lock = threading.Lock()
        self.stats = {
            "feed_id":         feed_id,
            "url":             url,
            "is_video":        is_video,
            "person_count":    0,
            "weapon_count":    0,
            "total_alerts":    0,
            "fps":             0,
            "status":          "idle",
            "last_updated":    None,
            "lstm_label":      "—",
            "lstm_confidence": 0.0,
            "lstm_severity":   "CLEAR",
            "lstm_loaded":     lstm_loaded,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Per-feed detection loop
# ═════════════════════════════════════════════════════════════════════════════

def detection_loop(feed_id: str):
    """Main detection loop for a single feed."""
    global total_alerts

    with feeds_lock:
        feed = feeds.get(feed_id)
    if not feed:
        return

    camera = cv2.VideoCapture(feed.url)
    if not camera.isOpened():
        feed.stats["status"] = "error"
        feed.running = False
        return

    feed.lstm_engine.reset()
    feed.stats["status"] = "buffering"
    feed.running = True

    # No pre-buffering needed; frontend shows 'ANALYZING...' until first prediction

    feed.stats["status"] = "monitoring"
    fps_counter = 0
    fps_start   = time.time()
    consecutive_weapon_frames = 0
    frame_count = 0

    while not feed.stop_flag:
        loop_start = time.time()
        ret, frame = camera.read()
        if not ret:
            # Try reconnecting once for RTSP streams
            camera.release()
            time.sleep(1.0)
            camera = cv2.VideoCapture(feed.url)
            if not camera.isOpened():
                break
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        # ── YOLO inference ─────────────────────────────────────────────────
        persons, weapons, dets = run_yolo(frame, h, w)

        # ── LSTM inference (synchronous, every 3rd frame for speed) ────────
        if frame_count % 3 == 0 and lstm_loaded:
            feed.lstm_engine.push(enhance_frame(frame.copy()))

        # ── Update per-feed stats ──────────────────────────────────────────
        feed.stats.update({
            "person_count":    persons,
            "weapon_count":    weapons,
            "last_updated":    datetime.now().strftime("%H:%M:%S"),
            "lstm_label":      feed.lstm_engine.label,
            "lstm_confidence": round(feed.lstm_engine.confidence, 3),
            "lstm_severity":   feed.lstm_engine.severity,
        })

        # ── Consecutive-weapon alert threshold ─────────────────────────────
        if weapons > 0:
            consecutive_weapon_frames += 1
        else:
            consecutive_weapon_frames = 0

        if consecutive_weapon_frames >= 2:
            feed.stats["status"] = "alert"
            if consecutive_weapon_frames == 2:
                total_alerts += 1
                feed.stats["total_alerts"] += 1
                alert_history.appendleft({
                    "id":              total_alerts,
                    "feed_id":         feed_id,
                    "time":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type":            "WEAPON DETECTED",
                    "weapon_count":    weapons,
                    "person_count":    persons,
                    "detections":      dets,
                    "location":        CAMERA_LOCATION["name"],
                    "severity":        "HIGH",
                    "behavior":        feed.lstm_engine.label,
                    "behavior_conf":   round(feed.lstm_engine.confidence, 3),
                    "behavior_severity": feed.lstm_engine.severity,
                })
        elif (lstm_loaded
              and feed.lstm_engine.label not in ("—", "Normal")
              and feed.lstm_engine.confidence >= LSTM_ALERT_THRESHOLD):
            feed.stats["status"] = "alert"
            # Generate behaviour alert
            if frame_count % 30 == 0:  # throttle behaviour-only alerts
                total_alerts += 1
                feed.stats["total_alerts"] += 1
                alert_history.appendleft({
                    "id":              total_alerts,
                    "feed_id":         feed_id,
                    "time":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type":            "ANOMALY DETECTED",
                    "weapon_count":    0,
                    "person_count":    persons,
                    "detections":      [],
                    "location":        CAMERA_LOCATION["name"],
                    "severity":        "CRITICAL",
                    "behavior":        feed.lstm_engine.label,
                    "behavior_conf":   round(feed.lstm_engine.confidence, 3),
                    "behavior_severity": feed.lstm_engine.severity,
                })
        else:
            feed.stats["status"] = "monitoring"

        # ── HUD overlay ────────────────────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            feed.stats["fps"] = fps_counter
            fps_counter = 0
            fps_start   = time.time()

        # Status bar on frame
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
        id_txt = f"[{feed_id}] W:{weapons} P:{persons}"
        cv2.putText(frame, id_txt, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if lstm_loaded:
            if feed.lstm_engine.label != "—":
                sev_col = (0, 200, 0) if feed.lstm_engine.severity == "CLEAR" else \
                          (0, 180, 255) if feed.lstm_engine.severity in ("LOW", "MEDIUM") else \
                          (0, 0, 220)
                lstm_txt = f"BEHAVIOR: {feed.lstm_engine.label}  {feed.lstm_engine.confidence:.0%}"
            else:
                sev_col = (150, 150, 150)
                lstm_txt = "BEHAVIOR: ANALYZING..."
        else:
            sev_col = (0, 0, 255)
            lstm_txt = "BEHAVIOR: MODELS UNAVAILABLE"

        (tw, _), _ = cv2.getTextSize(lstm_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, lstm_txt, (w - tw - 8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sev_col, 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with feed.frame_lock:
            feed.frame_b64 = base64.b64encode(buf).decode()

        elapsed = time.time() - loop_start
        if elapsed < 0.03:
            time.sleep(0.03 - elapsed)

    camera.release()
    feed.stats["status"] = "idle"
    feed.running = False


# ═════════════════════════════════════════════════════════════════════════════
#  Feed management helpers
# ═════════════════════════════════════════════════════════════════════════════

def add_feed(url: str, name: str = None, is_video: bool = False) -> dict:
    """Add a new RTSP feed. Returns feed info dict."""
    global feed_counter
    with feeds_lock:
        feed_counter += 1
        feed_id = name or f"CAM-{feed_counter}"
        # If name already exists, append counter
        while feed_id in feeds:
            feed_counter += 1
            feed_id = f"CAM-{feed_counter}"

        feed = FeedState(feed_id, url, is_video=is_video)
        feeds[feed_id] = feed

    # Start detection loop
    feed.thread = threading.Thread(target=detection_loop, args=(feed_id,), daemon=True)
    feed.thread.start()

    return {"feed_id": feed_id, "url": url, "status": "starting", "is_video": is_video}


def remove_feed(feed_id: str) -> bool:
    """Stop and remove a feed. Returns True if found."""
    global feed_counter
    with feeds_lock:
        feed = feeds.get(feed_id)
    if not feed:
        return False

    feed.stop_flag = True
    if feed.thread and feed.thread.is_alive():
        feed.thread.join(timeout=3.0)

    with feeds_lock:
        feeds.pop(feed_id, None)
        # Reset counter when all feeds are removed so numbering restarts
        if not feeds:
            feed_counter = 0
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Flask routes — multi-feed API
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboardrtsp.html", location=CAMERA_LOCATION)


# ── Feed management ────────────────────────────────────────────────────────

@app.route("/api/feeds/add", methods=["POST"])
def api_add_feed():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    name = data.get("name", "").strip() or None
    result = add_feed(url, name)
    return jsonify(result)


@app.route("/api/feeds/<feed_id>/restart", methods=["POST"])
def api_restart_feed(feed_id):
    """Restart a video feed (re-run the same file through the model)."""
    with feeds_lock:
        feed = feeds.get(feed_id)
    if not feed or not feed.is_video:
        return jsonify({"error": "Feed not found or not a video feed"}), 404

    video_path = feed.url
    if not os.path.isfile(video_path):
        return jsonify({"error": "Video file no longer exists"}), 404

    # Stop the current feed
    feed.stop_flag = True
    if feed.thread and feed.thread.is_alive():
        feed.thread.join(timeout=3.0)

    # Reset state in-place (keep the same feed_id and card)
    feed.stop_flag = False
    feed.running   = False
    feed.frame_b64 = None
    feed.lstm_engine.reset()
    feed.stats.update({
        "person_count":    0,
        "weapon_count":    0,
        "fps":             0,
        "status":          "idle",
        "last_updated":    None,
        "lstm_label":      "—",
        "lstm_confidence": 0.0,
        "lstm_severity":   "CLEAR",
    })

    # Re-start detection loop with the same file
    feed.thread = threading.Thread(target=detection_loop, args=(feed_id,), daemon=True)
    feed.thread.start()

    return jsonify({"status": "restarted", "feed_id": feed_id})


@app.route("/api/feeds/<feed_id>/remove", methods=["POST"])
def api_remove_feed(feed_id):
    if remove_feed(feed_id):
        return jsonify({"status": "removed", "feed_id": feed_id})
    return jsonify({"error": "Feed not found"}), 404


@app.route("/api/feeds", methods=["GET"])
def api_list_feeds():
    with feeds_lock:
        result = []
        for fid, feed in feeds.items():
            info = dict(feed.stats)
            info["lstm_loaded"] = lstm_loaded
            result.append(info)
    return jsonify(result)


@app.route("/api/feeds/<feed_id>/frame", methods=["GET"])
def api_feed_frame(feed_id):
    with feeds_lock:
        feed = feeds.get(feed_id)
    if not feed:
        return jsonify({"error": "Feed not found"}), 404
    with feed.frame_lock:
        return jsonify({"frame": feed.frame_b64, "feed_id": feed_id})


@app.route("/api/feeds/<feed_id>/stats", methods=["GET"])
def api_feed_stats(feed_id):
    with feeds_lock:
        feed = feeds.get(feed_id)
    if not feed:
        return jsonify({"error": "Feed not found"}), 404
    info = dict(feed.stats)
    info["lstm_loaded"] = lstm_loaded
    # Include LSTM probs
    info["probs"] = {}
    if feed.lstm_engine.probs:
        info["probs"] = {CLASSES[i]: round(float(p), 3)
                         for i, p in enumerate(feed.lstm_engine.probs)}
    return jsonify(info)


# ── Global endpoints ───────────────────────────────────────────────────────

@app.route("/api/alerts")
def api_alerts():
    return jsonify(list(alert_history))


@app.route("/api/clear_alerts", methods=["POST"])
def api_clear():
    global total_alerts
    alert_history.clear()
    total_alerts = 0
    with feeds_lock:
        for feed in feeds.values():
            feed.stats["total_alerts"] = 0
    return jsonify({"status": "cleared"})


@app.route("/api/global_stats")
def api_global_stats():
    """Aggregate stats across all feeds."""
    with feeds_lock:
        total_persons = sum(f.stats["person_count"] for f in feeds.values())
        total_weapons = sum(f.stats["weapon_count"] for f in feeds.values())
        total_feed_alerts = sum(f.stats["total_alerts"] for f in feeds.values())
        active_feeds = sum(1 for f in feeds.values() if f.running)
        statuses = [f.stats["status"] for f in feeds.values()]

    overall_status = "idle"
    if "alert" in statuses:
        overall_status = "alert"
    elif "monitoring" in statuses:
        overall_status = "monitoring"

    return jsonify({
        "person_count":  total_persons,
        "weapon_count":  total_weapons,
        "total_alerts":  total_alerts,
        "active_feeds":  active_feeds,
        "status":        overall_status,
        "last_updated":  datetime.now().strftime("%H:%M:%S"),
        "lstm_loaded":   lstm_loaded,
    })


# ── Legacy single-feed compat endpoints ────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    """Legacy: start a single feed."""
    data = request.get_json(silent=True) or {}
    url = data.get("source", "http://158.58.130.148/mjpg/video.mjpg")
    result = add_feed(url)
    return jsonify({"status": "started", "feed_id": result["feed_id"]})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop and remove all feeds, reset counter."""
    global feed_counter
    with feeds_lock:
        feed_ids = list(feeds.keys())
    for fid in feed_ids:
        remove_feed(fid)
    # Force reset counter even if remove_feed didn't catch it
    with feeds_lock:
        feed_counter = 0
    return jsonify({"status": "stopped"})


# ── Video upload ───────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ext     = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format"}), 400

    job_id   = str(uuid.uuid4())[:8]
    filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    f.save(filepath)

    # Play video as a feed
    result = add_feed(filepath, name=f"VIDEO-{job_id[:4].upper()}", is_video=True)
    return jsonify({"job_id": "simulated", "status": "playing_in_live_feed",
                    "feed_id": result["feed_id"]})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with jobs_lock:
        job = analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_yolo()
    load_lstm()
    print("[✓] AIVIS v4.0 — REAL TIME CRIME SURVEILLANCE (Multi-Feed)")
    print("[✓] Portal Active: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
