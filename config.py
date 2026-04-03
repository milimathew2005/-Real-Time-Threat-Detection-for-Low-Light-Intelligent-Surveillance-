"""
config.py — Unified Configuration
Real-Time Threat Detection for Low-Light Intelligent Surveillance
ARGUS Platform — Muthoot Institute of Technology and Science, Kochi

Covers both:
  • YOLO weapon-detection thresholds
  • LSTM behavioural-classification hyperparameters
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# ── Paths — YOLO
# ─────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "models/weapon_yolov8.pt"

# ─────────────────────────────────────────────────────────────────────────────
# ── YOLO detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
CONF_BASE           = 0.34   # base confidence for all detections
CONF_GENERIC_WEAPON = 0.45   # higher threshold for the generic "weapon" class
CONF_ARMED_PERSON   = 0.45   # threshold for "person hold weapon"
MIN_BOX_AREA_RATIO  = 0.002  # ignore boxes < 0.2 % of frame area

# YOLO class groupings (must match final-dataset labels exactly)
WEAPON_CLASSES       = {"pistol", "rifle", "shotgun", "knife", "ax", "weapon"}
ARMED_PERSON_CLASSES = {"person hold weapon"}
PERSON_CLASSES       = {"person"}

# ─────────────────────────────────────────────────────────────────────────────
# ── Paths — LSTM
# ─────────────────────────────────────────────────────────────────────────────
LSTM_CHECKPOINT_PATH = r"C:\Users\milim\Desktop\real-time-crime-detection\Threat\checkpoints\best_lstm.pth"

# ─────────────────────────────────────────────────────────────────────────────
# ── LSTM — UCF-Crime# ── Classes (2-class subset used during training) ─────────────
CLASSES = [
    "Normal",
    "Anomaly"
]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

NORMAL_CLASS  = "Normal"
NORMAL_IDX    = CLASS_TO_IDX["Normal"]

# Severity mapping: how dangerous is each predicted class?
CLASS_SEVERITY = {
    "Normal":       "CLEAR",
    "Anomaly":      "CRITICAL",
}

# ─────────────────────────────────────────────────────────────────────────────
# ── LSTM — CNN feature extraction
# ─────────────────────────────────────────────────────────────────────────────
CNN_BACKBONE    = "efficientnet_b0"
CNN_FEATURE_DIM = 512
IMG_SIZE        = (224, 224)
BATCH_SIZE_CNN  = 32

# ─────────────────────────────────────────────────────────────────────────────
# ── LSTM — sequence / model
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN     = 32     # frames per LSTM input window
OVERLAP     = 16     # sliding-window overlap# ── Model (LSTM) hyperparameters ──
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT     = 0.5
FC_HIDDEN   = 64

# ─────────────────────────────────────────────────────────────────────────────
#  LSTM — inference threshold
# ─────────────────────────────────────────────────────────────────────────────
LSTM_ALERT_THRESHOLD = 0.54   # minimum confidence to raise a behaviour alert

# ─────────────────────────────────────────────────────────────────────────────
# ── Camera location (for live-map widget)
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_LOCATION = {
    "name":    "Main Entrance — Block A",
    "lat":     9.9312,
    "lng":     76.2673,
    "address": "Muthoot Institute of Technology and Science, Kochi",
}

# ─────────────────────────────────────────────────────────────────────────────
# ── Training paths (used by Threat/ scripts, not by app.py)
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.join("Threat", "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_lstm.pth")
LOG_DIR         = os.path.join("Threat", "logs")
