#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Optional dependencies (graceful fallback if unavailable)
try:
    import mediapipe as mp  # type: ignore

    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

try:
    import face_alignment  # type: ignore

    FACE_ALIGNMENT_AVAILABLE = True
except Exception:
    face_alignment = None
    FACE_ALIGNMENT_AVAILABLE = False


# =========================
# Config (edit here)
# =========================
DATASET_ROOT = Path("./robot_head_dataset")
IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
REJECTED_IMAGE_DIR = DATASET_ROOT / "rejected" / "images"
REJECTED_LABEL_DIR = DATASET_ROOT / "rejected" / "labels"
METADATA_FILE = DATASET_ROOT / "metadata.jsonl"
REJECTED_METADATA_FILE = DATASET_ROOT / "rejected_metadata.jsonl"
CONFIG_FILE = DATASET_ROOT / "config.json"

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 5
JPEG_QUALITY = 95

CAPTURE_INTERVAL_SEC = 5.0
MOTOR_SETTLE_SEC = 1.0
MAX_SAMPLES = 1000
MAX_ATTEMPTS = 0  # 0 means unlimited attempts until MAX_SAMPLES accepted
PREVIEW = True
SAVE_REJECTED = True

# Quality thresholds
MIN_FRAME_WIDTH = 640
MIN_FRAME_HEIGHT = 480
MIN_BLUR_SCORE = 70.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 215.0
MAX_DARK_PIXEL_RATIO = 0.45
MAX_BRIGHT_PIXEL_RATIO = 0.35

# Face/pose thresholds
REQUIRE_SINGLE_FACE = True
MIN_FACE_AREA_RATIO = 0.04
MAX_CENTER_OFFSET_RATIO = 0.35
MIN_LANDMARK_SCORE = 0.10
MAX_ABS_YAW = 45.0
MAX_ABS_PITCH = 35.0
MAX_ABS_ROLL = 35.0

# Very mild geometry sanity checks (avoid obvious failed faces only)
MIN_EYE_DISTANCE_RATIO = 0.12
MAX_MOUTH_OPEN_RATIO = 0.65

# Detector preferences
PREFER_MEDIAPIPE = True
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.40
MEDIAPIPE_MAX_NUM_FACES = 2
FACE_ALIGNMENT_USE_CUDA_IF_AVAILABLE = True

# motor30 config
MOTOR_DIM = 30
MOTOR_GROUPS = {
    "brow": [0, 1, 2, 3],
    "eye": [4, 5, 6, 7, 8, 9],
    "head_jaw_pose": [10, 11, 12, 13, 14, 27, 28],
    "mouth": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "nose": [29],
}

MOTOR_SYMMETRY_PAIRS = [
    (0, 1),  # brow inner L/R
    (2, 3),  # brow outer L/R
    (4, 5),  # lower eyelid L/R
    (6, 7),  # upper eyelid L/R
    (16, 18),  # lower lip depress L/R
    (19, 20),  # lip corner raise L/R
    (21, 22),  # lip corner stretch L/R
    (24, 26),  # upper lip raise L/R
]

# Per-motor min/max, default [0,1]
MOTOR_MIN = np.array([0.0] * MOTOR_DIM, dtype=np.float32)
MOTOR_MAX = np.array([1.0] * MOTOR_DIM, dtype=np.float32)

NEUTRAL_MOTOR = np.array(
    [
        0.42, 0.42, 0.40, 0.40,  # brow
        0.45, 0.45, 0.50, 0.50, 0.50, 0.50,  # eyes + gaze
        0.50, 0.50, 0.50, 0.45, 0.45,  # head/jaw
        0.40, 0.40, 0.40, 0.40, 0.42, 0.42, 0.40, 0.40, 0.45, 0.42, 0.42, 0.42,  # mouth
        0.50, 0.50,  # neck
        0.20,  # nose wrinkle
    ],
    dtype=np.float32,
)

BASE_NOISE_STD = 0.06
SYMMETRY_BLEND = 0.85
BROW_MAX_DEVIATION_FROM_MEAN = 0.22
MAX_MOUTH_EXTREME_COUNT = 4
NOSE_EXTREME_PROB = 0.08
JAW_PRIMARY_INDEX = 13
MOUTH_OPEN_INDEX = 25

# Small expression templates (delta around neutral)
EXPRESSION_TEMPLATES = [
    {"name": "neutral_soft", "delta": {}},
    {"name": "smile", "delta": {19: 0.20, 20: 0.20, 21: 0.12, 22: 0.12, 23: 0.10, 25: 0.08}},
    {"name": "sad", "delta": {19: -0.16, 20: -0.16, 15: 0.12, 16: 0.08, 18: 0.08}},
    {"name": "surprise", "delta": {0: 0.15, 1: 0.15, 2: 0.14, 3: 0.14, 6: 0.15, 7: 0.15, 13: 0.18, 25: 0.20}},
    {"name": "blink_like", "delta": {4: 0.20, 5: 0.20, 6: -0.18, 7: -0.18}},
]


@dataclass
class QualityResult:
    accepted: bool
    reason: str
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    backend: str = "none"
    face_info: dict[str, Any] = field(default_factory=dict)


def setup_dataset_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def save_config(face_backend: str) -> None:
    cfg = {
        "dataset_root": str(DATASET_ROOT.resolve()),
        "camera": {
            "camera_id": CAMERA_ID,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "warmup_frames": CAMERA_WARMUP_FRAMES,
            "jpeg_quality": JPEG_QUALITY,
        },
        "capture": {
            "capture_interval_sec": CAPTURE_INTERVAL_SEC,
            "motor_settle_sec": MOTOR_SETTLE_SEC,
            "max_samples": MAX_SAMPLES,
            "max_attempts": MAX_ATTEMPTS,
            "preview": PREVIEW,
            "save_rejected": SAVE_REJECTED,
        },
        "quality_thresholds": {
            "min_frame_width": MIN_FRAME_WIDTH,
            "min_frame_height": MIN_FRAME_HEIGHT,
            "min_blur_score": MIN_BLUR_SCORE,
            "min_brightness": MIN_BRIGHTNESS,
            "max_brightness": MAX_BRIGHTNESS,
            "max_dark_pixel_ratio": MAX_DARK_PIXEL_RATIO,
            "max_bright_pixel_ratio": MAX_BRIGHT_PIXEL_RATIO,
        },
        "face_thresholds": {
            "require_single_face": REQUIRE_SINGLE_FACE,
            "min_face_area_ratio": MIN_FACE_AREA_RATIO,
            "max_center_offset_ratio": MAX_CENTER_OFFSET_RATIO,
            "min_landmark_score": MIN_LANDMARK_SCORE,
            "max_abs_yaw": MAX_ABS_YAW,
            "max_abs_pitch": MAX_ABS_PITCH,
            "max_abs_roll": MAX_ABS_ROLL,
            "min_eye_distance_ratio": MIN_EYE_DISTANCE_RATIO,
            "max_mouth_open_ratio": MAX_MOUTH_OPEN_RATIO,
        },
        "motor": {
            "dim": MOTOR_DIM,
            "groups": MOTOR_GROUPS,
            "symmetry_pairs": MOTOR_SYMMETRY_PAIRS,
            "neutral_motor": [float(x) for x in NEUTRAL_MOTOR.tolist()],
            "motor_min": [float(x) for x in MOTOR_MIN.tolist()],
            "motor_max": [float(x) for x in MOTOR_MAX.tolist()],
        },
        "face_backend": face_backend,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp_motor(motor: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(motor, MOTOR_MIN), MOTOR_MAX)


def generate_random_motor_command() -> list[float]:
    """
    Generate a constrained random motor30 command.
    Strategy:
      1) start from neutral
      2) add template-driven deltas + small noise
      3) enforce mild symmetry and sanity constraints
    """
    motor = NEUTRAL_MOTOR.copy()
    motor += np.random.normal(0.0, BASE_NOISE_STD, size=MOTOR_DIM).astype(np.float32)

    template = random.choice(EXPRESSION_TEMPLATES)
    intensity = random.uniform(0.35, 1.0)
    for idx, delta in template["delta"].items():
        motor[idx] += float(delta) * intensity

    # Symmetry coupling (left/right correlated but not identical)
    for left_idx, right_idx in MOTOR_SYMMETRY_PAIRS:
        left_v = motor[left_idx]
        noise = np.random.normal(0.0, 0.03)
        motor[right_idx] = float(SYMMETRY_BLEND * left_v + (1.0 - SYMMETRY_BLEND) * motor[right_idx] + noise)

    # Brow values should not diverge too much
    brow_idx = np.array(MOTOR_GROUPS["brow"], dtype=np.int64)
    brow_mean = float(np.mean(motor[brow_idx]))
    brow_delta = np.clip(motor[brow_idx] - brow_mean, -BROW_MAX_DEVIATION_FROM_MEAN, BROW_MAX_DEVIATION_FROM_MEAN)
    motor[brow_idx] = brow_mean + brow_delta

    # Mouth should not have too many extreme dimensions simultaneously
    mouth_idx = np.array(MOTOR_GROUPS["mouth"], dtype=np.int64)
    mouth_vals = motor[mouth_idx]
    extreme_mask = np.abs(mouth_vals - 0.5) > 0.35
    if int(np.sum(extreme_mask)) > MAX_MOUTH_EXTREME_COUNT:
        motor[mouth_idx] = 0.6 * mouth_vals + 0.4 * np.mean(mouth_vals)

    # Avoid jaw-open + mouth-open both strongly extreme all the time
    if motor[JAW_PRIMARY_INDEX] > 0.86 and motor[MOUTH_OPEN_INDEX] > 0.86:
        motor[JAW_PRIMARY_INDEX] = 0.72 + 0.08 * random.random()
        motor[MOUTH_OPEN_INDEX] = 0.72 + 0.08 * random.random()

    # Nose wrinkle should rarely hit extremes
    if random.random() < NOSE_EXTREME_PROB:
        motor[29] = random.uniform(0.0, 1.0)
    else:
        motor[29] = 0.10 + 0.45 * np.random.beta(2.5, 5.5)

    motor = _clamp_motor(motor)
    return [float(round(x, 6)) for x in motor.tolist()]


def apply_motor_command(motor_values: list[float]) -> None:
    """
    Placeholder for real robot control.
    Replace this function with hardware communication in future.
    """
    print(f"[MOTOR] apply command (stub): first6={motor_values[:6]} ...")


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={CAMERA_ID}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    frame = None
    for _ in range(max(1, CAMERA_WARMUP_FRAMES)):
        ok, frm = cap.read()
        if ok and frm is not None:
            frame = frm
    if frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def validate_image_quality(frame: np.ndarray) -> QualityResult:
    if frame is None or frame.size == 0:
        return QualityResult(accepted=False, reason="empty_frame")
    h, w = frame.shape[:2]
    if w < MIN_FRAME_WIDTH or h < MIN_FRAME_HEIGHT:
        return QualityResult(
            accepted=False,
            reason=f"frame_too_small({w}x{h})",
            frame_width=w,
            frame_height=h,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray < 20))
    bright_ratio = float(np.mean(gray > 245))

    if brightness_mean < MIN_BRIGHTNESS:
        return QualityResult(False, "too_dark", blur_score, brightness_mean, brightness_std, w, h)
    if brightness_mean > MAX_BRIGHTNESS:
        return QualityResult(False, "too_bright", blur_score, brightness_mean, brightness_std, w, h)
    if dark_ratio > MAX_DARK_PIXEL_RATIO:
        return QualityResult(False, "too_many_dark_pixels", blur_score, brightness_mean, brightness_std, w, h)
    if bright_ratio > MAX_BRIGHT_PIXEL_RATIO:
        return QualityResult(False, "too_many_bright_pixels", blur_score, brightness_mean, brightness_std, w, h)
    if blur_score < MIN_BLUR_SCORE:
        return QualityResult(False, "too_blurry", blur_score, brightness_mean, brightness_std, w, h)

    return QualityResult(True, "ok", blur_score, brightness_mean, brightness_std, w, h)


def _fold_angle_90(angle_deg: float) -> float:
    x = (angle_deg + 180.0) % 360.0 - 180.0
    if x > 90.0:
        x -= 180.0
    elif x < -90.0:
        x += 180.0
    return float(x)


def _estimate_pose_from_2d_points(points_2d: np.ndarray, width: int, height: int) -> tuple[bool, float, float, float]:
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],  # nose tip
            [0.0, -63.6, -12.5],  # chin
            [-43.3, 32.7, -26.0],  # left eye corner
            [43.3, 32.7, -26.0],  # right eye corner
            [-28.9, -28.9, -24.1],  # left mouth corner
            [28.9, -28.9, -24.1],  # right mouth corner
        ],
        dtype=np.float64,
    )

    focal = float(max(width, height))
    camera_matrix = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(model_points, points_2d.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat([rot_mat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(v) for v in euler.flatten()]
    return True, _fold_angle_90(yaw), _fold_angle_90(pitch), _fold_angle_90(roll)


def init_face_backend() -> dict[str, Any]:
    if PREFER_MEDIAPIPE and MEDIAPIPE_AVAILABLE:
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=MEDIAPIPE_MAX_NUM_FACES,
                refine_landmarks=False,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            )
            print("[INFO] face backend: mediapipe")
            return {"name": "mediapipe", "model": face_mesh}
        except Exception as e:
            print(f"[WARN] mediapipe init failed: {e}")

    if FACE_ALIGNMENT_AVAILABLE:
        try:
            device = "cpu"
            if FACE_ALIGNMENT_USE_CUDA_IF_AVAILABLE:
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        device = "cuda"
                except Exception:
                    device = "cpu"
            model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.THREE_D,
                device=device,
                face_detector="sfd",
                flip_input=False,
                verbose=False,
            )
            print(f"[INFO] face backend: face_alignment ({device})")
            return {"name": "face_alignment", "model": model}
        except Exception as e:
            print(f"[WARN] face_alignment init failed: {e}")

    print("[WARN] no face backend available, face validation will be skipped.")
    return {"name": "none", "model": None}


def detect_face_and_pose(frame: np.ndarray, backend: dict[str, Any]) -> dict[str, Any]:
    h, w = frame.shape[:2]
    backend_name = backend.get("name", "none")

    if backend_name == "mediapipe":
        face_mesh = backend["model"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        faces = result.multi_face_landmarks if result is not None else None
        if not faces:
            return {"detected": False, "reason": "no_face", "backend": backend_name}
        if REQUIRE_SINGLE_FACE and len(faces) != 1:
            return {"detected": False, "reason": f"face_count_{len(faces)}", "backend": backend_name, "face_count": len(faces)}

        # choose largest face by bbox area
        best_face = None
        best_area = -1.0
        for fm in faces:
            xs = np.array([lm.x for lm in fm.landmark], dtype=np.float64) * w
            ys = np.array([lm.y for lm in fm.landmark], dtype=np.float64) * h
            x1, y1 = float(np.min(xs)), float(np.min(ys))
            x2, y2 = float(np.max(xs)), float(np.max(ys))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_face = fm

        if best_face is None:
            return {"detected": False, "reason": "no_face_after_select", "backend": backend_name}

        xs = np.array([lm.x for lm in best_face.landmark], dtype=np.float64) * w
        ys = np.array([lm.y for lm in best_face.landmark], dtype=np.float64) * h
        x1, y1 = float(np.min(xs)), float(np.min(ys))
        x2, y2 = float(np.max(xs)), float(np.max(ys))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
        area_ratio = float((bw * bh) / max(float(w * h), 1.0))
        center_offset = math.sqrt((cx - w * 0.5) ** 2 + (cy - h * 0.5) ** 2) / (0.5 * math.sqrt(w * w + h * h))

        # MediaPipe landmark indexes for rough pose
        # nose tip=1, chin=152, left eye outer=33, right eye outer=263, mouth corners=61/291
        idxs = [1, 152, 33, 263, 61, 291]
        points_2d = np.array(
            [[best_face.landmark[i].x * w, best_face.landmark[i].y * h] for i in idxs],
            dtype=np.float64,
        )
        pose_ok, yaw, pitch, roll = _estimate_pose_from_2d_points(points_2d, width=w, height=h)

        eye_dist = math.dist(
            (best_face.landmark[33].x * w, best_face.landmark[33].y * h),
            (best_face.landmark[263].x * w, best_face.landmark[263].y * h),
        )
        mouth_open = math.dist(
            (best_face.landmark[13].x * w, best_face.landmark[13].y * h),
            (best_face.landmark[14].x * w, best_face.landmark[14].y * h),
        )

        return {
            "detected": True,
            "backend": backend_name,
            "face_count": len(faces),
            "bbox": [float(x1), float(y1), float(bw), float(bh)],
            "bbox_area_ratio": area_ratio,
            "center_offset_ratio": float(center_offset),
            "landmark_score": 1.0,
            "pose_ok": bool(pose_ok),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "eye_distance_ratio": float(eye_dist / max(bw, 1.0)),
            "mouth_open_ratio": float(mouth_open / max(bh, 1.0)),
        }

    if backend_name == "face_alignment":
        model = backend["model"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_all = model.get_landmarks_from_image(rgb)
        if landmarks_all is None or len(landmarks_all) == 0:
            return {"detected": False, "reason": "no_face", "backend": backend_name}
        if REQUIRE_SINGLE_FACE and len(landmarks_all) != 1:
            return {"detected": False, "reason": f"face_count_{len(landmarks_all)}", "backend": backend_name, "face_count": len(landmarks_all)}

        lm = np.asarray(landmarks_all[0], dtype=np.float64)  # shape approx (68, 3)
        x1, y1 = float(np.min(lm[:, 0])), float(np.min(lm[:, 1]))
        x2, y2 = float(np.max(lm[:, 0])), float(np.max(lm[:, 1]))
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
        area_ratio = float((bw * bh) / max(float(w * h), 1.0))
        center_offset = math.sqrt((cx - w * 0.5) ** 2 + (cy - h * 0.5) ** 2) / (0.5 * math.sqrt(w * w + h * h))

        # 68-point pose points: nose tip(30), chin(8), eye corners(36,45), mouth corners(48,54)
        points_2d = np.array([lm[30, :2], lm[8, :2], lm[36, :2], lm[45, :2], lm[48, :2], lm[54, :2]], dtype=np.float64)
        pose_ok, yaw, pitch, roll = _estimate_pose_from_2d_points(points_2d, width=w, height=h)

        eye_dist = math.dist((lm[36, 0], lm[36, 1]), (lm[45, 0], lm[45, 1]))
        mouth_open = math.dist((lm[62, 0], lm[62, 1]), (lm[66, 0], lm[66, 1]))

        return {
            "detected": True,
            "backend": backend_name,
            "face_count": int(len(landmarks_all)),
            "bbox": [float(x1), float(y1), float(bw), float(bh)],
            "bbox_area_ratio": area_ratio,
            "center_offset_ratio": float(center_offset),
            "landmark_score": 1.0,
            "pose_ok": bool(pose_ok),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "eye_distance_ratio": float(eye_dist / max(bw, 1.0)),
            "mouth_open_ratio": float(mouth_open / max(bh, 1.0)),
        }

    return {"detected": False, "reason": "face_validation_skipped", "backend": "none"}


def validate_robot_face_image(frame: np.ndarray, backend: dict[str, Any]) -> QualityResult:
    q = validate_image_quality(frame)
    q.backend = backend.get("name", "none")
    if not q.accepted:
        return q

    if backend.get("name", "none") == "none":
        q.accepted = True
        q.reason = "ok_quality_only_face_validation_skipped"
        q.face_info = {"backend": "none", "face_validation_skipped": True}
        return q

    face = detect_face_and_pose(frame, backend)
    q.face_info = face
    if not face.get("detected", False):
        q.accepted = False
        q.reason = f"face_rejected:{face.get('reason', 'unknown')}"
        return q

    if float(face.get("landmark_score", 0.0)) < MIN_LANDMARK_SCORE:
        q.accepted = False
        q.reason = "face_rejected:low_landmark_score"
        return q
    if float(face.get("bbox_area_ratio", 0.0)) < MIN_FACE_AREA_RATIO:
        q.accepted = False
        q.reason = "face_rejected:face_too_small"
        return q
    if float(face.get("center_offset_ratio", 1.0)) > MAX_CENTER_OFFSET_RATIO:
        q.accepted = False
        q.reason = "face_rejected:face_off_center"
        return q
    if not bool(face.get("pose_ok", False)):
        q.accepted = False
        q.reason = "face_rejected:pose_estimation_failed"
        return q
    if abs(float(face.get("yaw", 0.0))) > MAX_ABS_YAW:
        q.accepted = False
        q.reason = "face_rejected:yaw_too_large"
        return q
    if abs(float(face.get("pitch", 0.0))) > MAX_ABS_PITCH:
        q.accepted = False
        q.reason = "face_rejected:pitch_too_large"
        return q
    if abs(float(face.get("roll", 0.0))) > MAX_ABS_ROLL:
        q.accepted = False
        q.reason = "face_rejected:roll_too_large"
        return q
    if float(face.get("eye_distance_ratio", 0.0)) < MIN_EYE_DISTANCE_RATIO:
        q.accepted = False
        q.reason = "face_rejected:eye_distance_too_small"
        return q
    if float(face.get("mouth_open_ratio", 0.0)) > MAX_MOUTH_OPEN_RATIO:
        q.accepted = False
        q.reason = "face_rejected:mouth_open_too_large"
        return q

    q.accepted = True
    q.reason = "ok"
    return q


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_sample(
    sample_id: str,
    frame: np.ndarray,
    motor: list[float],
    quality: QualityResult,
    camera_id: int,
    accepted: bool,
) -> tuple[Path, Path]:
    ts = datetime.now().isoformat(timespec="milliseconds")
    if accepted:
        img_path = IMAGE_DIR / f"{sample_id}.jpg"
        lbl_path = LABEL_DIR / f"{sample_id}.json"
        meta_path = METADATA_FILE
    else:
        img_path = REJECTED_IMAGE_DIR / f"{sample_id}.jpg"
        lbl_path = REJECTED_LABEL_DIR / f"{sample_id}.json"
        meta_path = REJECTED_METADATA_FILE

    if accepted or SAVE_REJECTED:
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])

    record = {
        "sample_id": sample_id,
        "image_path": str(img_path.as_posix()),
        "label_path": str(lbl_path.as_posix()),
        "timestamp": ts,
        "camera_id": int(camera_id),
        "accepted": bool(accepted),
        "motor": [float(v) for v in motor],
        "quality": asdict(quality),
        "face_info": quality.face_info,
        "reject_reason": None if accepted else quality.reason,
    }

    if accepted or SAVE_REJECTED:
        lbl_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_jsonl(meta_path, record)
    return img_path, lbl_path


def _preview_frame(
    frame: np.ndarray,
    sample_id: str,
    accepted: bool,
    reason: str,
    accepted_count: int,
    rejected_count: int,
) -> None:
    disp = frame.copy()
    color = (0, 180, 0) if accepted else (0, 0, 220)
    cv2.putText(disp, f"{sample_id}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(disp, "ACCEPTED" if accepted else "REJECTED", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(disp, reason[:90], (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(
        disp,
        f"accepted={accepted_count} rejected={rejected_count}  (press q to quit)",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("capture_robot_head_dataset", disp)


def main() -> None:
    setup_dataset_dirs()
    backend = init_face_backend()
    save_config(face_backend=backend.get("name", "none"))

    cap = open_camera()
    accepted_count = 0
    rejected_count = 0
    attempt = 0
    start_time = time.time()

    print("[INFO] Start capture loop.")
    print(f"[INFO] target accepted samples={MAX_SAMPLES}")

    try:
        while accepted_count < MAX_SAMPLES:
            if MAX_ATTEMPTS > 0 and attempt >= MAX_ATTEMPTS:
                print(f"[INFO] Reached MAX_ATTEMPTS={MAX_ATTEMPTS}, stop.")
                break

            loop_t0 = time.time()
            attempt += 1
            sample_id = f"sample_{attempt:06d}"

            motor = generate_random_motor_command()
            apply_motor_command(motor)
            time.sleep(max(0.0, MOTOR_SETTLE_SEC))

            frame = capture_frame(cap)
            result = validate_robot_face_image(frame, backend=backend)
            accepted = bool(result.accepted)

            if accepted or SAVE_REJECTED:
                img_path, _ = save_sample(
                    sample_id=sample_id,
                    frame=frame,
                    motor=motor,
                    quality=result,
                    camera_id=CAMERA_ID,
                    accepted=accepted,
                )
            else:
                img_path = Path("<not_saved>")

            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1

            elapsed = time.time() - start_time
            speed = attempt / max(elapsed, 1e-6)
            print(
                f"[SAMPLE {attempt:06d}] "
                f"{'ACCEPT' if accepted else 'REJECT'} "
                f"reason={result.reason} "
                f"blur={result.blur_score:.1f} "
                f"bright_mean={result.brightness_mean:.1f} "
                f"saved={img_path} "
                f"accepted={accepted_count} rejected={rejected_count} speed={speed:.2f} sample/s"
            )

            if PREVIEW:
                _preview_frame(
                    frame=frame,
                    sample_id=sample_id,
                    accepted=accepted,
                    reason=result.reason,
                    accepted_count=accepted_count,
                    rejected_count=rejected_count,
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] User requested quit (q).")
                    break

            loop_elapsed = time.time() - loop_t0
            sleep_time = max(0.0, CAPTURE_INTERVAL_SEC - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if backend.get("name") == "mediapipe":
            try:
                backend["model"].close()
            except Exception:
                pass

    total_elapsed = time.time() - start_time
    print(
        f"[DONE] attempts={attempt}, accepted={accepted_count}, rejected={rejected_count}, "
        f"elapsed={total_elapsed:.1f}s, avg_speed={attempt / max(total_elapsed, 1e-6):.2f} sample/s"
    )
    print(f"[DONE] dataset root: {DATASET_ROOT.resolve()}")


if __name__ == "__main__":
    main()
