#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Optional dependencies
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
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
PREVIEW = True
PROCESS_EVERY_N_FRAMES = 2
PROCESS_INTERVAL_SEC = 0.10  # ~10 FPS max for face pipeline
CAMERA_WARMUP_FRAMES = 5

TEMP_DIR = Path("./runtime/human_input")
LATEST_INPUT_PATH = TEMP_DIR / "latest_input.jpg"
LATEST_FULL_FRAME_PATH = TEMP_DIR / "latest_full_frame.jpg"
LATEST_METADATA_PATH = TEMP_DIR / "latest_metadata.json"
HISTORY_DIR = TEMP_DIR / "history"
SAVE_DEBUG_FRAMES = True
SAVE_HISTORY_EVERY_VALID_FRAME = False
JPEG_QUALITY = 95

ENABLE_LIVE_INFERENCE = False
LIVE_INFERENCE_MIN_INTERVAL_SEC = 0.12

# Detector preference
PREFER_MEDIAPIPE = True
MEDIAPIPE_MAX_NUM_FACES = 5
MEDIAPIPE_MIN_DET_CONF = 0.45
MEDIAPIPE_MIN_TRACK_CONF = 0.45
FACE_ALIGNMENT_USE_CUDA_IF_AVAILABLE = True
HAAR_CASCADE_NAME = "haarcascade_frontalface_default.xml"

# Score weights (best face selection)
WEIGHT_AREA = 0.30
WEIGHT_CENTER = 0.20
WEIGHT_POSE = 0.30
WEIGHT_SHARPNESS = 0.10
WEIGHT_CONFIDENCE = 0.10

# Quality thresholds
MIN_FACE_AREA_RATIO = 0.04
MAX_CENTER_OFFSET_RATIO = 0.35
MIN_BLUR_SCORE = 60.0
MIN_BRIGHTNESS = 45.0
MAX_BRIGHTNESS = 210.0
MIN_FACE_PIXELS = 90
MIN_QUALITY_SCORE = 0.50

# Pose thresholds
MAX_ABS_YAW = 20.0
MAX_ABS_PITCH = 20.0
MAX_ABS_ROLL = 15.0

# Crop/align
CROP_EXPAND_RATIO = 1.25
OUTPUT_FACE_SIZE = (512, 512)  # (w, h)


@dataclass
class FaceCandidate:
    backend: str
    bbox_xywh: tuple[float, float, float, float]
    confidence: float
    landmarks_2d: np.ndarray | None = None  # Nx2 in pixel coords
    landmark_format: str = "none"
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    area_ratio: float = 0.0
    center_offset_ratio: float = 1.0
    score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityResult:
    accepted: bool
    reason: str
    score: float = 0.0
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    face_area_ratio: float = 0.0
    center_offset_ratio: float = 1.0
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


def setup_runtime_dirs() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


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
    camera_matrix = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(model_points, points_2d.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat([rot_mat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(v) for v in euler.flatten()]
    return True, _fold_angle_90(yaw), _fold_angle_90(pitch), _fold_angle_90(roll)


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={CAMERA_ID}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def _safe_crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(max(0, math.floor(x)))
    y1 = int(max(0, math.floor(y)))
    x2 = int(min(w, math.ceil(x + bw)))
    y2 = int(min(h, math.ceil(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _compute_basic_face_metrics(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    area_ratio = float((bw * bh) / max(w * h, 1))
    cx, cy = x + bw * 0.5, y + bh * 0.5
    diag_half = 0.5 * math.sqrt(w * w + h * h)
    center_offset = float(math.sqrt((cx - w * 0.5) ** 2 + (cy - h * 0.5) ** 2) / max(diag_half, 1e-6))

    crop = _safe_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return area_ratio, center_offset, 0.0, 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    b_mean = float(np.mean(gray))
    b_std = float(np.std(gray))
    return area_ratio, center_offset, blur, b_mean, b_std


def init_detector_backend() -> dict[str, Any]:
    if PREFER_MEDIAPIPE and MEDIAPIPE_AVAILABLE:
        try:
            face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=MEDIAPIPE_MIN_DET_CONF,
            )
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=MEDIAPIPE_MAX_NUM_FACES,
                refine_landmarks=False,
                min_detection_confidence=MEDIAPIPE_MIN_DET_CONF,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACK_CONF,
            )
            print("[INFO] detector backend=mediapipe")
            return {"name": "mediapipe", "face_det": face_det, "face_mesh": face_mesh}
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
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.THREE_D,
                device=device,
                face_detector="sfd",
                flip_input=False,
                verbose=False,
            )
            print(f"[INFO] detector backend=face_alignment ({device})")
            return {"name": "face_alignment", "fa": fa}
        except Exception as e:
            print(f"[WARN] face_alignment init failed: {e}")

    cascade_path = cv2.data.haarcascades + HAAR_CASCADE_NAME
    haar = cv2.CascadeClassifier(cascade_path)
    if haar.empty():
        print("[WARN] Haar cascade init failed. face detection unavailable.")
        return {"name": "none"}
    print("[WARN] detector backend=opencv_haar (low accuracy fallback)")
    return {"name": "opencv_haar", "haar": haar}


def _mediapipe_faces(frame: np.ndarray, backend: dict[str, Any]) -> list[FaceCandidate]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_res = backend["face_det"].process(rgb)
    mesh_res = backend["face_mesh"].process(rgb)

    mesh_faces = mesh_res.multi_face_landmarks if mesh_res is not None else None
    mesh_centers: list[tuple[float, float, Any]] = []
    if mesh_faces:
        for fm in mesh_faces:
            xs = np.array([lm.x for lm in fm.landmark], dtype=np.float64) * w
            ys = np.array([lm.y for lm in fm.landmark], dtype=np.float64) * h
            mesh_centers.append((float(np.mean(xs)), float(np.mean(ys)), fm))

    out: list[FaceCandidate] = []
    detections = det_res.detections if det_res is not None and det_res.detections else []
    for det in detections:
        rb = det.location_data.relative_bounding_box
        x = float(rb.xmin * w)
        y = float(rb.ymin * h)
        bw = float(rb.width * w)
        bh = float(rb.height * h)
        conf = float(det.score[0]) if det.score else 0.0

        # match nearest face mesh
        lm2d = None
        yaw = pitch = roll = None
        if mesh_centers:
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            nearest = min(mesh_centers, key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2)
            fm = nearest[2]
            lm2d = np.array([[lm.x * w, lm.y * h] for lm in fm.landmark], dtype=np.float64)
            idxs = [1, 152, 33, 263, 61, 291]
            p2d = np.array([[fm.landmark[i].x * w, fm.landmark[i].y * h] for i in idxs], dtype=np.float64)
            ok, yaw_v, pitch_v, roll_v = _estimate_pose_from_2d_points(p2d, w, h)
            if ok:
                yaw, pitch, roll = yaw_v, pitch_v, roll_v

        area_ratio, center_offset, blur, b_mean, b_std = _compute_basic_face_metrics(frame, (x, y, bw, bh))
        out.append(
            FaceCandidate(
                backend="mediapipe",
                bbox_xywh=(x, y, bw, bh),
                confidence=conf,
                landmarks_2d=lm2d,
                landmark_format="mediapipe468" if lm2d is not None else "none",
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                blur_score=blur,
                brightness_mean=b_mean,
                brightness_std=b_std,
                area_ratio=area_ratio,
                center_offset_ratio=center_offset,
            )
        )
    return out


def _face_alignment_faces(frame: np.ndarray, backend: dict[str, Any]) -> list[FaceCandidate]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lms = backend["fa"].get_landmarks_from_image(rgb)
    if lms is None:
        return []
    out: list[FaceCandidate] = []
    for lm in lms:
        lm = np.asarray(lm, dtype=np.float64)
        x1, y1 = float(np.min(lm[:, 0])), float(np.min(lm[:, 1]))
        x2, y2 = float(np.max(lm[:, 0])), float(np.max(lm[:, 1]))
        bw, bh = float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))
        p2d = np.array([lm[30, :2], lm[8, :2], lm[36, :2], lm[45, :2], lm[48, :2], lm[54, :2]], dtype=np.float64)
        ok, yaw, pitch, roll = _estimate_pose_from_2d_points(p2d, w, h)
        yaw_v = yaw if ok else None
        pitch_v = pitch if ok else None
        roll_v = roll if ok else None
        area_ratio, center_offset, blur, b_mean, b_std = _compute_basic_face_metrics(frame, (x1, y1, bw, bh))
        out.append(
            FaceCandidate(
                backend="face_alignment",
                bbox_xywh=(x1, y1, bw, bh),
                confidence=0.9,
                landmarks_2d=lm[:, :2].copy(),
                landmark_format="fa68",
                yaw=yaw_v,
                pitch=pitch_v,
                roll=roll_v,
                blur_score=blur,
                brightness_mean=b_mean,
                brightness_std=b_std,
                area_ratio=area_ratio,
                center_offset_ratio=center_offset,
            )
        )
    return out


def _haar_faces(frame: np.ndarray, backend: dict[str, Any]) -> list[FaceCandidate]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = backend["haar"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    out: list[FaceCandidate] = []
    for (x, y, bw, bh) in rects:
        area_ratio, center_offset, blur, b_mean, b_std = _compute_basic_face_metrics(frame, (float(x), float(y), float(bw), float(bh)))
        out.append(
            FaceCandidate(
                backend="opencv_haar",
                bbox_xywh=(float(x), float(y), float(bw), float(bh)),
                confidence=0.5,
                landmarks_2d=None,
                landmark_format="none",
                yaw=None,
                pitch=None,
                roll=None,
                blur_score=blur,
                brightness_mean=b_mean,
                brightness_std=b_std,
                area_ratio=area_ratio,
                center_offset_ratio=center_offset,
            )
        )
    return out


def detect_faces(frame: np.ndarray, backend: dict[str, Any]) -> list[FaceCandidate]:
    name = backend.get("name", "none")
    if name == "mediapipe":
        return _mediapipe_faces(frame, backend)
    if name == "face_alignment":
        return _face_alignment_faces(frame, backend)
    if name == "opencv_haar":
        return _haar_faces(frame, backend)
    return []


def _pose_score(face: FaceCandidate) -> float:
    if face.yaw is None or face.pitch is None or face.roll is None:
        return 0.5
    yaw_s = max(0.0, 1.0 - abs(face.yaw) / max(MAX_ABS_YAW, 1e-6))
    pitch_s = max(0.0, 1.0 - abs(face.pitch) / max(MAX_ABS_PITCH, 1e-6))
    roll_s = max(0.0, 1.0 - abs(face.roll) / max(MAX_ABS_ROLL, 1e-6))
    return float((yaw_s + pitch_s + roll_s) / 3.0)


def select_best_face(frame: np.ndarray, faces: list[FaceCandidate]) -> FaceCandidate | None:
    if not faces:
        return None

    for face in faces:
        area_score = min(1.0, face.area_ratio / max(MIN_FACE_AREA_RATIO * 2.0, 1e-6))
        center_score = max(0.0, 1.0 - face.center_offset_ratio / max(MAX_CENTER_OFFSET_RATIO * 1.5, 1e-6))
        pose_score = _pose_score(face)
        sharpness_score = min(1.0, face.blur_score / 200.0)
        confidence_score = float(np.clip(face.confidence, 0.0, 1.0))
        face.score = (
            WEIGHT_AREA * area_score
            + WEIGHT_CENTER * center_score
            + WEIGHT_POSE * pose_score
            + WEIGHT_SHARPNESS * sharpness_score
            + WEIGHT_CONFIDENCE * confidence_score
        )
    return max(faces, key=lambda f: f.score)


def validate_face_quality(frame: np.ndarray, face: FaceCandidate | None) -> QualityResult:
    if face is None:
        return QualityResult(False, "no_face")

    x, y, bw, bh = face.bbox_xywh
    h, w = frame.shape[:2]
    if bw < MIN_FACE_PIXELS or bh < MIN_FACE_PIXELS:
        return QualityResult(False, "face_too_small", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if x < 0 or y < 0 or (x + bw) > w or (y + bh) > h:
        return QualityResult(False, "bbox_out_of_bounds", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.area_ratio < MIN_FACE_AREA_RATIO:
        return QualityResult(False, "face_area_too_small", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.center_offset_ratio > MAX_CENTER_OFFSET_RATIO:
        return QualityResult(False, "face_off_center", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.blur_score < MIN_BLUR_SCORE:
        return QualityResult(False, "too_blurry", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.brightness_mean < MIN_BRIGHTNESS:
        return QualityResult(False, "too_dark", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.brightness_mean > MAX_BRIGHTNESS:
        return QualityResult(False, "too_bright", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.yaw is not None and abs(face.yaw) > MAX_ABS_YAW:
        return QualityResult(False, "yaw_too_large", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.pitch is not None and abs(face.pitch) > MAX_ABS_PITCH:
        return QualityResult(False, "pitch_too_large", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.roll is not None and abs(face.roll) > MAX_ABS_ROLL:
        return QualityResult(False, "roll_too_large", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)
    if face.score < MIN_QUALITY_SCORE:
        return QualityResult(False, "quality_score_too_low", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)

    return QualityResult(True, "ok", score=face.score, blur_score=face.blur_score, brightness_mean=face.brightness_mean, face_area_ratio=face.area_ratio, center_offset_ratio=face.center_offset_ratio, yaw=face.yaw, pitch=face.pitch, roll=face.roll)


def _eye_points(face: FaceCandidate) -> tuple[np.ndarray, np.ndarray] | None:
    if face.landmarks_2d is None:
        return None
    if face.landmark_format == "mediapipe468":
        left = face.landmarks_2d[33]
        right = face.landmarks_2d[263]
        return left, right
    if face.landmark_format == "fa68":
        left = face.landmarks_2d[36]
        right = face.landmarks_2d[45]
        return left, right
    return None


def align_and_crop_face(frame: np.ndarray, face: FaceCandidate) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = face.bbox_xywh
    work = frame
    M = None

    eyes = _eye_points(face)
    if eyes is not None:
        left, right = eyes
        dy = float(right[1] - left[1])
        dx = float(right[0] - left[0])
        angle = math.degrees(math.atan2(dy, dx))
        eye_center = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
        M = cv2.getRotationMatrix2D((float(eye_center[0]), float(eye_center[1])), angle, 1.0)
        work = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # bbox in rotated frame
    corners = np.array(
        [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
        dtype=np.float64,
    )
    if M is not None:
        ones = np.ones((4, 1), dtype=np.float64)
        corners_h = np.hstack([corners, ones])
        corners = (M @ corners_h.T).T

    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    bw2 = max(1.0, x2 - x1)
    bh2 = max(1.0, y2 - y1)
    cx, cy = x1 + 0.5 * bw2, y1 + 0.5 * bh2

    ext_w = bw2 * CROP_EXPAND_RATIO
    ext_h = bh2 * CROP_EXPAND_RATIO
    rx1 = int(max(0, math.floor(cx - ext_w * 0.5)))
    ry1 = int(max(0, math.floor(cy - ext_h * 0.5)))
    rx2 = int(min(w, math.ceil(cx + ext_w * 0.5)))
    ry2 = int(min(h, math.ceil(cy + ext_h * 0.5)))
    if rx2 <= rx1 or ry2 <= ry1:
        raise RuntimeError("Invalid crop region after alignment.")

    crop = work[ry1:ry2, rx1:rx2]
    out = cv2.resize(crop, OUTPUT_FACE_SIZE, interpolation=cv2.INTER_LINEAR)
    return out


def save_latest(face_img: np.ndarray, full_frame: np.ndarray, metadata: dict[str, Any]) -> None:
    cv2.imwrite(str(LATEST_INPUT_PATH), face_img, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
    cv2.imwrite(str(LATEST_FULL_FRAME_PATH), full_frame, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
    LATEST_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_history(face_img: np.ndarray, metadata: dict[str, Any], full_frame: np.ndarray | None = None) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = HISTORY_DIR / f"input_{ts}.jpg"
    meta_path = HISTORY_DIR / f"metadata_{ts}.json"
    cv2.imwrite(str(img_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
    if full_frame is not None:
        cv2.imwrite(str(HISTORY_DIR / f"full_{ts}.jpg"), full_frame, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
    metadata = dict(metadata)
    metadata["history_image_path"] = str(img_path.as_posix())
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def on_new_valid_face(image_path: Path, metadata: dict[str, Any]) -> None:
    # Placeholder callback. Keep light/non-blocking.
    _ = image_path, metadata


def _draw_overlay(
    frame: np.ndarray,
    faces: list[FaceCandidate],
    best_face: FaceCandidate | None,
    quality: QualityResult | None,
) -> np.ndarray:
    vis = frame.copy()
    for f in faces:
        x, y, bw, bh = [int(v) for v in f.bbox_xywh]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
    if best_face is not None:
        x, y, bw, bh = [int(v) for v in best_face.bbox_xywh]
        ok = quality.accepted if quality is not None else False
        color = (0, 220, 0) if ok else (0, 0, 220)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(vis, f"score={best_face.score:.3f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def _put_status_text(
    frame: np.ndarray,
    face_count: int,
    quality: QualityResult | None,
    backend_name: str,
) -> None:
    cv2.putText(frame, f"backend={backend_name} faces={face_count}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if quality is None:
        msg = "status=waiting"
    else:
        msg = f"status={'ACCEPT' if quality.accepted else 'REJECT'} reason={quality.reason}"
    cv2.putText(frame, msg[:100], (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if quality is not None:
        pose_txt = f"yaw={quality.yaw} pitch={quality.pitch} roll={quality.roll}"
        q_txt = f"blur={quality.blur_score:.1f} bright={quality.brightness_mean:.1f} score={quality.score:.3f}"
        cv2.putText(frame, pose_txt[:90], (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, q_txt[:90], (15, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "keys: q quit | s save snapshot", (15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _run_live_inference_if_enabled(image_path: Path, cooldown_state: dict[str, float]) -> None:
    if not ENABLE_LIVE_INFERENCE:
        return
    now = time.time()
    if now - cooldown_state.get("last_ts", 0.0) < LIVE_INFERENCE_MIN_INTERVAL_SEC:
        return
    cooldown_state["last_ts"] = now
    try:
        from run_human_expression_inference import predict_motor_from_image

        motor = predict_motor_from_image(image_path=image_path)
        print(f"[INFER] motor30 first6={motor[:6]} ...")
    except Exception as e:
        print(f"[WARN] live inference failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human camera capture with best-face selection and quality filter.")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window.")
    args = parser.parse_args()
    preview = PREVIEW and (not args.no_preview)

    setup_runtime_dirs()
    backend = init_detector_backend()
    backend_name = backend.get("name", "none")

    cap = open_camera()
    for _ in range(max(0, CAMERA_WARMUP_FRAMES)):
        cap.read()

    print("[INFO] running. press q to quit, s to snapshot.")
    frame_idx = 0
    processed_idx = 0
    last_process_ts = 0.0
    last_faces: list[FaceCandidate] = []
    last_best: FaceCandidate | None = None
    last_quality: QualityResult | None = None
    latest_face_img: np.ndarray | None = None
    latest_metadata: dict[str, Any] | None = None
    infer_state = {"last_ts": 0.0}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] frame read failed, continue...")
                continue
            frame_idx += 1
            now = time.time()

            should_process = (
                (frame_idx % max(1, PROCESS_EVERY_N_FRAMES) == 0)
                and (now - last_process_ts >= PROCESS_INTERVAL_SEC)
            )
            if should_process:
                last_process_ts = now
                processed_idx += 1
                faces = detect_faces(frame, backend=backend)
                best = select_best_face(frame, faces)
                quality = validate_face_quality(frame, best)

                last_faces = faces
                last_best = best
                last_quality = quality

                if best is not None and quality.accepted:
                    aligned = align_and_crop_face(frame, best)
                    ts = datetime.now().isoformat(timespec="milliseconds")
                    metadata = {
                        "timestamp": ts,
                        "image_path": str(LATEST_INPUT_PATH.as_posix()),
                        "full_frame_path": str(LATEST_FULL_FRAME_PATH.as_posix()),
                        "bbox": [float(v) for v in best.bbox_xywh],
                        "quality": asdict(quality),
                        "pose": {"yaw": best.yaw, "pitch": best.pitch, "roll": best.roll},
                        "selected_from_faces": int(len(faces)),
                        "face_score": float(best.score),
                        "backend": backend_name,
                    }
                    save_latest(aligned, frame, metadata)
                    if SAVE_DEBUG_FRAMES and SAVE_HISTORY_EVERY_VALID_FRAME:
                        save_history(aligned, metadata, full_frame=frame)
                    latest_face_img = aligned
                    latest_metadata = metadata
                    on_new_valid_face(LATEST_INPUT_PATH, metadata)
                    _run_live_inference_if_enabled(LATEST_INPUT_PATH, cooldown_state=infer_state)

                print(
                    f"[FRAME {frame_idx:06d}] processed={processed_idx} "
                    f"faces={len(faces)} best_score={best.score if best else None} "
                    f"status={'ACCEPT' if quality.accepted else 'REJECT'} reason={quality.reason}"
                )

            if preview:
                vis = _draw_overlay(frame, last_faces, last_best, last_quality)
                _put_status_text(vis, len(last_faces), last_quality, backend_name)
                cv2.imshow("human_camera_capture", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if latest_face_img is not None and latest_metadata is not None:
                        save_history(latest_face_img, latest_metadata, full_frame=frame)
                        print("[INFO] manual snapshot saved to history.")
                    else:
                        print("[INFO] no valid face cached yet.")
            else:
                # Keep CPU usage reasonable if preview disabled.
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if backend_name == "mediapipe":
            try:
                backend["face_det"].close()
            except Exception:
                pass
            try:
                backend["face_mesh"].close()
            except Exception:
                pass

    print("[DONE] human_camera_capture stopped.")


if __name__ == "__main__":
    main()

