#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


# =========================
# Config (edit here)
# =========================
MODEL_PATH = Path("./checkpoints/b1vnext_rel/best.pt")
INPUT_MODE = "REL193"  # REL190 / REL193 / FEATURE385
SUBJECT_NEUTRAL_PATH = Path("./runtime/subject_profiles/default/neutral_abs.json")
OUTPUT_JSON = Path("./runtime/latest_motor_prediction.json")
DEVICE = "cuda"
ENABLE_ROBOT_CONTROL = False

# Placeholder dimensions (replace after real feature pipeline is connected)
FEATURE_DIM_BY_MODE = {
    "REL190": 190,
    "REL193": 193,
    "FEATURE385": 385,
}
MOTOR_DIM = 30


@dataclass
class LoadedModel:
    model_obj: Any
    model_path: Path
    device: str
    is_placeholder: bool


# Cache for repeated live inference use
_MODEL_CACHE: dict[tuple[str, str], LoadedModel] = {}


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"


def load_model(model_path: Path, device: str) -> LoadedModel:
    """
    Placeholder model loader.
    TODO:
      - Replace with real B1vnext-REL model class construction.
      - Load checkpoint weights and set eval().
    """
    key = (str(model_path.resolve()), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_obj: Any = None
    is_placeholder = True
    if TORCH_AVAILABLE and model_path.exists():
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model_obj = {"checkpoint": ckpt}
            is_placeholder = True  # still placeholder inference path
            print(f"[INFO] loaded checkpoint file: {model_path}")
        except Exception as e:
            print(f"[WARN] failed to load checkpoint: {e}. fallback to placeholder model.")
            model_obj = None
    else:
        if not model_path.exists():
            print(f"[WARN] model file not found: {model_path}. using placeholder model.")
        elif not TORCH_AVAILABLE:
            print("[WARN] torch not available. using placeholder model.")

    loaded = LoadedModel(model_obj=model_obj, model_path=model_path, device=device, is_placeholder=is_placeholder)
    _MODEL_CACHE[key] = loaded
    return loaded


def extract_abs_features_from_image(image_path: Path) -> np.ndarray:
    """
    Placeholder ABS extractor.
    TODO:
      - Connect existing ABS extraction pipeline/scripts.
      - Return full ABS vector used by training input builder.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lightweight synthetic ABS proxy for pipeline debugging
    h, w = gray.shape[:2]
    mean = float(np.mean(gray)) / 255.0
    std = float(np.std(gray)) / 255.0
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_n = min(1.0, lap / 400.0)
    cx = 0.5
    cy = 0.5
    arr = np.array(
        [
            mean,
            std,
            lap_n,
            w / 1920.0,
            h / 1080.0,
            cx,
            cy,
            mean * std,
            std * lap_n,
            mean * lap_n,
            mean - std,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return arr


def load_subject_neutral(neutral_path: Path) -> dict[str, Any] | None:
    """
    Load subject-specific neutral profile.
    Expected future schema can include ABS baseline vectors.
    """
    if not neutral_path.exists():
        print(f"[WARN] neutral file not found: {neutral_path}. REL mode will use zeros baseline.")
        return None
    try:
        return json.loads(neutral_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to parse neutral file: {e}. ignore neutral.")
        return None


def _neutral_abs_vector(neutral_obj: dict[str, Any] | None, dim: int) -> np.ndarray:
    if neutral_obj is None:
        return np.zeros(dim, dtype=np.float32)
    for key in ("abs_vector", "ABS", "abs"):
        if key in neutral_obj and isinstance(neutral_obj[key], list):
            arr = np.asarray(neutral_obj[key], dtype=np.float32)
            if arr.size == dim:
                return arr
    # fallback: zero baseline
    return np.zeros(dim, dtype=np.float32)


def build_rel_features(abs_expr: np.ndarray, abs_neutral: np.ndarray, input_mode: str) -> np.ndarray:
    """
    Build model input vector by mode.
    This is placeholder mapping for end-to-end pipeline test.
    Replace with your exact feature-building logic later.
    """
    if abs_expr.ndim != 1:
        raise RuntimeError("abs_expr must be 1D.")
    if abs_neutral.shape != abs_expr.shape:
        raise RuntimeError("abs_neutral shape mismatch.")

    target_dim = FEATURE_DIM_BY_MODE.get(input_mode)
    if target_dim is None:
        raise RuntimeError(f"unsupported input_mode: {input_mode}")

    rel = abs_expr - abs_neutral
    if input_mode == "REL190":
        base = rel
    elif input_mode == "REL193":
        # Example append pose-like placeholders (3 dims)
        base = np.concatenate([rel, np.array([0.0, 0.0, 0.0], dtype=np.float32)], axis=0)
    elif input_mode == "FEATURE385":
        # Placeholder expansion; replace with true FEATURE385 builder
        rep = max(1, math_ceil_div(target_dim, rel.size))
        base = np.tile(rel, rep)[:target_dim]
    else:
        raise RuntimeError(f"unsupported input_mode: {input_mode}")

    # pad/truncate to target_dim for stable shape
    if base.size < target_dim:
        base = np.pad(base, (0, target_dim - base.size), mode="constant", constant_values=0.0)
    elif base.size > target_dim:
        base = base[:target_dim]
    return base.astype(np.float32)


def math_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def predict_motor(model: LoadedModel, feature_vec: np.ndarray, device: str) -> list[float]:
    """
    Placeholder prediction.
    TODO:
      - Convert feature_vec to torch tensor
      - Run real model forward
      - Post-process to valid motor30 range
    """
    if feature_vec.ndim != 1:
        raise RuntimeError("feature_vec must be 1D.")

    # Deterministic placeholder around neutral motor profile
    neutral = np.array(
        [
            0.42, 0.42, 0.40, 0.40,
            0.45, 0.45, 0.50, 0.50, 0.50, 0.50,
            0.50, 0.50, 0.50, 0.45, 0.45,
            0.40, 0.40, 0.40, 0.40, 0.42, 0.42, 0.40, 0.40, 0.45, 0.42, 0.42, 0.42,
            0.50, 0.50,
            0.20,
        ],
        dtype=np.float32,
    )
    seed = float(np.mean(feature_vec) + np.std(feature_vec) * 0.5)
    delta = (np.sin(np.linspace(0.0, 2.8, MOTOR_DIM) + seed * 3.0) * 0.08).astype(np.float32)
    motor = np.clip(neutral + delta, 0.0, 1.0)
    return [float(round(v, 6)) for v in motor.tolist()]


def save_prediction(output_path: Path, image_path: Path, model_path: Path, input_mode: str, motor: list[float], extra_info: dict[str, Any] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "image_path": str(image_path.as_posix()),
        "model_path": str(model_path.as_posix()),
        "input_mode": input_mode,
        "motor": [float(v) for v in motor],
        "motor_dim": int(len(motor)),
        "status": "ok",
    }
    if extra_info:
        obj["extra_info"] = extra_info
    output_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_error(output_path: Path, image_path: Path | None, reason: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "image_path": str(image_path.as_posix()) if image_path else None,
        "status": "error",
        "reason": reason,
    }
    output_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def send_motor_to_robot(motor: list[float]) -> None:
    # TODO: Connect real robot communication here.
    _ = motor
    print("[ROBOT] send_motor_to_robot placeholder (not sending).")


def predict_motor_from_image(
    image_path: Path,
    model_path: Path = MODEL_PATH,
    neutral_path: Path = SUBJECT_NEUTRAL_PATH,
    input_mode: str = INPUT_MODE,
    device: str = DEVICE,
) -> list[float]:
    resolved_device = resolve_device(device)
    model = load_model(model_path=model_path, device=resolved_device)
    abs_expr = extract_abs_features_from_image(image_path=image_path)
    neutral_obj = load_subject_neutral(neutral_path=neutral_path)
    abs_neutral = _neutral_abs_vector(neutral_obj, dim=abs_expr.size)
    feature_vec = build_rel_features(abs_expr=abs_expr, abs_neutral=abs_neutral, input_mode=input_mode)
    motor = predict_motor(model=model, feature_vec=feature_vec, device=resolved_device)
    return motor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human face image inference to motor30.")
    parser.add_argument("--image", type=Path, required=True, help="Input face image path.")
    parser.add_argument("--neutral", type=Path, default=SUBJECT_NEUTRAL_PATH, help="Subject neutral profile json path.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Model checkpoint path.")
    parser.add_argument("--device", type=str, default=DEVICE, help="cpu or cuda")
    parser.add_argument("--input-mode", type=str, default=INPUT_MODE, choices=["REL190", "REL193", "FEATURE385"])
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output prediction json path.")
    parser.add_argument("--send-robot", action="store_true", help="If set, call send_motor_to_robot after prediction.")
    args = parser.parse_args()

    t0 = time.time()
    try:
        if not args.image.exists():
            raise RuntimeError(f"image not found: {args.image}")

        motor = predict_motor_from_image(
            image_path=args.image,
            model_path=args.model,
            neutral_path=args.neutral,
            input_mode=args.input_mode,
            device=args.device,
        )
        extra = {
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "device_resolved": resolve_device(args.device),
            "note": "placeholder inference path; replace with real model forward.",
        }
        save_prediction(
            output_path=args.output,
            image_path=args.image,
            model_path=args.model,
            input_mode=args.input_mode,
            motor=motor,
            extra_info=extra,
        )
        print(f"[DONE] motor30 predicted. first6={motor[:6]} ...")
        print(f"[DONE] output={args.output}")

        if ENABLE_ROBOT_CONTROL or args.send_robot:
            send_motor_to_robot(motor)

    except Exception as e:
        save_error(output_path=args.output, image_path=args.image if "args" in locals() else None, reason=str(e))
        print(f"[ERROR] inference failed: {e}")


if __name__ == "__main__":
    main()

