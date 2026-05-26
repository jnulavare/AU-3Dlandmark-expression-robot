#!/usr/bin/env python3
"""Human image pair -> Adapter -> FEATURE385 -> B1vnext motor30 prediction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
import yaml

try:
    from pipeline.human_robot_adapter import (
        ABS_DIM_DEFAULT,
        FEATURE_DIM_TOTAL_DEFAULT,
        POSE_DIM_DEFAULT,
        REL_DIM_DEFAULT,
        HumanToRobotFeatureAdapter,
    )
    from motor_regression_baseline.model import MotorRegressorMLP
    from motor_regression_baseline.run_utils import select_eval_state_dict
except ModuleNotFoundError:
    # Fallback for direct script execution:
    # python au-landmark/pipeline/predict_human_to_motor.py
    project_dir = Path(__file__).resolve().parent.parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from pipeline.human_robot_adapter import (
        ABS_DIM_DEFAULT,
        FEATURE_DIM_TOTAL_DEFAULT,
        POSE_DIM_DEFAULT,
        REL_DIM_DEFAULT,
        HumanToRobotFeatureAdapter,
    )
    from motor_regression_baseline.model import MotorRegressorMLP
    from motor_regression_baseline.run_utils import select_eval_state_dict


# Default paths (edit these directly for quick testing).
HUMAN_NEUTRAL_IMAGE = r"D:/code/AU+landmark/au-landmark/assets/test1.jpg"
HUMAN_EXPR_IMAGE = r"D:/code/AU+landmark/au-landmark/assets/test2.png"
ROBOT_NEUTRAL_FEATURE = r"D:/code/AU+landmark/au-landmark/assets/robot_neutral_feature.json"
FEATURE_NORMALIZER_PATH = r""
B1VNEXT_CONFIG = r"D:/code/AU+landmark/au-landmark/motor_regression_baseline/configs/baseline.yaml"
B1VNEXT_CKPT = r"D:/code/AU+landmark/dataset/motor_regression_compare6/run_002/best.pt"
OUTPUT_DIR = r"D:/code/AU+landmark/au-landmark/output"
OUTPUT_JSON = "motor30_prediction.json"
OUTPUT_CSV = "motor30_prediction.csv"


MOTOR_NAMES: List[str] = [
    "Brow Inner Left",
    "Brow Inner Right",
    "Brow Outer Left",
    "Brow Outer Right",
    "Eyelid Lower Left",
    "Eyelid Lower Right",
    "Eyelid Upper Left",
    "Eyelid Upper Right",
    "Gaze Target Phi",
    "Gaze Target Theta",
    "Head Pitch",
    "Head Roll",
    "Head Yaw",
    "Jaw Pitch",
    "Jaw Yaw",
    "Lip Bottom Curl",
    "Lip Bottom Depress Left",
    "Lip Bottom Depress Middle",
    "Lip Bottom Depress Right",
    "Lip Corner Raise Left",
    "Lip Corner Raise Right",
    "Lip Corner Stretch Left",
    "Lip Corner Stretch Right",
    "Lip Top Curl",
    "Lip Top Raise Left",
    "Lip Top Raise Middle",
    "Lip Top Raise Right",
    "Neck Pitch",
    "Neck Roll",
    "Nose Wrinkle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human->Robot adapter + B1vnext inference")
    p.add_argument("--human_neutral", type=str, default=HUMAN_NEUTRAL_IMAGE)
    p.add_argument("--human_expr", type=str, default=HUMAN_EXPR_IMAGE)
    p.add_argument("--robot_neutral", type=str, default=ROBOT_NEUTRAL_FEATURE)
    p.add_argument("--normalizer", type=str, default=FEATURE_NORMALIZER_PATH)
    p.add_argument("--config", type=str, default=B1VNEXT_CONFIG)
    p.add_argument("--ckpt", type=str, default=B1VNEXT_CKPT)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def resolve_device(requested: str, cfg: Mapping[str, Any]) -> torch.device:
    # Adaptive strategy:
    # 1) Explicit CLI --device has highest priority.
    # 2) Otherwise follow config train.device.
    # 3) If CUDA requested but unavailable, fallback to CPU.
    if requested:
        requested = requested.strip().lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        if requested.startswith("cuda"):
            return torch.device("cpu")
    cfg_device = str(cfg.get("train", {}).get("device", "cuda")).strip().lower()
    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_feature385_to_model_inputs(feature385: np.ndarray) -> Dict[str, np.ndarray]:
    f = np.asarray(feature385, dtype=np.float32).reshape(-1)
    if f.shape[0] != FEATURE_DIM_TOTAL_DEFAULT:
        raise ValueError(f"feature dim mismatch: {f.shape[0]} vs {FEATURE_DIM_TOTAL_DEFAULT}")
    return {
        "brow_abs": f[0:37],
        "brow_rel": f[37:74],
        "eye_abs": f[74:119],
        "eye_rel": f[119:164],
        "mouth_abs": f[164:243],
        "mouth_rel": f[243:322],
        "jaw_abs": f[322:350],
        "jaw_rel": f[350:378],
        "global_abs": f[378:381],
        "global_rel": f[381:382],
        "pose": f[382:385],
    }


def build_model_input_tensor(feature385: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    parts = split_feature385_to_model_inputs(feature385)
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in parts.items()}


def load_model(cfg: Mapping[str, Any], ckpt_path: Path, device: torch.device) -> MotorRegressorMLP:
    model = MotorRegressorMLP().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_ema = bool(cfg.get("eval", {}).get("use_ema", True))
    state = select_eval_state_dict(ckpt, use_ema=use_ema)
    model.load_state_dict(state)
    model.eval()
    return model


def validate_compare6_shape(model: MotorRegressorMLP) -> None:
    # Ensure we are using compare6-style architecture.
    if int(model.brow_residual_head[0].in_features) != 102:
        raise RuntimeError("Loaded model is not compare6 brow residual shape (expected 102).")
    if int(model.mouth_residual_head[0].in_features) != 108:
        raise RuntimeError("Loaded model is not compare6 mouth residual shape (expected 108).")


def write_csv(path: Path, pred_raw: np.ndarray, pred_clipped: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["motor_idx", "motor_name", "value_raw", "value_clipped"])
        for i, name in enumerate(MOTOR_NAMES):
            writer.writerow([i, name, float(pred_raw[i]), float(pred_clipped[i])])


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device, cfg)

    adapter = HumanToRobotFeatureAdapter(
        robot_neutral_feature_path=args.robot_neutral,
        feature_normalizer_path=(args.normalizer if args.normalizer else None),
        feature_dim_total=FEATURE_DIM_TOTAL_DEFAULT,
        abs_dim=ABS_DIM_DEFAULT,
        rel_dim=REL_DIM_DEFAULT,
        pose_dim=POSE_DIM_DEFAULT,
        device=str(device),
        config=cfg,
    )

    try:
        adapted = adapter.adapt(args.human_neutral, args.human_expr)
    except NotImplementedError as exc:
        print("[ERROR] Adapter feature extractor is not connected.")
        print(str(exc))
        print("Please implement HumanToRobotFeatureAdapter.extract_face_feature().")
        return

    feature385 = np.asarray(adapted["feature385"], dtype=np.float32).reshape(-1)
    if feature385.shape[0] != FEATURE_DIM_TOTAL_DEFAULT:
        raise ValueError(f"adapted feature dim mismatch: {feature385.shape[0]} vs {FEATURE_DIM_TOTAL_DEFAULT}")

    # IMPORTANT:
    # The adapted FEATURE385 must follow the same normalization convention
    # as the training FEATURE385 used by B1vnext.
    # compare6 uses prebuilt FEATURE385 directly in data_utils (no extra runtime transform),
    # so this script currently passes adapter output as-is.
    x = build_model_input_tensor(feature385, device=device)

    model = load_model(cfg, ckpt_path=ckpt_path, device=device)
    validate_compare6_shape(model)

    with torch.no_grad():
        pred_out = model(x)
    if isinstance(pred_out, tuple):
        pred_out = pred_out[0]
    pred = pred_out.squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred_clipped = np.clip(pred, 0.0, 1.0).astype(np.float32)

    # Save adapter outputs.
    np.save(output_dir / "adapted_feature385.npy", feature385)
    np.save(output_dir / "robot_style_abs.npy", np.asarray(adapted["robot_style_abs"], dtype=np.float32))
    np.save(output_dir / "robot_style_rel.npy", np.asarray(adapted["robot_style_rel"], dtype=np.float32))
    (output_dir / "adapted_feature_debug.json").write_text(
        json.dumps(adapted.get("debug", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        "human_neutral_image": str(args.human_neutral),
        "human_expr_image": str(args.human_expr),
        "robot_neutral_feature": str(args.robot_neutral),
        "ckpt": str(ckpt_path),
        "feature_dim": int(feature385.shape[0]),
        "motor_dim": int(pred.shape[0]),
        "motor30_raw": [float(v) for v in pred.tolist()],
        "motor30_clipped": [float(v) for v in pred_clipped.tolist()],
        "motor_names": MOTOR_NAMES,
        "adapter_debug": adapted.get("debug", {}),
    }
    (output_dir / OUTPUT_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / OUTPUT_CSV, pred_raw=pred, pred_clipped=pred_clipped)

    print("Motor30 prediction:")
    for i, name in enumerate(MOTOR_NAMES):
        print(f"[{i:02d}] {name}: raw={pred[i]:.6f} clipped={pred_clipped[i]:.6f}")

    print(f"[DONE] output_dir={output_dir}")
    print(f"[DONE] json={output_dir / OUTPUT_JSON}")
    print(f"[DONE] csv={output_dir / OUTPUT_CSV}")


if __name__ == "__main__":
    main()
