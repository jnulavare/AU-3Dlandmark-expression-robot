#!/usr/bin/env python3
"""Build robot neutral feature JSON from a single robot neutral image.

This script is inference-data preparation only. It does not train or modify B1vnext.
Output format matches HumanToRobotFeatureAdapter expectation:
{
  "feature_dim_total": 385,
  "abs": [...192...],
  "rel": [...190...],   # neutral default zeros
  "pose": [...3...],    # yaw/pitch/roll
  "feature385": [...385...],
  "schema": {"abs_dim": 192, "rel_dim": 190, "pose_dim": 3}
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# Default paths (edit directly if needed).
ROBOT_NEUTRAL_IMAGE = r"D:/code/AU+landmark/au-landmark/assets/ameca_neutral.jpg"
OUTPUT_JSON = r"D:/code/AU+landmark/au-landmark/assets/robot_neutral_feature.json"
OUTPUT_NPY = r""
DEVICE = "cuda"
DETECTOR = "sfd"
TORCH_HOME = r"D:/torch_cache"

ABS_DIM = 192
REL_DIM = 190
POSE_DIM = 3
FEATURE_DIM_TOTAL = 385
FEATURE_LAYOUT = "compare6_interleaved"
FEATURE385_PACK_ORDER = (
    "brow_abs,brow_rel,eye_abs,eye_rel,mouth_abs,mouth_rel,"
    "jaw_abs,jaw_rel,global_abs,global_rel,pose"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robot_neutral_feature.json from one image")
    p.add_argument("--image", type=str, default=ROBOT_NEUTRAL_IMAGE, help="Robot neutral image path")
    p.add_argument("--output-json", type=str, default=OUTPUT_JSON, help="Output robot_neutral_feature.json")
    p.add_argument("--output-npy", type=str, default=OUTPUT_NPY, help="Optional output .npy for feature385")
    p.add_argument("--device", type=str, default=DEVICE, help="cuda or cpu")
    p.add_argument("--detector", type=str, default=DETECTOR, help="face-alignment detector backend")
    p.add_argument("--torch-home", type=str, default=TORCH_HOME, help="Torch cache directory")
    return p.parse_args()


def pack_feature385_interleaved(abs_vec: np.ndarray, rel_vec: np.ndarray, pose_vec: np.ndarray) -> np.ndarray:
    abs_arr = np.asarray(abs_vec, dtype=np.float32).reshape(-1)
    rel_arr = np.asarray(rel_vec, dtype=np.float32).reshape(-1)
    pose_arr = np.asarray(pose_vec, dtype=np.float32).reshape(-1)
    if abs_arr.shape[0] != 192:
        raise ValueError(f"abs_vec must be 192, got {abs_arr.shape[0]}")
    if rel_arr.shape[0] != 190:
        raise ValueError(f"rel_vec must be 190, got {rel_arr.shape[0]}")
    if pose_arr.shape[0] != 3:
        raise ValueError(f"pose_vec must be 3, got {pose_arr.shape[0]}")

    feature385 = np.concatenate(
        [
            abs_arr[0:37],      # brow_abs
            rel_arr[0:37],      # brow_rel
            abs_arr[37:82],     # eye_abs
            rel_arr[37:82],     # eye_rel
            abs_arr[82:161],    # mouth_abs
            rel_arr[82:161],    # mouth_rel
            abs_arr[161:189],   # jaw_abs
            rel_arr[161:189],   # jaw_rel
            abs_arr[189:192],   # global_abs
            rel_arr[189:190],   # global_rel
            pose_arr,           # pose
        ],
        axis=0,
    ).astype(np.float32)
    if feature385.shape[0] != 385:
        raise ValueError(f"feature385 must be 385, got {feature385.shape[0]}")
    return feature385


def validate_feature385_interleaved(
    feature385: np.ndarray,
    abs_vec: np.ndarray,
    rel_vec: np.ndarray,
    pose_vec: np.ndarray,
) -> None:
    f = np.asarray(feature385, dtype=np.float32).reshape(-1)
    a = np.asarray(abs_vec, dtype=np.float32).reshape(-1)
    r = np.asarray(rel_vec, dtype=np.float32).reshape(-1)
    p = np.asarray(pose_vec, dtype=np.float32).reshape(-1)
    if f.shape[0] != 385:
        raise ValueError(f"feature385 must be 385, got {f.shape[0]}")
    if a.shape[0] != 192:
        raise ValueError(f"abs_vec must be 192, got {a.shape[0]}")
    if r.shape[0] != 190:
        raise ValueError(f"rel_vec must be 190, got {r.shape[0]}")
    if p.shape[0] != 3:
        raise ValueError(f"pose_vec must be 3, got {p.shape[0]}")

    checks = [
        np.allclose(f[0:37], a[0:37]),
        np.allclose(f[37:74], r[0:37]),
        np.allclose(f[74:119], a[37:82]),
        np.allclose(f[119:164], r[37:82]),
        np.allclose(f[164:243], a[82:161]),
        np.allclose(f[243:322], r[82:161]),
        np.allclose(f[322:350], a[161:189]),
        np.allclose(f[350:378], r[161:189]),
        np.allclose(f[378:381], a[189:192]),
        np.allclose(f[381:382], r[189:190]),
        np.allclose(f[382:385], p),
    ]
    if not all(checks):
        raise AssertionError("feature385 interleaved layout validation failed")


def _import_abs_extractor(project_dir: Path):
    """Import preprocess.extract_abs_input_vec_gpu with cwd-robust behavior."""
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from preprocess.extract_abs_input_vec_gpu import extract_one  # type: ignore

    return extract_one


def _extract_single_abs_row(
    image_path: Path,
    device: str,
    detector: str,
    torch_home: Path,
) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parent.parent
    extract_one = _import_abs_extractor(project_dir)

    os.environ["TORCH_HOME"] = str(torch_home)
    torch_home.mkdir(parents=True, exist_ok=True)

    import torch  # local import so TORCH_HOME is honored
    import face_alignment as fa

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or enable CUDA.")

    model = fa.FaceAlignment(
        fa.LandmarksType.THREE_D,
        device=device,
        face_detector=detector,
        flip_input=False,
        verbose=False,
    )

    # dataset_root only affects image_path formatting in row.
    dataset_root = image_path.parent
    out = extract_one(
        path=image_path,
        dataset_root=dataset_root,
        model=model,
        tracked_bbox=None,
        tracked_conf=0.0,
    )

    # extract_one has a mixed return contract in early failure branches.
    row: Dict[str, Any]
    if isinstance(out, tuple):
        row = out[0]
    else:
        row = out

    return row


def _row_to_feature(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face_found = int(float(row.get("face_found", 0) or 0))
    if face_found != 1:
        raise RuntimeError(f"No valid face detected on neutral image. error={row.get('error', '')}")

    abs_vec = np.array([float(row.get(f"abs_{i:03d}", 0.0)) for i in range(ABS_DIM)], dtype=np.float32)
    if abs_vec.shape[0] != ABS_DIM:
        raise RuntimeError(f"ABS dim mismatch: got {abs_vec.shape[0]}, expect {ABS_DIM}")

    pose = np.array(
        [
            float(row.get("yaw", 0.0)),
            float(row.get("pitch", 0.0)),
            float(row.get("roll", 0.0)),
        ],
        dtype=np.float32,
    )
    rel_vec = np.zeros((REL_DIM,), dtype=np.float32)
    feature385 = pack_feature385_interleaved(abs_vec, rel_vec, pose)
    validate_feature385_interleaved(feature385, abs_vec, rel_vec, pose)
    if feature385.shape[0] != FEATURE_DIM_TOTAL:
        raise RuntimeError(f"FEATURE385 dim mismatch: got {feature385.shape[0]}, expect {FEATURE_DIM_TOTAL}")

    return abs_vec, rel_vec, pose, feature385


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npy = Path(args.output_npy) if str(args.output_npy).strip() else None
    if out_npy is not None:
        out_npy.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    row = _extract_single_abs_row(
        image_path=image_path,
        device=str(args.device),
        detector=str(args.detector),
        torch_home=Path(args.torch_home),
    )
    abs_vec, rel_vec, pose, feature385 = _row_to_feature(row)
    feature_layout_check_passed = True
    validate_feature385_interleaved(feature385, abs_vec, rel_vec, pose)

    payload: Dict[str, Any] = {
        "image_path": str(image_path),
        "feature_dim_total": FEATURE_DIM_TOTAL,
        "feature_layout": FEATURE_LAYOUT,
        "feature385_pack_order": FEATURE385_PACK_ORDER,
        "neutral_rel_is_zero": True,
        "feature_layout_check_passed": bool(feature_layout_check_passed),
        "abs": abs_vec.tolist(),
        "rel": rel_vec.tolist(),
        "pose": pose.tolist(),
        "feature385": feature385.tolist(),
        "schema": {
            "abs_dim": ABS_DIM,
            "rel_dim": REL_DIM,
            "pose_dim": POSE_DIM,
        },
        "meta": {
            "face_found": int(float(row.get("face_found", 0) or 0)),
            "face_detect_conf": float(row.get("face_detect_conf", 0.0) or 0.0),
            "landmark_conf": float(row.get("landmark_conf", 0.0) or 0.0),
            "face_bbox": row.get("face_bbox", "[-1,-1,-1,-1]"),
            "yaw": float(row.get("yaw", 0.0) or 0.0),
            "pitch": float(row.get("pitch", 0.0) or 0.0),
            "roll": float(row.get("roll", 0.0) or 0.0),
            "extract_error": row.get("error", ""),
        },
        "notes": [
            "neutral rel is initialized to zeros(190).",
            "feature385 uses compare6_interleaved layout: [brow_abs,brow_rel,eye_abs,eye_rel,mouth_abs,mouth_rel,jaw_abs,jaw_rel,global_abs,global_rel,pose].",
            "This file is intended for HumanToRobotFeatureAdapter input.",
        ],
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_npy is not None:
        np.save(out_npy, feature385)

    print(f"[DONE] robot neutral json: {out_json}")
    if out_npy is not None:
        print(f"[DONE] feature385 npy: {out_npy}")


if __name__ == "__main__":
    main()
