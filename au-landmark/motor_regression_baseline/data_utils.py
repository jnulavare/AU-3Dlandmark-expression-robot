#!/usr/bin/env python3
"""Data loading utilities for REL190 / REL193 -> motor30 regression.

REL190:
  AU_rel + LMK_rel + DIST_rel + ENERGY_rel (190 dims)
REL193:
  REL190 + pose(yaw/pitch/roll) (193 dims)
"""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


BROW_AU = ["AU1", "AU2", "AU4"]
EYE_AU = ["AU5", "AU6", "AU7"]
MOUTH_AU = ["AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25"]
JAW_AU = ["AU26"]

BROW_DIST = [
    "brow_left_eye_dist",
    "brow_right_eye_dist",
    "brow_inner_dist",
    "brow_outer_height_diff",
]
EYE_DIST = [
    "eye_left_open",
    "eye_right_open",
    "eye_left_width",
    "eye_right_width",
    "eye_left_ratio",
    "eye_right_ratio",
]
MOUTH_DIST = [
    "mouth_width",
    "mouth_open",
    "mouth_left_corner_to_nose",
    "mouth_right_corner_to_nose",
    "mouth_left_corner_raise",
    "mouth_right_corner_raise",
    "upper_lip_to_lower_lip",
    "upper_lip_to_nose",
    "lower_lip_to_chin",
    "mouth_center_to_nose",
    "mouth_center_to_chin",
]
JAW_DIST = ["jaw_open", "chin_to_nose", "chin_to_upper_lip"]

POSE_CANDIDATES = [
    ("yaw", "pitch", "roll"),
    ("pose_yaw", "pose_pitch", "pose_roll"),
    ("head_yaw", "head_pitch", "head_roll"),
]


def _parse_idx_from_name(name: str) -> int:
    return int(Path(name).stem)


def _safe_float(v: object) -> float:
    if v is None:
        return 0.0
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s == "":
        return 0.0
    return float(s)


def _open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _au_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def _dist_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def _lmk_rel_cols(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_rel_{i:02d}_x", f"lmk_rel_{i:02d}_y", f"lmk_rel_{i:02d}_z"])
    return cols


def build_region_columns_rel190() -> Dict[str, List[str]]:
    return {
        "brow_rel": _au_rel_cols(BROW_AU) + _lmk_rel_cols(0, 9) + _dist_rel_cols(BROW_DIST),
        "eye_rel": _au_rel_cols(EYE_AU) + _lmk_rel_cols(10, 21) + _dist_rel_cols(EYE_DIST),
        "mouth_rel": _au_rel_cols(MOUTH_AU) + _lmk_rel_cols(22, 41) + _dist_rel_cols(MOUTH_DIST),
        "jaw_rel": _au_rel_cols(JAW_AU) + _lmk_rel_cols(42, 49) + _dist_rel_cols(JAW_DIST),
        "global_rel": ["ENERGY_rel"],
    }


REGION_COLUMNS_REL190 = build_region_columns_rel190()
REL190_REGION_INPUT_DIMS = {k: len(v) for k, v in REGION_COLUMNS_REL190.items()}
REL193_REGION_INPUT_DIMS = {
    "brow_rel": 37,
    "eye_rel": 45,
    "mouth_rel": 79,
    "jaw_rel": 28,
    "global_rel": 1,
    "pose": 3,
}

# Backward-compat alias (REL190 default)
REGION_INPUT_DIMS = REL190_REGION_INPUT_DIMS

REL190_REGION_RANGES = {
    "brow_rel": (0, 37),
    "eye_rel": (37, 82),
    "mouth_rel": (82, 161),
    "jaw_rel": (161, 189),
    "global_rel": (189, 190),
}

REL193_REGION_RANGES = {
    "brow_rel": (0, 37),
    "eye_rel": (37, 82),
    "mouth_rel": (82, 161),
    "jaw_rel": (161, 189),
    "global_rel": (189, 190),
    "pose": (190, 193),
}


def get_region_input_dims(feature_mode: str) -> Dict[str, int]:
    mode = str(feature_mode).strip().upper()
    if mode == "REL193":
        return dict(REL193_REGION_INPUT_DIMS)
    if mode == "REL190":
        return dict(REL190_REGION_INPUT_DIMS)
    raise RuntimeError(f"unsupported feature_mode: {feature_mode}")


def load_target30_map(metadata_normalize_file: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    with metadata_normalize_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = _parse_idx_from_name(obj["file_name"])
            ctrl = np.asarray(obj["ctrl_value"], dtype=np.float32)
            if ctrl.shape[0] != 30:
                raise RuntimeError(f"ctrl_value dim != 30: {obj['file_name']}")
            out[idx] = ctrl
    return out


def load_split_indices(split_pkl: Path) -> List[int]:
    obj = pickle.load(open(split_pkl, "rb"))
    if "img_path" not in obj:
        raise RuntimeError(f"split file format error: {split_pkl}")
    return [_parse_idx_from_name(str(p)) for p in obj["img_path"]]


def _build_index_lookup(indices: Sequence[int]) -> Dict[int, List[int]]:
    lookup: Dict[int, List[int]] = {}
    for pos, idx in enumerate(indices):
        lookup.setdefault(int(idx), []).append(pos)
    return lookup


def _extract_row(row: Mapping[str, str], cols: Sequence[str]) -> np.ndarray:
    return np.asarray([_safe_float(row.get(c, 0.0)) for c in cols], dtype=np.float32)


def _feature_vec_from_row(row: Mapping[str, str], dim: int) -> np.ndarray:
    return np.asarray([_safe_float(row.get(f"feat_{i:03d}", 0.0)) for i in range(dim)], dtype=np.float32)


def _split_rel190(feature190: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "brow_rel": feature190[0:37],
        "eye_rel": feature190[37:82],
        "mouth_rel": feature190[82:161],
        "jaw_rel": feature190[161:189],
        "global_rel": feature190[189:190],
    }


def _split_rel193(feature193: np.ndarray) -> Dict[str, np.ndarray]:
    out = _split_rel190(feature193[:190])
    out["pose"] = feature193[190:193]
    return out


def _resolve_pose_columns(fieldnames: Sequence[str]) -> Tuple[str, str, str]:
    fields = set(fieldnames)
    for y, p, r in POSE_CANDIDATES:
        if {y, p, r}.issubset(fields):
            return y, p, r
    raise RuntimeError(
        "pose columns not found. expected one of: "
        "(yaw,pitch,roll) / (pose_yaw,pose_pitch,pose_roll) / (head_yaw,head_pitch,head_roll)"
    )


def _load_pose_map(abs_file: Path) -> Dict[int, np.ndarray]:
    pose_map: Dict[int, np.ndarray] = {}
    with _open_text(abs_file, "rt") as f_abs:
        reader = csv.DictReader(f_abs)
        if "image_name" not in (reader.fieldnames or []):
            raise RuntimeError(f"ABS file missing image_name: {abs_file}")
        y_col, p_col, r_col = _resolve_pose_columns(reader.fieldnames or [])
        for row in reader:
            idx = _parse_idx_from_name(str(row["image_name"]))
            pose_map[idx] = np.asarray(
                [_safe_float(row.get(y_col)), _safe_float(row.get(p_col)), _safe_float(row.get(r_col))],
                dtype=np.float32,
            )
    return pose_map


def build_region_inputs_from_split(
    split_pkl: Path,
    abs_file: Path,
    rel_file: Path,
    target30_map: Mapping[int, np.ndarray],
    feature190_file: Path | None = None,
    feature193_file: Path | None = None,
    feature_file: Path | None = None,
    feature_mode: str = "REL190",
    feature385_file: Path | None = None,  # legacy compatibility, unused
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build regional inputs aligned to split order for REL190/REL193."""
    del feature385_file

    mode = str(feature_mode).strip().upper()
    if mode not in {"REL190", "REL193"}:
        raise RuntimeError(f"unsupported feature_mode: {feature_mode}")

    if mode == "REL190":
        dims = REL190_REGION_INPUT_DIMS
        feat_dim = 190
    else:
        dims = REL193_REGION_INPUT_DIMS
        feat_dim = 193

    indices = load_split_indices(split_pkl)
    n = len(indices)
    if n == 0:
        raise RuntimeError(f"empty split: {split_pkl}")

    lookup = _build_index_lookup(indices)
    inputs: Dict[str, np.ndarray] = {k: np.zeros((n, d), dtype=np.float32) for k, d in dims.items()}
    found = np.zeros(n, dtype=np.bool_)

    selected_feature_file = None
    if mode == "REL190":
        selected_feature_file = feature190_file or feature_file
    if mode == "REL193":
        selected_feature_file = feature193_file or feature_file or feature190_file

    if selected_feature_file is not None and selected_feature_file.exists():
        with _open_text(selected_feature_file, "rt") as f_feat:
            reader = csv.DictReader(f_feat)
            need_cols = ["image_name"] + [f"feat_{i:03d}" for i in range(feat_dim)]
            miss = [c for c in need_cols if c not in (reader.fieldnames or [])]
            if miss:
                raise RuntimeError(f"{selected_feature_file.name} missing columns: {miss[:8]}")

            for row in reader:
                idx = _parse_idx_from_name(str(row["image_name"]))
                positions = lookup.get(idx)
                if not positions:
                    continue
                vec = _feature_vec_from_row(row, feat_dim)
                cache = _split_rel190(vec) if mode == "REL190" else _split_rel193(vec)
                for p in positions:
                    for key in inputs:
                        inputs[key][p] = cache[key]
                    found[p] = True
    else:
        # Fallback to raw REL columns. For REL193, add pose from ABS.
        pose_map: Dict[int, np.ndarray] | None = None
        if mode == "REL193":
            if abs_file is None or (not abs_file.exists()) or (not abs_file.is_file()):
                raise RuntimeError("REL193 fallback requires abs_file with pose columns")
            pose_map = _load_pose_map(abs_file)

        with _open_text(rel_file, "rt") as f_rel:
            reader = csv.DictReader(f_rel)
            need_cols = ["image_name"]
            for cols in REGION_COLUMNS_REL190.values():
                need_cols.extend(cols)
            miss = [c for c in need_cols if c not in (reader.fieldnames or [])]
            if miss:
                raise RuntimeError(f"REL file missing columns: {miss[:8]}")

            for row in reader:
                idx = _parse_idx_from_name(str(row["image_name"]))
                positions = lookup.get(idx)
                if not positions:
                    continue
                cache = {k: _extract_row(row, cols) for k, cols in REGION_COLUMNS_REL190.items()}
                if mode == "REL193":
                    if pose_map is None or idx not in pose_map:
                        continue
                    cache["pose"] = pose_map[idx]
                for p in positions:
                    for key in inputs:
                        inputs[key][p] = cache[key]
                    found[p] = True

    miss_count = int(np.sum(~found))
    if miss_count > 0:
        raise RuntimeError(f"split {split_pkl.name} missing {mode} rows: {miss_count}")

    y = np.zeros((n, 30), dtype=np.float32)
    miss_target = 0
    for i, idx in enumerate(indices):
        target = target30_map.get(int(idx))
        if target is None:
            miss_target += 1
            continue
        y[i] = target
    if miss_target > 0:
        raise RuntimeError(f"split {split_pkl.name} missing target rows: {miss_target}")

    print(f"[INFO] {mode} split={split_pkl.name} feature_dim={feat_dim}")
    for key in inputs.keys():
        print(f"[INFO] {key} shape={inputs[key].shape}")
    return inputs, y


class RegionalInputDataset(Dataset):
    """Torch dataset returning ({regional inputs}, target30)."""

    def __init__(self, inputs: Mapping[str, np.ndarray], y: np.ndarray):
        self.inputs = {k: torch.from_numpy(v).float() for k, v in inputs.items()}
        self.y = torch.from_numpy(y).float()
        n = int(self.y.shape[0])
        for k, v in self.inputs.items():
            if int(v.shape[0]) != n:
                raise RuntimeError(f"input '{k}' first dim mismatch: {v.shape[0]} vs y={n}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = {k: v[idx] for k, v in self.inputs.items()}
        return x, self.y[idx]
