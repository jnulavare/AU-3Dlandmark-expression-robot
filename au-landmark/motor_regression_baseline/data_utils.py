#!/usr/bin/env python3
"""Data loading utilities for regional ABS/REL/Pose -> motor regression.

This module is responsible for:
1) defining region-wise feature column groups,
2) aligning ABS/REL rows with split indices,
3) building model-ready tensors and target vectors.
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


def _au_abs_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_abs_intensity" for n in names]


def _au_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def _dist_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def _lmk_abs_cols(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_abs_norm_{i:02d}_x", f"lmk_abs_norm_{i:02d}_y", f"lmk_abs_norm_{i:02d}_z"])
    return cols


def _lmk_rel_cols(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_rel_{i:02d}_x", f"lmk_rel_{i:02d}_y", f"lmk_rel_{i:02d}_z"])
    return cols


def build_region_columns() -> Dict[str, List[str]]:
    # Regional feature partition used by regional encoders in model.py
    return {
        "brow_abs": _au_abs_cols(BROW_AU) + _lmk_abs_cols(0, 9) + BROW_DIST,
        "brow_rel": _au_rel_cols(BROW_AU) + _lmk_rel_cols(0, 9) + _dist_rel_cols(BROW_DIST),
        "eye_abs": _au_abs_cols(EYE_AU) + _lmk_abs_cols(10, 21) + EYE_DIST,
        "eye_rel": _au_rel_cols(EYE_AU) + _lmk_rel_cols(10, 21) + _dist_rel_cols(EYE_DIST),
        "mouth_abs": _au_abs_cols(MOUTH_AU) + _lmk_abs_cols(22, 41) + MOUTH_DIST,
        "mouth_rel": _au_rel_cols(MOUTH_AU) + _lmk_rel_cols(22, 41) + _dist_rel_cols(MOUTH_DIST),
        "jaw_abs": _au_abs_cols(JAW_AU) + _lmk_abs_cols(42, 49) + JAW_DIST,
        "jaw_rel": _au_rel_cols(JAW_AU) + _lmk_rel_cols(42, 49) + _dist_rel_cols(JAW_DIST),
        "global_abs": ["yaw", "pitch", "roll"],
        "global_rel": ["ENERGY_rel"],
    }


REGION_COLUMNS = build_region_columns()
REGION_INPUT_DIMS = {k: len(v) for k, v in REGION_COLUMNS.items()}


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


def build_region_inputs_from_split(
    split_pkl: Path,
    abs_file: Path,
    rel_file: Path,
    target30_map: Mapping[int, np.ndarray],
    feature385_file: Path | None = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    # Split order is the single source of truth for sample ordering.
    # ABS/REL rows are looked up by image index and written into that order.
    indices = load_split_indices(split_pkl)
    n = len(indices)
    if n == 0:
        raise RuntimeError(f"empty split: {split_pkl}")

    lookup = _build_index_lookup(indices)
    inputs: Dict[str, np.ndarray] = {k: np.zeros((n, len(cols)), dtype=np.float32) for k, cols in REGION_COLUMNS.items()}
    inputs["pose"] = np.zeros((n, 3), dtype=np.float32)
    found_abs = np.zeros(n, dtype=np.bool_)
    found_rel = np.zeros(n, dtype=np.bool_)

    # Optional fast path: directly read prebuilt FEATURE385 from x2c_data_compare3.
    # Layout:
    # brow_abs(37), brow_rel(37), eye_abs(45), eye_rel(45),
    # mouth_abs(79), mouth_rel(79), jaw_abs(28), jaw_rel(28),
    # global_abs(3), global_rel(1), pose(3) => total 385
    if feature385_file is not None and feature385_file.exists():
        with _open_text(feature385_file, "rt") as f_feat:
            feat_reader = csv.DictReader(f_feat)
            need_cols = ["image_name"] + [f"feat_{i:03d}" for i in range(385)]
            miss = [c for c in need_cols if c not in (feat_reader.fieldnames or [])]
            if miss:
                raise RuntimeError(f"FEATURE385 file missing columns: {miss[:8]}")

            found_feat = np.zeros(n, dtype=np.bool_)
            for row in feat_reader:
                idx = _parse_idx_from_name(str(row["image_name"]))
                positions = lookup.get(idx)
                if not positions:
                    continue
                feat = np.asarray([_safe_float(row.get(f"feat_{i:03d}", 0.0)) for i in range(385)], dtype=np.float32)
                row_cache = {
                    "brow_abs": feat[0:37],
                    "brow_rel": feat[37:74],
                    "eye_abs": feat[74:119],
                    "eye_rel": feat[119:164],
                    "mouth_abs": feat[164:243],
                    "mouth_rel": feat[243:322],
                    "jaw_abs": feat[322:350],
                    "jaw_rel": feat[350:378],
                    "global_abs": feat[378:381],
                    "global_rel": feat[381:382],
                    "pose": feat[382:385],
                }
                for p in positions:
                    for key in row_cache:
                        inputs[key][p] = row_cache[key]
                    found_feat[p] = True

            miss_feat_count = int(np.sum(~found_feat))
            if miss_feat_count > 0:
                raise RuntimeError(f"split {split_pkl.name} missing FEATURE385 rows: {miss_feat_count}")
    else:
        # Default path: construct inputs from ABS + REL source files.
        with _open_text(abs_file, "rt") as f_abs:
            abs_reader = csv.DictReader(f_abs)
            need_abs = ["image_name", "yaw", "pitch", "roll"]
            for key in ("brow_abs", "eye_abs", "mouth_abs", "jaw_abs", "global_abs"):
                need_abs.extend(REGION_COLUMNS[key])
            miss_abs = [c for c in need_abs if c not in (abs_reader.fieldnames or [])]
            if miss_abs:
                raise RuntimeError(f"ABS file missing columns: {miss_abs[:8]}")

            for row in abs_reader:
                idx = _parse_idx_from_name(str(row["image_name"]))
                positions = lookup.get(idx)
                if not positions:
                    continue
                # Cache parsed vectors once, then write to all positions that share this index.
                row_cache = {
                    "brow_abs": _extract_row(row, REGION_COLUMNS["brow_abs"]),
                    "eye_abs": _extract_row(row, REGION_COLUMNS["eye_abs"]),
                    "mouth_abs": _extract_row(row, REGION_COLUMNS["mouth_abs"]),
                    "jaw_abs": _extract_row(row, REGION_COLUMNS["jaw_abs"]),
                    "global_abs": _extract_row(row, REGION_COLUMNS["global_abs"]),
                    "pose": np.asarray(
                        [_safe_float(row["yaw"]), _safe_float(row["pitch"]), _safe_float(row["roll"])], dtype=np.float32
                    ),
                }
                for p in positions:
                    inputs["brow_abs"][p] = row_cache["brow_abs"]
                    inputs["eye_abs"][p] = row_cache["eye_abs"]
                    inputs["mouth_abs"][p] = row_cache["mouth_abs"]
                    inputs["jaw_abs"][p] = row_cache["jaw_abs"]
                    inputs["global_abs"][p] = row_cache["global_abs"]
                    inputs["pose"][p] = row_cache["pose"]
                    found_abs[p] = True

        with _open_text(rel_file, "rt") as f_rel:
            rel_reader = csv.DictReader(f_rel)
            need_rel = ["image_name"]
            for key in ("brow_rel", "eye_rel", "mouth_rel", "jaw_rel", "global_rel"):
                need_rel.extend(REGION_COLUMNS[key])
            miss_rel = [c for c in need_rel if c not in (rel_reader.fieldnames or [])]
            if miss_rel:
                raise RuntimeError(f"REL file missing columns: {miss_rel[:8]}")

            for row in rel_reader:
                idx = _parse_idx_from_name(str(row["image_name"]))
                positions = lookup.get(idx)
                if not positions:
                    continue
                row_cache = {
                    "brow_rel": _extract_row(row, REGION_COLUMNS["brow_rel"]),
                    "eye_rel": _extract_row(row, REGION_COLUMNS["eye_rel"]),
                    "mouth_rel": _extract_row(row, REGION_COLUMNS["mouth_rel"]),
                    "jaw_rel": _extract_row(row, REGION_COLUMNS["jaw_rel"]),
                    "global_rel": _extract_row(row, REGION_COLUMNS["global_rel"]),
                }
                for p in positions:
                    inputs["brow_rel"][p] = row_cache["brow_rel"]
                    inputs["eye_rel"][p] = row_cache["eye_rel"]
                    inputs["mouth_rel"][p] = row_cache["mouth_rel"]
                    inputs["jaw_rel"][p] = row_cache["jaw_rel"]
                    inputs["global_rel"][p] = row_cache["global_rel"]
                    found_rel[p] = True

        miss_abs_count = int(np.sum(~found_abs))
        miss_rel_count = int(np.sum(~found_rel))
        if miss_abs_count > 0 or miss_rel_count > 0:
            raise RuntimeError(
                f"split {split_pkl.name} missing rows: abs={miss_abs_count} rel={miss_rel_count}"
            )

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

    return inputs, y


class RegionalInputDataset(Dataset):
    """Torch dataset returning ({region_inputs}, target30)."""

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
