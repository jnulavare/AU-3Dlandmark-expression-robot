#!/usr/bin/env python3
"""Utilities for building compare6 FEATURE385 by columns.json mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


FEATURE_DIM = 385
FEATURE_LAYOUT = "compare6_columns_json"
FEATURE385_PACK_ORDER = (
    "brow_abs,brow_rel,eye_abs,eye_rel,mouth_abs,mouth_rel,"
    "jaw_abs,jaw_rel,global_abs,global_rel,pose"
)


def safe_get_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def load_feature_columns(columns_json_path: str | Path) -> List[Dict[str, Any]]:
    p = Path(columns_json_path)
    if not p.exists():
        raise FileNotFoundError(f"columns_json not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("columns json must be a list")

    rows: List[Dict[str, Any]] = []
    for item in obj:
        if not isinstance(item, dict):
            raise ValueError("columns json item must be object")
        if "feat_idx" not in item or "source_column" not in item:
            raise ValueError("each columns item must contain feat_idx and source_column")
        idx = int(item["feat_idx"])
        row = dict(item)
        row["feat_idx"] = idx
        rows.append(row)

    rows.sort(key=lambda x: int(x["feat_idx"]))
    if len(rows) != FEATURE_DIM:
        raise ValueError(f"columns length must be {FEATURE_DIM}, got {len(rows)}")
    idxs = [int(r["feat_idx"]) for r in rows]
    if idxs != list(range(FEATURE_DIM)):
        raise ValueError("feat_idx must cover 0..384 exactly")
    return rows


def row_to_compare6_feature385(
    row: Mapping[str, Any],
    columns: List[Mapping[str, Any]],
    *,
    zero_rel: bool = False,
) -> Dict[str, Any]:
    feature = np.zeros((FEATURE_DIM,), dtype=np.float32)
    missing_columns: List[str] = []
    used_columns: List[str] = []

    for item in columns:
        idx = int(item["feat_idx"])
        source = str(item.get("source", "")).strip().lower()
        source_col = str(item.get("source_column", "")).strip()

        if zero_rel and source == "rel":
            feature[idx] = 0.0
            continue

        if source_col in row:
            feature[idx] = safe_get_float(row.get(source_col), 0.0)
            used_columns.append(source_col)
            continue

        # limited explicit pose fallback only
        low = source_col.lower()
        if low in {"yaw", "pitch", "roll"} and low in row:
            feature[idx] = safe_get_float(row.get(low), 0.0)
            used_columns.append(low)
            continue

        missing_columns.append(source_col)
        feature[idx] = 0.0

    return {
        "feature385": feature,
        "missing_columns": sorted(set(missing_columns)),
        "used_columns": sorted(set(used_columns)),
        "feature_layout": FEATURE_LAYOUT,
        "feature_build_method": "columns_json_source_column",
        "rel_columns_filled_zero": bool(zero_rel),
    }


def compare_feature385_by_columns(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    columns: List[Mapping[str, Any]],
    top_k: int = 50,
) -> Dict[str, Any]:
    a = np.asarray(feature_a, dtype=np.float32).reshape(-1)
    b = np.asarray(feature_b, dtype=np.float32).reshape(-1)
    if a.shape[0] != FEATURE_DIM or b.shape[0] != FEATURE_DIM:
        raise ValueError("feature_a and feature_b must be shape (385,)")

    diff = np.abs(a - b)
    order = np.argsort(-diff)[:top_k]
    rows = []
    for idx in order.tolist():
        meta = columns[idx]
        rows.append(
            {
                "feat_idx": int(idx),
                "feat_name": meta.get("feat_name", f"feat_{idx:03d}"),
                "region": meta.get("region"),
                "source": meta.get("source"),
                "source_column": meta.get("source_column"),
                "value_a": float(a[idx]),
                "value_b": float(b[idx]),
                "abs_diff": float(diff[idx]),
            }
        )
    return {
        "feature_mae": float(np.mean(diff)),
        "feature_max_abs_diff": float(np.max(diff)),
        "top_feature_diffs": rows,
    }


def split_feature385_to_model_inputs(feature385: np.ndarray) -> Dict[str, np.ndarray]:
    f = np.asarray(feature385, dtype=np.float32).reshape(-1)
    if f.shape[0] != FEATURE_DIM:
        raise ValueError(f"feature dim mismatch: {f.shape[0]} vs 385")
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


def feature385_to_internal_abs_rel_pose(feature385: np.ndarray) -> Dict[str, np.ndarray]:
    parts = split_feature385_to_model_inputs(feature385)
    abs_vec = np.concatenate(
        [
            parts["brow_abs"],
            parts["eye_abs"],
            parts["mouth_abs"],
            parts["jaw_abs"],
            parts["global_abs"],
        ],
        axis=0,
    ).astype(np.float32)
    rel_vec = np.concatenate(
        [
            parts["brow_rel"],
            parts["eye_rel"],
            parts["mouth_rel"],
            parts["jaw_rel"],
            parts["global_rel"],
        ],
        axis=0,
    ).astype(np.float32)
    pose = parts["pose"].astype(np.float32)
    return {"abs": abs_vec, "rel": rel_vec, "pose": pose}
