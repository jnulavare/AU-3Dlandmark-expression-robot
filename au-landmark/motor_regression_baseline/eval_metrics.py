#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


# 30维电机默认区域映射（可在 config.metrics.motor_region_indices 中覆盖）
# 依据用户给出的控制定义分组：
# brow: 0-3
# eye: 4-9 (eyelids + gaze)
# jaw: 10-14 + 27-28 (head + jaw + neck)
# mouth: 15-26 + 29 (lips + nose wrinkle)
DEFAULT_MOTOR_REGION_INDICES: Dict[str, List[int]] = {
    "brow": [0, 1, 2, 3],
    "eye": [4, 5, 6, 7, 8, 9],
    "jaw": [10, 11, 12, 13, 14, 27, 28],
    "mouth": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29],
}


def _to_index_list(v: object) -> List[int]:
    if not isinstance(v, list):
        raise RuntimeError(f"region indices must be list[int], got: {type(v)}")
    out: List[int] = []
    for x in v:
        if not isinstance(x, int):
            raise RuntimeError(f"region index must be int, got: {x!r}")
        out.append(x)
    return out


def load_motor_region_indices(metrics_cfg: Mapping[str, object] | None, dim: int) -> Dict[str, List[int]]:
    # 从配置加载区域映射；未配置时使用默认映射
    region_cfg = None
    if isinstance(metrics_cfg, Mapping):
        region_cfg = metrics_cfg.get("motor_region_indices")

    if isinstance(region_cfg, Mapping):
        region_map: Dict[str, List[int]] = {str(k): _to_index_list(v) for k, v in region_cfg.items()}
    else:
        region_map = {k: list(v) for k, v in DEFAULT_MOTOR_REGION_INDICES.items()}

    # 校验索引合法性
    for region, idxs in region_map.items():
        if len(idxs) == 0:
            raise RuntimeError(f"region '{region}' has empty indices")
        for i in idxs:
            if i < 0 or i >= dim:
                raise RuntimeError(f"region '{region}' index out of range: {i}, dim={dim}")
    return region_map


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    # 统一收集 y_true / y_pred，便于计算分布类指标与R2/EV
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            ys.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    if len(ys) == 0:
        raise RuntimeError("empty loader: no samples to evaluate")
    return np.vstack(ys).astype(np.float64), np.vstack(preds).astype(np.float64)


def _r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, int]:
    # 按维度计算R2；若该维真值方差为0，则该维R2记为NaN
    err = y_true - y_pred
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    r2[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return r2, int(np.sum(valid))


def _explained_variance_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, int]:
    # 按维度计算 Explained Variance；若该维真值方差为0，则该维EV记为NaN
    err = y_true - y_pred
    var_true = np.var(y_true, axis=0)
    var_err = np.var(err, axis=0)
    valid = var_true > 1e-12
    ev = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    ev[valid] = 1.0 - (var_err[valid] / var_true[valid])
    return ev, int(np.sum(valid))


def _jsonable_float_list(arr: np.ndarray) -> List[float | None]:
    # JSON不支持NaN，这里转成None
    out: List[float | None] = []
    for v in arr.tolist():
        if v is None:
            out.append(None)
            continue
        fv = float(v)
        out.append(fv if np.isfinite(fv) else None)
    return out


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_indices: Mapping[str, Iterable[int]],
    abs_error_percentile: float = 95.0,
    out_range_lo: float = 0.0,
    out_range_hi: float = 1.0,
) -> Dict[str, object]:
    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 2:
        raise RuntimeError(f"expect 2D arrays, got y_true.ndim={y_true.ndim}")

    n, d = y_true.shape
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err * err

    # 基础指标：MAE/RMSE（总体 + 每维）
    mae_per_dim = np.mean(abs_err, axis=0)
    rmse_per_dim = np.sqrt(np.mean(sq_err, axis=0))
    mae = float(np.mean(mae_per_dim))
    rmse = float(np.mean(rmse_per_dim))

    # 拟合优度指标：R2 / Explained Variance（总体 + 每维）
    r2_per_dim, r2_valid_dims = _r2_per_dim(y_true, y_pred)
    ev_per_dim, ev_valid_dims = _explained_variance_per_dim(y_true, y_pred)
    r2 = float(np.nanmean(r2_per_dim)) if r2_valid_dims > 0 else None
    explained_variance = float(np.nanmean(ev_per_dim)) if ev_valid_dims > 0 else None

    # 误差分布指标：每维P95 absolute error / max absolute error
    pctl = float(abs_error_percentile)
    p95_abs_err_per_dim = np.percentile(abs_err, pctl, axis=0)
    max_abs_err_per_dim = np.max(abs_err, axis=0)

    # 分区域指标：按索引聚合 MAE / RMSE / P95 / Max
    region_metrics: Dict[str, object] = {}
    for region_name, idxs in region_indices.items():
        idx_arr = np.asarray(list(idxs), dtype=np.int64)
        if idx_arr.size == 0:
            continue
        r_abs = abs_err[:, idx_arr]
        r_sq = sq_err[:, idx_arr]
        region_metrics[str(region_name)] = {
            "indices": idx_arr.tolist(),
            "mae": float(np.mean(r_abs)),
            "rmse": float(np.sqrt(np.mean(r_sq))),
            "p95_abs_err": float(np.percentile(r_abs, pctl)),
            "max_abs_err": float(np.max(r_abs)),
        }

    # 输出范围一致性：预测值超出 [lo, hi] 的比例（总体 + 每维）
    oor_mask = (y_pred < out_range_lo) | (y_pred > out_range_hi)
    out_of_range_count = int(np.sum(oor_mask))
    total_value_count = int(n * d)
    out_of_range_ratio = float(out_of_range_count / max(total_value_count, 1))
    out_of_range_ratio_per_dim = np.mean(oor_mask.astype(np.float64), axis=0)

    return {
        "samples": int(n),
        "dim": int(d),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": explained_variance,
        "r2_valid_dims": r2_valid_dims,
        "explained_variance_valid_dims": ev_valid_dims,
        "mae_per_dim": [float(v) for v in mae_per_dim.tolist()],
        "rmse_per_dim": [float(v) for v in rmse_per_dim.tolist()],
        "r2_per_dim": _jsonable_float_list(r2_per_dim),
        "explained_variance_per_dim": _jsonable_float_list(ev_per_dim),
        "p95_abs_err_percentile": pctl,
        "p95_abs_err_per_dim": [float(v) for v in p95_abs_err_per_dim.tolist()],
        "max_abs_err_per_dim": [float(v) for v in max_abs_err_per_dim.tolist()],
        "p95_abs_err_mean": float(np.mean(p95_abs_err_per_dim)),
        "max_abs_err_mean": float(np.mean(max_abs_err_per_dim)),
        "region_metrics": region_metrics,
        "out_of_range": {
            "lo": float(out_range_lo),
            "hi": float(out_range_hi),
            "count": out_of_range_count,
            "total": total_value_count,
            "ratio": out_of_range_ratio,
            "within_range_ratio": float(1.0 - out_of_range_ratio),
            "ratio_per_dim": [float(v) for v in out_of_range_ratio_per_dim.tolist()],
        },
    }
