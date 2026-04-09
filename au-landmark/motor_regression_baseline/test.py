#!/usr/bin/env python3
"""Test entry for regional ABS/REL/Pose -> z24 -> motor30 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import RegionalInputDataset, build_region_inputs_from_split, load_split_indices, load_target30_map
from eval_metrics import (
    analyze_error_vs_context,
    clip_predictions_to_range,
    collect_predictions,
    compute_boundary_violation_metrics,
    compute_pose_slice_mae_analysis,
    compute_regression_metrics,
    load_context_feature_arrays,
    load_motor_names,
    load_motor_region_indices,
)
from model import MotorRegressorMLP
from run_utils import resolve_eval_ckpt_path


def _as_bool(v: object, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def resolve_boundary_eval_cfg(cfg: dict) -> dict:
    # 测试阶段边界配置：与 val 口径一致，支持评估前 clip。
    metrics_cfg = cfg.get("metrics", {})
    boundary_cfg = cfg.get("boundary", {})
    lo = float(boundary_cfg.get("lo", metrics_cfg.get("out_range_lo", 0.0)))
    hi = float(boundary_cfg.get("hi", metrics_cfg.get("out_range_hi", 1.0)))
    if hi <= lo:
        raise RuntimeError(f"invalid boundary range: lo={lo}, hi={hi}")
    return {
        "lo": lo,
        "hi": hi,
        "clip_predictions_in_eval": _as_bool(boundary_cfg.get("clip_predictions_in_eval", True), default=True),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test regional-gated regressor on test split.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Override config.eval and use explicit checkpoint path")
    return p.parse_args()


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _context_data_root(data_cfg: dict) -> Path:
    # 上下文特征默认从数据目录推断，也可在配置里显式指定。
    if "abs_file" in data_cfg:
        return Path(str(data_cfg["abs_file"])).parent
    if "rel_file" in data_cfg:
        return Path(str(data_cfg["rel_file"])).parent
    if "latent_file" in data_cfg:
        return Path(str(data_cfg["latent_file"])).parent
    return Path(".")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(str(cfg["train"]["device"]))
    metrics_cfg = cfg.get("metrics", {})
    boundary_eval_cfg = resolve_boundary_eval_cfg(cfg)

    ckpt_path, output_dir, run_name = resolve_eval_ckpt_path(cfg, args.ckpt)

    # 1) 按 test split 顺序构建输入，确保输出指标与样本一一对应。
    target_map = load_target30_map(Path(cfg["data"]["target_file"]))
    split_path = Path(cfg["data"]["test_split"])
    feature385_file = Path(cfg["data"]["feature385_file"]) if "feature385_file" in cfg["data"] else None
    x_test, y_test = build_region_inputs_from_split(
        split_path,
        abs_file=Path(cfg["data"]["abs_file"]),
        rel_file=Path(cfg["data"]["rel_file"]),
        target30_map=target_map,
        feature385_file=feature385_file,
    )

    ds = RegionalInputDataset(x_test, y_test)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # 2) 加载最优/指定 checkpoint，执行完整测试集推理。
    model = MotorRegressorMLP().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred_raw = collect_predictions(model, loader, device)
    if boundary_eval_cfg["clip_predictions_in_eval"]:
        y_pred = clip_predictions_to_range(y_pred_raw, lo=boundary_eval_cfg["lo"], hi=boundary_eval_cfg["hi"])
    else:
        y_pred = y_pred_raw
    region_indices = load_motor_region_indices(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    motor_names = load_motor_names(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    # 3) 计算回归指标并保持 JSON 字段结构稳定，便于横向对比。
    metric_dict = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        region_indices=region_indices,
        abs_error_percentile=float(metrics_cfg.get("abs_error_percentile", 95.0)),
        out_range_lo=float(boundary_eval_cfg["lo"]),
        out_range_hi=float(boundary_eval_cfg["hi"]),
        motor_names=motor_names,
        out_of_range_top_k=int(metrics_cfg.get("out_of_range_top_k", 10)),
    )
    metric_dict["boundary_constraint"] = {
        "clip_predictions_in_eval": bool(boundary_eval_cfg["clip_predictions_in_eval"]),
        "raw_prediction_boundary": compute_boundary_violation_metrics(
            y_pred=y_pred_raw,
            lo=float(boundary_eval_cfg["lo"]),
            hi=float(boundary_eval_cfg["hi"]),
        ),
        "final_prediction_boundary": compute_boundary_violation_metrics(
            y_pred=y_pred,
            lo=float(boundary_eval_cfg["lo"]),
            hi=float(boundary_eval_cfg["hi"]),
        ),
    }

    context_cfg = metrics_cfg.get("error_context", {})
    pose_slice_cfg = metrics_cfg.get("pose_slice", {})
    # 只要开启 error_context 或 pose_slice，就先准备上下文数组。
    need_pose_context = bool(context_cfg.get("enabled", True)) or bool(pose_slice_cfg.get("enabled", True))
    context = None
    if need_pose_context:
        split_indices = load_split_indices(split_path)
        if len(split_indices) != int(y_true.shape[0]):
            raise RuntimeError(
                f"split size mismatch for context analysis: split={len(split_indices)} eval={y_true.shape[0]}"
            )

        parent = _context_data_root(cfg["data"])
        rel_file = Path(context_cfg.get("rel_file", str(parent / "REL_input_vec_X2C_gpu.csv.gz")))
        abs_file = Path(context_cfg.get("abs_file", str(parent / "ABS_input_vec_X2C_gpu.csv.gz")))
        context = load_context_feature_arrays(split_indices=split_indices, rel_file=rel_file, abs_file=abs_file)

    if bool(context_cfg.get("enabled", True)):
        if context is None:
            raise RuntimeError("context should not be None when error_context is enabled")

        sample_mae = np.mean(np.abs(y_pred - y_true), axis=1)
        # 统计样本误差与上下文变量（如 yaw/pitch/roll/energy）的关系。
        metric_dict["error_context_analysis"] = analyze_error_vs_context(
            sample_mae=sample_mae,
            context=context,
            bins=int(context_cfg.get("bins", 10)),
        )
    else:
        metric_dict["error_context_analysis"] = {"enabled": False}

    if bool(pose_slice_cfg.get("enabled", True)):
        if context is None:
            raise RuntimeError("context should not be None when pose_slice is enabled")
        # 按姿态角度分段（frontal/moderate/extreme）比较分桶 MAE。
        metric_dict["pose_slice_mae_analysis"] = compute_pose_slice_mae_analysis(
            y_true=y_true,
            y_pred=y_pred,
            yaw=np.asarray(context.get("yaw", np.array([])), dtype=np.float64),
            pitch=np.asarray(context.get("pitch", np.array([])), dtype=np.float64),
            roll=np.asarray(context.get("roll", np.array([])), dtype=np.float64),
            region_indices=region_indices,
            motor_names=motor_names,
            frontal_max_deg=float(pose_slice_cfg.get("frontal_max_deg", 10.0)),
            moderate_max_deg=float(pose_slice_cfg.get("moderate_max_deg", 25.0)),
        )
    else:
        metric_dict["pose_slice_mae_analysis"] = {"enabled": False}

    metrics = {
        "split": "test",
        **metric_dict,
        "ckpt": str(ckpt_path),
        "run_name": run_name,
        "device": str(device),
        "region_indices": region_indices,
        "motor_names": motor_names,
    }
    out_json = output_dir / "test_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[DONE] test "
        f"run={run_name if run_name is not None else 'none'} "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
        f"r2={metrics['r2'] if metrics['r2'] is not None else 'NA'} "
        f"ev={metrics['explained_variance'] if metrics['explained_variance'] is not None else 'NA'} "
        f"oor={metrics['out_of_range']['ratio']:.6f} "
        f"raw_oor={metrics['boundary_constraint']['raw_prediction_boundary']['ratio']:.6f} "
        f"samples={metrics['samples']}"
    )
    if metrics["pose_slice_mae_analysis"].get("enabled", False):
        slices = metrics["pose_slice_mae_analysis"]["slices"]
        print(
            "[POSE] "
            f"frontal_mae={slices['frontal']['overall_mae']} "
            f"moderate_mae={slices['moderate_pose']['overall_mae']} "
            f"extreme_mae={slices['extreme_pose']['overall_mae']}"
        )
    print(f"[DONE] ckpt={ckpt_path}")
    print(f"[DONE] metrics={out_json}")


if __name__ == "__main__":
    main()
