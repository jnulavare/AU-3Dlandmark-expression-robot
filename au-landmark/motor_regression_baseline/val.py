#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import XYDataset, build_xy_from_split, load_latent24_map, load_split_indices, load_target30_map
from eval_metrics import (
    analyze_error_vs_context,
    collect_predictions,
    compute_regression_metrics,
    load_context_feature_arrays,
    load_motor_names,
    load_motor_region_indices,
)
from model import MotorRegressorMLP
from run_utils import resolve_eval_ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate baseline regressor on val split.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Override config.eval and use explicit checkpoint path")
    return p.parse_args()


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(str(cfg["train"]["device"]))
    metrics_cfg = cfg.get("metrics", {})

    ckpt_path, output_dir, run_name = resolve_eval_ckpt_path(cfg, args.ckpt)

    latent_map = load_latent24_map(Path(cfg["data"]["latent_file"]))
    target_map = load_target30_map(Path(cfg["data"]["target_file"]))
    split_path = Path(cfg["data"]["val_split"])
    x_val, y_val = build_xy_from_split(split_path, latent_map, target_map)

    ds = XYDataset(x_val, y_val)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    model = MotorRegressorMLP(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden1=int(cfg["model"]["hidden_dim1"]),
        hidden2=int(cfg["model"]["hidden_dim2"]),
        output_dim=int(cfg["model"]["output_dim"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred = collect_predictions(model, loader, device)
    region_indices = load_motor_region_indices(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    motor_names = load_motor_names(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    metric_dict = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        region_indices=region_indices,
        abs_error_percentile=float(metrics_cfg.get("abs_error_percentile", 95.0)),
        out_range_lo=float(metrics_cfg.get("out_range_lo", 0.0)),
        out_range_hi=float(metrics_cfg.get("out_range_hi", 1.0)),
        motor_names=motor_names,
        out_of_range_top_k=int(metrics_cfg.get("out_of_range_top_k", 10)),
    )

    context_cfg = metrics_cfg.get("error_context", {})
    if bool(context_cfg.get("enabled", True)):
        split_indices = load_split_indices(split_path)
        if len(split_indices) != int(y_true.shape[0]):
            raise RuntimeError(
                f"split size mismatch for context analysis: split={len(split_indices)} eval={y_true.shape[0]}"
            )

        latent_parent = Path(cfg["data"]["latent_file"]).parent
        rel_file = Path(context_cfg.get("rel_file", str(latent_parent / "REL_input_vec_X2C_gpu.csv.gz")))
        abs_file = Path(context_cfg.get("abs_file", str(latent_parent / "ABS_input_vec_X2C_gpu.csv.gz")))
        context = load_context_feature_arrays(split_indices=split_indices, rel_file=rel_file, abs_file=abs_file)

        sample_mae = np.mean(np.abs(y_pred - y_true), axis=1)
        metric_dict["error_context_analysis"] = analyze_error_vs_context(
            sample_mae=sample_mae,
            context=context,
            bins=int(context_cfg.get("bins", 10)),
        )
    else:
        metric_dict["error_context_analysis"] = {"enabled": False}

    metrics = {
        "split": "val",
        **metric_dict,
        "ckpt": str(ckpt_path),
        "run_name": run_name,
        "device": str(device),
        "region_indices": region_indices,
        "motor_names": motor_names,
    }
    out_json = output_dir / "val_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[DONE] val "
        f"run={run_name if run_name is not None else 'none'} "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
        f"r2={metrics['r2'] if metrics['r2'] is not None else 'NA'} "
        f"ev={metrics['explained_variance'] if metrics['explained_variance'] is not None else 'NA'} "
        f"oor={metrics['out_of_range']['ratio']:.6f} samples={metrics['samples']}"
    )
    print(f"[DONE] ckpt={ckpt_path}")
    print(f"[DONE] metrics={out_json}")


if __name__ == "__main__":
    main()
