#!/usr/bin/env python3
"""Training entry for branch6 regional model.

Key changes in this version:
- Pre-LN residual trunk + base/residual heads in model.py
- Weighted SmoothL1 task loss by facial region
- EMA (exponential moving average) model for validation/checkpoint
- Early stopping with minimum epoch guard
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data_utils import RegionalInputDataset, build_region_inputs_from_split, load_target30_map
from eval_metrics import load_motor_region_indices
from model import MotorRegressorMLP
from run_utils import resolve_train_output_dir


def _as_bool(v: object, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def resolve_boundary_train_cfg(cfg: dict) -> dict:
    metrics_cfg = cfg.get("metrics", {})
    boundary_cfg = cfg.get("boundary", {})
    train_boundary_cfg = boundary_cfg.get("train", {}) if isinstance(boundary_cfg, dict) else {}
    lo = float(boundary_cfg.get("lo", metrics_cfg.get("out_range_lo", 0.0)))
    hi = float(boundary_cfg.get("hi", metrics_cfg.get("out_range_hi", 1.0)))
    if hi <= lo:
        raise RuntimeError(f"invalid boundary range: lo={lo}, hi={hi}")
    return {
        "lo": lo,
        "hi": hi,
        "clip_predictions_in_eval": _as_bool(boundary_cfg.get("clip_predictions_in_eval", True), default=True),
        "clamp_for_task_loss": _as_bool(train_boundary_cfg.get("clamp_for_task_loss", True), default=True),
        "enable_boundary_loss": _as_bool(train_boundary_cfg.get("enable_boundary_loss", True), default=True),
        "boundary_loss_weight": float(train_boundary_cfg.get("boundary_loss_weight", 0.1)),
    }


def resolve_region_loss_weights(cfg: dict) -> Dict[str, float]:
    """Resolve weighted SmoothL1 coefficients.

    Defaults requested by user:
    brow=1.25, eye=1.0, jaw=1.0, mouth=1.0
    """
    default = {"brow": 1.25, "eye": 1.0, "jaw": 1.0, "mouth": 1.0}
    train_cfg = cfg.get("train", {})
    weights_cfg = train_cfg.get("region_loss_weights", {})
    if not isinstance(weights_cfg, Mapping):
        weights_cfg = {}
    out = {}
    for k, dv in default.items():
        out[k] = float(weights_cfg.get(k, dv))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train branch6 regional model.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_inputs_to_device(x: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in x.items()}


def compute_boundary_loss_torch(pred: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (torch.relu(lo - pred) + torch.relu(pred - hi)).mean()


def compute_weighted_smooth_l1_task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    region_indices_t: Mapping[str, torch.Tensor],
    region_weights: Mapping[str, float],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    per_region: Dict[str, torch.Tensor] = {}
    total = pred.new_tensor(0.0)
    for region in ("brow", "eye", "jaw", "mouth"):
        idx = region_indices_t[region]
        loss_r = F.smooth_l1_loss(pred.index_select(1, idx), target.index_select(1, idx), reduction="mean")
        per_region[region] = loss_r
        total = total + float(region_weights.get(region, 1.0)) * loss_r
    return total, per_region


def evaluate_mae(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    clip_predictions: bool,
    lo: float,
    hi: float,
) -> float:
    model.eval()
    mae_sum = 0.0
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x = move_inputs_to_device(x, device=device)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            if clip_predictions:
                pred = torch.clamp(pred, min=lo, max=hi)
            mae = torch.mean(torch.abs(pred - y), dim=1)
            mae_sum += float(mae.sum().item())
            n += int(y.shape[0])
    return mae_sum / max(n, 1)


class ModelEMA:
    """Maintain exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, beta: float = 0.999):
        self.beta = float(beta)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()
        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()
            if not ema_v.dtype.is_floating_point:
                ema_v.copy_(model_v)
            else:
                ema_v.mul_(self.beta).add_(model_v, alpha=(1.0 - self.beta))


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    boundary_cfg = resolve_boundary_train_cfg(cfg)
    region_weights = resolve_region_loss_weights(cfg)

    train_cfg = cfg["train"]
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    use_ema_for_val = _as_bool(train_cfg.get("use_ema_for_val", True), default=True)
    min_epochs_before_early_stop = int(train_cfg.get("min_epochs_before_early_stop", 120))

    device = resolve_device(str(train_cfg["device"]))
    output_dir, run_name = resolve_train_output_dir(train_cfg)

    abs_file = Path(cfg["data"]["abs_file"])
    rel_file = Path(cfg["data"]["rel_file"])
    feature385_file = Path(cfg["data"]["feature385_file"]) if "feature385_file" in cfg["data"] else None
    target_file = Path(cfg["data"]["target_file"])
    train_split = Path(cfg["data"]["train_split"])
    val_split = Path(cfg["data"]["val_split"])

    print(f"[INFO] device={device}")
    if run_name is not None:
        print(f"[INFO] run_name={run_name}")
    print(f"[INFO] output_dir={output_dir}")
    print(
        "[INFO] boundary "
        f"lo={boundary_cfg['lo']} hi={boundary_cfg['hi']} "
        f"eval_clip={boundary_cfg['clip_predictions_in_eval']} "
        f"clamp_for_task_loss={boundary_cfg['clamp_for_task_loss']} "
        f"boundary_loss={boundary_cfg['enable_boundary_loss']} "
        f"weight={boundary_cfg['boundary_loss_weight']}"
    )
    print(
        "[INFO] region_weights "
        f"brow={region_weights['brow']} eye={region_weights['eye']} "
        f"jaw={region_weights['jaw']} mouth={region_weights['mouth']}"
    )
    print(
        f"[INFO] ema_decay={ema_decay} use_ema_for_val={use_ema_for_val} "
        f"min_epochs_before_early_stop={min_epochs_before_early_stop}"
    )

    print("[INFO] loading target map...")
    target_map = load_target30_map(target_file)

    print("[INFO] building train/val arrays from ABS+REL+Pose...")
    if feature385_file is not None and feature385_file.exists():
        print(f"[INFO] using feature385_file={feature385_file}")
    else:
        print(f"[INFO] using abs_file={abs_file} rel_file={rel_file}")

    x_train, y_train = build_region_inputs_from_split(
        train_split,
        abs_file=abs_file,
        rel_file=rel_file,
        target30_map=target_map,
        feature385_file=feature385_file,
    )
    x_val, y_val = build_region_inputs_from_split(
        val_split,
        abs_file=abs_file,
        rel_file=rel_file,
        target30_map=target_map,
        feature385_file=feature385_file,
    )
    print(
        f"[INFO] train={y_train.shape} val={y_val.shape} "
        f"feature385={sum(v.shape[1] for v in x_train.values())}"
    )

    region_indices = load_motor_region_indices(metrics_cfg=cfg.get("metrics", {}), dim=int(y_train.shape[1]))
    for key in ("brow", "eye", "jaw", "mouth"):
        if key not in region_indices:
            raise RuntimeError(f"missing region indices in metrics.motor_region_indices: '{key}'")
    region_indices_t = {
        k: torch.tensor(v, dtype=torch.long, device=device) for k, v in region_indices.items() if k in {"brow", "eye", "jaw", "mouth"}
    }

    train_ds = RegionalInputDataset(x_train, y_train)
    val_ds = RegionalInputDataset(x_val, y_val)

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MotorRegressorMLP().to(device)
    ema = ModelEMA(model, beta=ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping"]["patience"])
    min_delta = float(train_cfg["early_stopping"]["min_delta"])

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history_path = output_dir / "train_history.csv"
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"

    with history_path.open("w", encoding="utf-8", newline="") as f_hist:
        writer = csv.writer(f_hist)
        writer.writerow(
            [
                "epoch",
                "train_task_smoothl1_weighted",
                "train_boundary_loss",
                "train_total_loss",
                "train_brow_loss",
                "train_eye_loss",
                "train_jaw_loss",
                "train_mouth_loss",
                "val_mae",
            ]
        )

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            task_loss_sum = 0.0
            boundary_loss_sum = 0.0
            total_loss_sum = 0.0
            region_loss_sums = {"brow": 0.0, "eye": 0.0, "jaw": 0.0, "mouth": 0.0}
            n = 0

            for x, y in train_loader:
                x = move_inputs_to_device(x, device=device)
                y = y.to(device, non_blocking=True)
                raw_pred = model(x)
                pred_for_task = (
                    torch.clamp(raw_pred, min=boundary_cfg["lo"], max=boundary_cfg["hi"])
                    if boundary_cfg["clamp_for_task_loss"]
                    else raw_pred
                )

                task_loss, per_region = compute_weighted_smooth_l1_task_loss(
                    pred=pred_for_task,
                    target=y,
                    region_indices_t=region_indices_t,
                    region_weights=region_weights,
                )
                boundary_loss = compute_boundary_loss_torch(raw_pred, lo=boundary_cfg["lo"], hi=boundary_cfg["hi"])
                if boundary_cfg["enable_boundary_loss"]:
                    loss = task_loss + float(boundary_cfg["boundary_loss_weight"]) * boundary_loss
                else:
                    loss = task_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ema.update(model)

                bs = int(y.shape[0])
                task_loss_sum += float(task_loss.item()) * bs
                boundary_loss_sum += float(boundary_loss.item()) * bs
                total_loss_sum += float(loss.item()) * bs
                for key in region_loss_sums:
                    region_loss_sums[key] += float(per_region[key].item()) * bs
                n += bs

            train_task_loss = task_loss_sum / max(n, 1)
            train_boundary_loss = boundary_loss_sum / max(n, 1)
            train_total_loss = total_loss_sum / max(n, 1)
            train_region_losses = {k: v / max(n, 1) for k, v in region_loss_sums.items()}

            eval_model = ema.ema_model if use_ema_for_val else model
            val_mae = evaluate_mae(
                eval_model,
                val_loader,
                device,
                clip_predictions=bool(boundary_cfg["clip_predictions_in_eval"]),
                lo=float(boundary_cfg["lo"]),
                hi=float(boundary_cfg["hi"]),
            )

            writer.writerow(
                [
                    epoch,
                    train_task_loss,
                    train_boundary_loss,
                    train_total_loss,
                    train_region_losses["brow"],
                    train_region_losses["eye"],
                    train_region_losses["jaw"],
                    train_region_losses["mouth"],
                    val_mae,
                ]
            )
            f_hist.flush()

            improved = val_mae < (best_val - min_delta)
            if improved:
                best_val = val_mae
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_mae": best_val,
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.ema_model.state_dict(),
                        "ema_decay": ema_decay,
                        "used_ema_for_val": use_ema_for_val,
                        "config": cfg,
                    },
                    best_ckpt,
                )
            else:
                bad_epochs += 1

            print(
                f"[EPOCH {epoch:03d}] "
                f"task={train_task_loss:.6f} "
                f"boundary={train_boundary_loss:.6f} "
                f"total={train_total_loss:.6f} "
                f"brow={train_region_losses['brow']:.6f} "
                f"eye={train_region_losses['eye']:.6f} "
                f"jaw={train_region_losses['jaw']:.6f} "
                f"mouth={train_region_losses['mouth']:.6f} "
                f"val_mae={val_mae:.6f} best={best_val:.6f}"
            )

            if epoch >= min_epochs_before_early_stop and bad_epochs >= patience:
                print(f"[INFO] early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break

        elapsed = time.time() - t0

    torch.save(
        {
            "epoch": epoch,
            "best_val_mae": best_val,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema_model.state_dict(),
            "ema_decay": ema_decay,
            "used_ema_for_val": use_ema_for_val,
            "config": cfg,
        },
        last_ckpt,
    )

    summary = {
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "device": str(device),
        "train_samples": int(y_train.shape[0]),
        "val_samples": int(y_val.shape[0]),
        "feature_dim_total": int(sum(v.shape[1] for v in x_train.values())),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "boundary_train_cfg": boundary_cfg,
        "region_loss_weights": region_weights,
        "ema": {"enabled": True, "decay": ema_decay, "used_for_val": use_ema_for_val},
        "early_stopping": {
            "patience": patience,
            "min_delta": min_delta,
            "min_epochs_before_early_stop": min_epochs_before_early_stop,
        },
        "paths": {
            "best_ckpt": str(best_ckpt),
            "last_ckpt": str(last_ckpt),
            "history_csv": str(history_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] best_val_mae={best_val:.6f} at epoch={best_epoch}")
    print(f"[DONE] output_dir={output_dir}")


if __name__ == "__main__":
    main()
