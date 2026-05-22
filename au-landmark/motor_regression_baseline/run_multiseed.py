#!/usr/bin/env python3
"""Run compare6/B1vnext multi-seed experiments and aggregate results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import yaml


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float | None]:
    vals = [r.get(key) for r in rows]
    arr = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _best_seed(rows: List[Dict[str, Any]], key: str) -> int | None:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    best = min(valid, key=lambda x: float(x[key]))
    return int(best["seed"])


def _extract_seed_row(seed: int, seed_dir: Path) -> Dict[str, Any]:
    train_summary = _read_json(seed_dir / "train_summary.json")
    val_metrics = _read_json(seed_dir / "val_metrics.json")
    test_metrics = _read_json(seed_dir / "test_metrics.json")

    explain_dir = seed_dir / "explainability" / "test_best"
    region_corr_path = explain_dir / "region_corr_stats.json"
    residual_path = explain_dir / "residual_head_contribution.json"
    region_corr = _read_json(region_corr_path) if region_corr_path.exists() else None
    residual = _read_json(residual_path) if residual_path.exists() else None

    alpha = train_summary.get("alpha", {})
    region_metrics = test_metrics.get("region_metrics", {})
    mae_per_dim = test_metrics.get("mae_per_dim", [])

    def _motor_mae(idx: int) -> float | None:
        if not isinstance(mae_per_dim, list) or idx >= len(mae_per_dim):
            return None
        return float(mae_per_dim[idx])

    row: Dict[str, Any] = {
        "seed": int(seed),
        "best_epoch": int(train_summary.get("best_epoch", -1)),
        "best_val_mae": float(train_summary.get("best_val_mae")) if train_summary.get("best_val_mae") is not None else None,
        "val_mae": float(val_metrics.get("mae")) if val_metrics.get("mae") is not None else None,
        "val_rmse": float(val_metrics.get("rmse")) if val_metrics.get("rmse") is not None else None,
        "val_r2": float(val_metrics.get("r2")) if val_metrics.get("r2") is not None else None,
        "test_mae": float(test_metrics.get("mae")) if test_metrics.get("mae") is not None else None,
        "test_rmse": float(test_metrics.get("rmse")) if test_metrics.get("rmse") is not None else None,
        "test_r2": float(test_metrics.get("r2")) if test_metrics.get("r2") is not None else None,
        "test_explained_variance": float(test_metrics.get("explained_variance"))
        if test_metrics.get("explained_variance") is not None
        else None,
        "brow_mae": float(region_metrics.get("brow", {}).get("mae")) if region_metrics.get("brow", {}).get("mae") is not None else None,
        "eye_mae": float(region_metrics.get("eye", {}).get("mae")) if region_metrics.get("eye", {}).get("mae") is not None else None,
        "jaw_mae": float(region_metrics.get("jaw", {}).get("mae")) if region_metrics.get("jaw", {}).get("mae") is not None else None,
        "mouth_mae": float(region_metrics.get("mouth", {}).get("mae")) if region_metrics.get("mouth", {}).get("mae") is not None else None,
        "motor29_mae": _motor_mae(29),
        "motor13_mae": _motor_mae(13),
        "motor2_mae": _motor_mae(2),
        "motor3_mae": _motor_mae(3),
        "motor0_mae": _motor_mae(0),
        "motor1_mae": _motor_mae(1),
        "motor6_mae": _motor_mae(6),
        "motor7_mae": _motor_mae(7),
        "alpha_b": float(alpha.get("alpha_b")) if alpha.get("alpha_b") is not None else None,
        "alpha_m": float(alpha.get("alpha_m")) if alpha.get("alpha_m") is not None else None,
        "raw_out_of_range_ratio": float(
            test_metrics.get("boundary_constraint", {}).get("raw_prediction_boundary", {}).get("ratio")
        )
        if test_metrics.get("boundary_constraint", {}).get("raw_prediction_boundary", {}).get("ratio") is not None
        else None,
        "final_out_of_range_ratio": float(
            test_metrics.get("boundary_constraint", {}).get("final_prediction_boundary", {}).get("ratio")
        )
        if test_metrics.get("boundary_constraint", {}).get("final_prediction_boundary", {}).get("ratio") is not None
        else None,
        "run_dir": str(seed_dir),
        "region_corr_loaded": bool(region_corr is not None),
        "residual_contribution_loaded": bool(residual is not None),
    }

    if isinstance(region_corr, Mapping):
        matched = region_corr.get("matched_region_stats", {})
        for region in ("brow", "eye", "jaw", "mouth"):
            val = matched.get(region, {}).get("mean_abs_corr") if isinstance(matched, Mapping) else None
            row[f"region_corr_{region}_mean_abs"] = float(val) if val is not None else None

    if isinstance(residual, Mapping):
        row["motor29_base_mean_abs"] = (
            float(residual.get("motor29_base_mean_abs")) if residual.get("motor29_base_mean_abs") is not None else None
        )
        row["motor29_mouth_residual_mean_abs"] = (
            float(residual.get("motor29_mouth_residual_mean_abs"))
            if residual.get("motor29_mouth_residual_mean_abs") is not None
            else None
        )

    return row


def aggregate_results(base_output_dir: Path, seeds: List[int]) -> tuple[Path, Path]:
    rows: List[Dict[str, Any]] = []
    skipped_or_failed: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = base_output_dir / f"seed_{seed}"
        try:
            row = _extract_seed_row(seed=seed, seed_dir=seed_dir)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            skipped_or_failed.append({"seed": int(seed), "reason": str(exc), "seed_dir": str(seed_dir)})

    csv_fields = [
        "seed",
        "best_epoch",
        "best_val_mae",
        "val_mae",
        "val_rmse",
        "val_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_explained_variance",
        "brow_mae",
        "eye_mae",
        "jaw_mae",
        "mouth_mae",
        "motor29_mae",
        "motor13_mae",
        "motor2_mae",
        "motor3_mae",
        "motor0_mae",
        "motor1_mae",
        "motor6_mae",
        "motor7_mae",
        "alpha_b",
        "alpha_m",
        "raw_out_of_range_ratio",
        "final_out_of_range_ratio",
        "run_dir",
    ]
    csv_path = base_output_dir / "multiseed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["seed"]):
            writer.writerow({k: row.get(k) for k in csv_fields})

    summary = {
        "num_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "completed_seeds": [int(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])],
        "skipped_or_failed": skipped_or_failed,
        "test_mae": _stats(rows, "test_mae"),
        "test_rmse": _stats(rows, "test_rmse"),
        "test_r2": _stats(rows, "test_r2"),
        "brow_mae": _stats(rows, "brow_mae"),
        "eye_mae": _stats(rows, "eye_mae"),
        "jaw_mae": _stats(rows, "jaw_mae"),
        "mouth_mae": _stats(rows, "mouth_mae"),
        "motor29_mae": _stats(rows, "motor29_mae"),
        "motor13_mae": _stats(rows, "motor13_mae"),
        "motor2_mae": _stats(rows, "motor2_mae"),
        "motor3_mae": _stats(rows, "motor3_mae"),
        "alpha_b": _stats(rows, "alpha_b"),
        "alpha_m": _stats(rows, "alpha_m"),
        "best_seed_by_test_mae": _best_seed(rows, "test_mae"),
        "best_seed_by_val_mae": _best_seed(rows, "val_mae"),
        "best_seed_by_motor29_mae": _best_seed(rows, "motor29_mae"),
        "per_seed": sorted(rows, key=lambda x: x["seed"]),
        "paths": {
            "summary_csv": str(csv_path),
        },
    }
    summary_path = base_output_dir / "multiseed_summary.json"
    summary["paths"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B1vnext multi-seed experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ms_cfg = cfg.get("multi_seed", {})
    if not _as_bool(ms_cfg.get("enabled", False), default=False):
        raise RuntimeError("multi_seed.enabled is false in config; set it true to run this script.")

    seeds = [int(x) for x in ms_cfg.get("seeds", [])]
    if not seeds:
        raise RuntimeError("multi_seed.seeds is empty.")

    base_output_dir = Path(str(ms_cfg.get("base_output_dir", ""))).expanduser()
    if str(base_output_dir) == ".":
        raise RuntimeError("multi_seed.base_output_dir is not configured.")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_name_prefix = str(ms_cfg.get("run_name_prefix", "compare9_b1vnext_seed"))
    run_test_after_train = _as_bool(ms_cfg.get("run_test_after_train", True), default=True)
    run_explainability_after_test = _as_bool(ms_cfg.get("run_explainability_after_test", True), default=True)
    aggregate_enabled = _as_bool(ms_cfg.get("aggregate_results", True), default=True)
    skip_existing = _as_bool(ms_cfg.get("skip_existing", True), default=True)

    workdir = Path(__file__).resolve().parent
    tmp_cfg_dir = base_output_dir / "_tmp_seed_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    failed: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = base_output_dir / f"seed_{seed}"
        done_flag = seed_dir / "test_metrics.json"

        if skip_existing and done_flag.exists():
            print("=" * 50)
            print(f"Skipping seed {seed} (existing result found)")
            print(f"Output dir: {seed_dir}")
            print("=" * 50)
            continue

        print("=" * 50)
        print(f"Running seed {seed}")
        print(f"Output dir: {seed_dir}")
        print("=" * 50)

        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["seed"] = int(seed)
        cfg_seed["run_name"] = f"{run_name_prefix}_{seed}"
        cfg_seed.setdefault("train", {})["seed"] = int(seed)
        cfg_seed["train"]["output_dir"] = str(seed_dir)
        cfg_seed["train"]["use_run_subdir"] = False
        cfg_seed["train"]["allow_existing_run"] = True
        cfg_seed["train"]["run_name"] = ""

        cfg_seed.setdefault("eval", {})["run_name"] = "latest"
        cfg_seed["eval"]["ckpt_file"] = "best.pt"

        tmp_cfg_path = tmp_cfg_dir / f"seed_{seed}.yaml"
        tmp_cfg_path.write_text(yaml.safe_dump(cfg_seed, allow_unicode=True, sort_keys=False), encoding="utf-8")

        try:
            _run_cmd([sys.executable, "train.py", "--config", str(tmp_cfg_path)], cwd=workdir)
            best_ckpt = seed_dir / "best.pt"
            _run_cmd([sys.executable, "val.py", "--config", str(tmp_cfg_path), "--ckpt", str(best_ckpt)], cwd=workdir)

            if run_test_after_train:
                _run_cmd([sys.executable, "test.py", "--config", str(tmp_cfg_path), "--ckpt", str(best_ckpt)], cwd=workdir)

            if run_test_after_train and run_explainability_after_test:
                _run_cmd(
                    [
                        sys.executable,
                        "explainability.py",
                        "--config",
                        str(tmp_cfg_path),
                        "--ckpt",
                        str(best_ckpt),
                        "--split",
                        "test",
                    ],
                    cwd=workdir,
                )
        except Exception as exc:  # noqa: BLE001
            failed.append({"seed": int(seed), "seed_dir": str(seed_dir), "reason": str(exc)})
            print(f"[ERROR] seed {seed} failed: {exc}")
            continue

        try:
            train_summary = _read_json(seed_dir / "train_summary.json")
            test_metrics = _read_json(seed_dir / "test_metrics.json")
            motor29_mae = None
            if isinstance(test_metrics.get("mae_per_dim"), list) and len(test_metrics["mae_per_dim"]) > 29:
                motor29_mae = float(test_metrics["mae_per_dim"][29])
            print(f"Seed {seed} finished:")
            print(f"best_val_mae = {train_summary.get('best_val_mae')}")
            print(f"test_mae = {test_metrics.get('mae')}")
            print(f"test_r2 = {test_metrics.get('r2')}")
            print(f"motor29_mae = {motor29_mae}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] seed {seed} finished but summary print failed: {exc}")

    if aggregate_enabled:
        csv_path, json_path = aggregate_results(base_output_dir=base_output_dir, seeds=seeds)
        summary = _read_json(json_path)
        print("Multi-seed summary:")
        test_mae_stats = summary.get("test_mae", {})
        print(f"test_mae mean ± std = {test_mae_stats.get('mean')} ± {test_mae_stats.get('std')}")
        print(f"best seed (test_mae) = {summary.get('best_seed_by_test_mae')}")
        print(f"summary csv = {csv_path}")
        print(f"summary json = {json_path}")
        if failed:
            print(f"[WARN] failed seeds: {failed}")


if __name__ == "__main__":
    main()

