#!/usr/bin/env python3
"""Run/checkpoint path resolution helpers.

This module centralizes:
- training output directory policy (`run_xxx` creation),
- evaluation checkpoint resolution (`--ckpt`, explicit run name, or latest run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _extract_run_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix) :]
    if tail.isdigit():
        return int(tail)
    return None


def _latest_run_dir(output_root: Path, prefix: str) -> Path:
    candidates = []
    for p in output_root.iterdir():
        if not p.is_dir():
            continue
        idx = _extract_run_index(p.name, prefix)
        if idx is None:
            continue
        candidates.append((idx, p))
    if not candidates:
        raise RuntimeError(f"no run directories found under: {output_root} with prefix '{prefix}'")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _next_run_dir(output_root: Path, prefix: str, digits: int) -> Path:
    max_idx = 0
    for p in output_root.iterdir():
        if not p.is_dir():
            continue
        idx = _extract_run_index(p.name, prefix)
        if idx is not None and idx > max_idx:
            max_idx = idx
    next_idx = max_idx + 1
    return output_root / f"{prefix}{next_idx:0{digits}d}"


def resolve_train_output_dir(train_cfg: Mapping[str, object]) -> Tuple[Path, str | None]:
    """Resolve train output directory from `train` config.

    Returns:
    - output_dir: actual directory used by current training run
    - run_name: run directory name when subdir mode is enabled, otherwise None
    """
    output_root = Path(str(train_cfg["output_dir"]))
    use_run_subdir = _as_bool(train_cfg.get("use_run_subdir", True), default=True)
    run_prefix = str(train_cfg.get("run_prefix", "run_"))
    run_digits = int(train_cfg.get("run_digits", 3))
    run_name_cfg = str(train_cfg.get("run_name", "")).strip()
    allow_existing_run = _as_bool(train_cfg.get("allow_existing_run", False), default=False)

    output_root.mkdir(parents=True, exist_ok=True)
    if not use_run_subdir:
        return output_root, None

    if run_name_cfg:
        output_dir = output_root / run_name_cfg
    else:
        output_dir = _next_run_dir(output_root, prefix=run_prefix, digits=run_digits)

    if output_dir.exists() and not allow_existing_run:
        raise RuntimeError(
            f"run directory already exists: {output_dir}. "
            "Set train.allow_existing_run=true or choose another train.run_name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, output_dir.name


def resolve_eval_ckpt_path(cfg: Mapping[str, object], explicit_ckpt: Path | None) -> Tuple[Path, Path, str | None]:
    """Resolve checkpoint path for validation/test/explainability.

    Priority:
    1) explicit `--ckpt`
    2) config.eval.{run_name,ckpt_file} with config.train.output_dir

    Returns:
    - ckpt_path: checkpoint file path
    - output_dir: output directory for writing eval artifacts
    - run_name: resolved run name, None when subdir mode is disabled
    """
    if explicit_ckpt is not None:
        ckpt_path = Path(explicit_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
        return ckpt_path, ckpt_path.parent, ckpt_path.parent.name

    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})

    output_root = Path(str(train_cfg["output_dir"]))
    use_run_subdir = _as_bool(train_cfg.get("use_run_subdir", True), default=True)
    run_prefix = str(train_cfg.get("run_prefix", "run_"))
    ckpt_file = str(eval_cfg.get("ckpt_file", "best.pt"))

    if not use_run_subdir:
        ckpt_path = output_root / ckpt_file
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
        return ckpt_path, output_root, None

    run_name = str(eval_cfg.get("run_name", "latest")).strip()
    if run_name == "" or run_name.lower() == "latest":
        if not output_root.exists():
            raise FileNotFoundError(f"train.output_dir not found: {output_root}")
        try:
            run_dir = _latest_run_dir(output_root, prefix=run_prefix)
        except RuntimeError:
            # Backward compatibility: fallback to root-level checkpoint layout.
            legacy_ckpt = output_root / ckpt_file
            if legacy_ckpt.exists():
                return legacy_ckpt, output_root, None
            raise
    else:
        run_dir = output_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory not found: {run_dir}")

    ckpt_path = run_dir / ckpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    return ckpt_path, run_dir, run_dir.name
