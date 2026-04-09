#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FEATURE385 = (ABS+REL 382) + pose(3) into data_compare3.")
    p.add_argument("--abs-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\ABS_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--rel-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\REL_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--target-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\metadata_normalize.jsonl"))
    p.add_argument("--output-feature-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_compare3\FEATURE385_X2C_gpu.csv.gz"))
    p.add_argument("--output-target-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_compare3\metadata_normalize.jsonl"))
    p.add_argument("--output-columns-json", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_compare3\FEATURE385_X2C_gpu.csv.gz.columns.json"))
    p.add_argument("--output-summary-json", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_compare3\FEATURE385_X2C_gpu.csv.gz.summary.json"))
    return p.parse_args()


def lmk_abs_cols(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_abs_norm_{i:02d}_x", f"lmk_abs_norm_{i:02d}_y", f"lmk_abs_norm_{i:02d}_z"])
    return cols


def lmk_rel_cols(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_rel_{i:02d}_x", f"lmk_rel_{i:02d}_y", f"lmk_rel_{i:02d}_z"])
    return cols


def au_abs_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_abs_intensity" for n in names]


def au_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def dist_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def build_feature_layout() -> List[Dict[str, object]]:
    return [
        {"region": "brow", "source": "abs", "columns": au_abs_cols(BROW_AU) + lmk_abs_cols(0, 9) + BROW_DIST},
        {"region": "brow", "source": "rel", "columns": au_rel_cols(BROW_AU) + lmk_rel_cols(0, 9) + dist_rel_cols(BROW_DIST)},
        {"region": "eye", "source": "abs", "columns": au_abs_cols(EYE_AU) + lmk_abs_cols(10, 21) + EYE_DIST},
        {"region": "eye", "source": "rel", "columns": au_rel_cols(EYE_AU) + lmk_rel_cols(10, 21) + dist_rel_cols(EYE_DIST)},
        {"region": "mouth", "source": "abs", "columns": au_abs_cols(MOUTH_AU) + lmk_abs_cols(22, 41) + MOUTH_DIST},
        {"region": "mouth", "source": "rel", "columns": au_rel_cols(MOUTH_AU) + lmk_rel_cols(22, 41) + dist_rel_cols(MOUTH_DIST)},
        {"region": "jaw", "source": "abs", "columns": au_abs_cols(JAW_AU) + lmk_abs_cols(42, 49) + JAW_DIST},
        {"region": "jaw", "source": "rel", "columns": au_rel_cols(JAW_AU) + lmk_rel_cols(42, 49) + dist_rel_cols(JAW_DIST)},
        {"region": "global", "source": "abs", "columns": ["yaw", "pitch", "roll"]},
        {"region": "global", "source": "rel", "columns": ["ENERGY_rel"]},
        {"region": "pose", "source": "pose", "columns": ["yaw", "pitch", "roll"]},
    ]


def open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_float(v: object) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "":
        return 0.0
    return float(s)


def build_columns_map(layout: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    feat_idx = 0
    for block in layout:
        for col in block["columns"]:
            out.append(
                {
                    "feat_idx": feat_idx,
                    "feat_name": f"feat_{feat_idx:03d}",
                    "region": block["region"],
                    "source": block["source"],
                    "source_column": str(col),
                }
            )
            feat_idx += 1
    return out


def main() -> None:
    args = parse_args()
    if not args.abs_file.exists():
        raise FileNotFoundError(f"ABS file not found: {args.abs_file}")
    if not args.rel_file.exists():
        raise FileNotFoundError(f"REL file not found: {args.rel_file}")
    if not args.target_file.exists():
        raise FileNotFoundError(f"target file not found: {args.target_file}")

    args.output_feature_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_target_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_columns_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    layout = build_feature_layout()
    feature_columns = [col for block in layout for col in block["columns"]]
    feature_dim = len(feature_columns)
    if feature_dim != 385:
        raise RuntimeError(f"feature dim mismatch: expect 385, got {feature_dim}")

    output_cols = [
        "image_path",
        "image_name",
        "face_found",
        "face_detect_conf",
        "landmark_conf",
        "error",
    ] + [f"feat_{i:03d}" for i in range(feature_dim)]

    t0 = time.time()
    rows = 0
    path_mismatch = 0
    name_mismatch = 0

    with open_text(args.abs_file, "rt") as f_abs, open_text(args.rel_file, "rt") as f_rel, open_text(args.output_feature_file, "wt") as f_out:
        ra = csv.DictReader(f_abs)
        rr = csv.DictReader(f_rel)

        needed_abs = []
        needed_rel = []
        for block in layout:
            if block["source"] in {"abs", "pose"}:
                needed_abs.extend(block["columns"])
            if block["source"] == "rel":
                needed_rel.extend(block["columns"])
        needed_abs.extend(["image_path", "image_name", "face_found", "face_detect_conf", "landmark_conf", "error"])
        needed_rel.extend(["image_path", "image_name"])

        miss_abs = [c for c in needed_abs if c not in (ra.fieldnames or [])]
        miss_rel = [c for c in needed_rel if c not in (rr.fieldnames or [])]
        if miss_abs:
            raise RuntimeError(f"ABS missing columns: {miss_abs[:8]}")
        if miss_rel:
            raise RuntimeError(f"REL missing columns: {miss_rel[:8]}")

        writer = csv.DictWriter(f_out, fieldnames=output_cols)
        writer.writeheader()

        for abs_row, rel_row in zip(ra, rr):
            if abs_row.get("image_path", "") != rel_row.get("image_path", ""):
                path_mismatch += 1
            if abs_row.get("image_name", "") != rel_row.get("image_name", ""):
                name_mismatch += 1

            feat_values: List[float] = []
            for block in layout:
                src = abs_row if block["source"] in {"abs", "pose"} else rel_row
                for col in block["columns"]:
                    feat_values.append(safe_float(src.get(col, 0.0)))

            out: Dict[str, object] = {
                "image_path": abs_row.get("image_path", ""),
                "image_name": abs_row.get("image_name", ""),
                "face_found": int(float(abs_row.get("face_found", 0) or 0)),
                "face_detect_conf": safe_float(abs_row.get("face_detect_conf", 0.0)),
                "landmark_conf": safe_float(abs_row.get("landmark_conf", 0.0)),
                "error": abs_row.get("error", ""),
            }
            for i, v in enumerate(feat_values):
                out[f"feat_{i:03d}"] = float(v)
            writer.writerow(out)
            rows += 1

    shutil.copyfile(args.target_file, args.output_target_file)

    columns_map = build_columns_map(layout)
    args.output_columns_json.write_text(json.dumps(columns_map, ensure_ascii=False, indent=2), encoding="utf-8")

    region_feature_ranges: Dict[str, Dict[str, int]] = {}
    start = 0
    for region in ["brow", "eye", "mouth", "jaw", "global", "pose"]:
        dim = sum(len(block["columns"]) for block in layout if block["region"] == region)
        region_feature_ranges[region] = {"start": start, "end": start + dim - 1, "dim": dim}
        start += dim

    elapsed = time.time() - t0
    summary = {
        "abs_file": str(args.abs_file),
        "rel_file": str(args.rel_file),
        "target_file": str(args.target_file),
        "output_feature_file": str(args.output_feature_file),
        "output_target_file": str(args.output_target_file),
        "rows_processed": rows,
        "feature_dim": feature_dim,
        "feature_layout": [{"region": b["region"], "source": b["source"], "dim": len(b["columns"])} for b in layout],
        "region_feature_ranges": region_feature_ranges,
        "row_mismatch": {
            "image_path_mismatch": path_mismatch,
            "image_name_mismatch": name_mismatch,
        },
        "elapsed_seconds": round(elapsed, 3),
        "avg_rows_per_sec": round(rows / max(elapsed, 1e-6), 4),
        "notes": [
            "FEATURE385 = (ABS+REL 382) + pose(yaw,pitch,roll) 3",
            "feature order = brow(abs+rel), eye(abs+rel), mouth(abs+rel), jaw(abs+rel), global(abs+rel), pose",
        ],
    }
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] rows={rows} feature_dim={feature_dim}")
    print(f"[DONE] feature={args.output_feature_file}")
    print(f"[DONE] target={args.output_target_file}")
    print(f"[DONE] columns={args.output_columns_json}")
    print(f"[DONE] summary={args.output_summary_json}")


if __name__ == "__main__":
    main()
