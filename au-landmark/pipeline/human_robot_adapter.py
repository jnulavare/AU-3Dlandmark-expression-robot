#!/usr/bin/env python3
"""Human -> Robot feature adapter for compare6/B1vnext inference.

This module only handles feature adaptation. It does not run the model.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np
try:
    from pipeline.feature385_builder import load_feature_columns
except ModuleNotFoundError:
    from feature385_builder import load_feature_columns

# compare6 FEATURE385 canonical layout (used in data_utils.py fast path):
# [0:37]    brow_abs
# [37:74]   brow_rel
# [74:119]  eye_abs
# [119:164] eye_rel
# [164:243] mouth_abs
# [243:322] mouth_rel
# [322:350] jaw_abs
# [350:378] jaw_rel
# [378:381] global_abs
# [381:382] global_rel
# [382:385] pose(yaw, pitch, roll)
REGION_ABS_DIMS: Dict[str, int] = {
    "brow_abs": 37,
    "eye_abs": 45,
    "mouth_abs": 79,
    "jaw_abs": 28,
    "global_abs": 3,
}
REGION_REL_DIMS: Dict[str, int] = {
    "brow_rel": 37,
    "eye_rel": 45,
    "mouth_rel": 79,
    "jaw_rel": 28,
    "global_rel": 1,
}
POSE_DIM_DEFAULT = 3
ABS_DIM_DEFAULT = sum(REGION_ABS_DIMS.values())  # 192
REL_DIM_DEFAULT = sum(REGION_REL_DIMS.values())  # 190
FEATURE_DIM_TOTAL_DEFAULT = ABS_DIM_DEFAULT + REL_DIM_DEFAULT + POSE_DIM_DEFAULT  # 385
FEATURE_LAYOUT_NAME = "compare6_interleaved"
FEATURE385_PACK_ORDER = (
    "brow_abs,brow_rel,eye_abs,eye_rel,mouth_abs,mouth_rel,"
    "jaw_abs,jaw_rel,global_abs,global_rel,pose"
)

# internal abs192 slices
ABS_BROW = slice(0, 37)
ABS_EYE = slice(37, 82)
ABS_MOUTH = slice(82, 161)
ABS_JAW = slice(161, 189)
ABS_GLOBAL = slice(189, 192)

# internal rel190 slices
REL_BROW = slice(0, 37)
REL_EYE = slice(37, 82)
REL_MOUTH = slice(82, 161)
REL_JAW = slice(161, 189)
REL_GLOBAL = slice(189, 190)


def _to_np_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def pack_feature385_interleaved(abs_vec: np.ndarray, rel_vec: np.ndarray, pose_vec: np.ndarray) -> np.ndarray:
    a = _to_np_1d(abs_vec, "abs_vec")
    r = _to_np_1d(rel_vec, "rel_vec")
    p = _to_np_1d(pose_vec, "pose_vec")
    if a.shape[0] != 192 or r.shape[0] != 190 or p.shape[0] != 3:
        raise ValueError(f"pack_feature385_interleaved shape mismatch: abs={a.shape}, rel={r.shape}, pose={p.shape}")
    feat = np.concatenate(
        [
            a[ABS_BROW], r[REL_BROW],
            a[ABS_EYE], r[REL_EYE],
            a[ABS_MOUTH], r[REL_MOUTH],
            a[ABS_JAW], r[REL_JAW],
            a[ABS_GLOBAL], r[REL_GLOBAL],
            p,
        ],
        axis=0,
    ).astype(np.float32)
    if feat.shape[0] != 385:
        raise ValueError(f"packed feature dim mismatch: {feat.shape[0]}")
    return feat


def unpack_feature385_interleaved(feature385: np.ndarray) -> Dict[str, np.ndarray]:
    f = _to_np_1d(feature385, "feature385")
    if f.shape[0] != 385:
        raise ValueError(f"feature385 dim mismatch: got {f.shape[0]}, expect 385")
    brow_abs = f[0:37]
    brow_rel = f[37:74]
    eye_abs = f[74:119]
    eye_rel = f[119:164]
    mouth_abs = f[164:243]
    mouth_rel = f[243:322]
    jaw_abs = f[322:350]
    jaw_rel = f[350:378]
    global_abs = f[378:381]
    global_rel = f[381:382]
    pose = f[382:385]
    abs_vec = np.concatenate([brow_abs, eye_abs, mouth_abs, jaw_abs, global_abs], axis=0).astype(np.float32)
    rel_vec = np.concatenate([brow_rel, eye_rel, mouth_rel, jaw_rel, global_rel], axis=0).astype(np.float32)
    return {"abs": abs_vec, "rel": rel_vec, "pose": pose.astype(np.float32), "feature385": f.astype(np.float32)}


class HumanToRobotFeatureAdapter:
    """Build robot-style FEATURE385 from human neutral/expression pair.

    IMPORTANT:
    - Do not directly use human ABS as B1vnext ABS input.
    - We use robot_neutral_ABS + adapted_REL to build robot-style ABS.
      This reduces human/robot morphology gap in ABS channel.
    """

    def __init__(
        self,
        robot_neutral_feature_path: str | Path,
        feature_normalizer_path: str | Path | None = None,
        feature_dim_total: int = FEATURE_DIM_TOTAL_DEFAULT,
        abs_dim: int = ABS_DIM_DEFAULT,
        rel_dim: int = REL_DIM_DEFAULT,
        pose_dim: int = POSE_DIM_DEFAULT,
        device: str = "cpu",
        config: Mapping[str, Any] | None = None,
        clip_rel: bool = True,
        rel_clip_min: float = -3.0,
        rel_clip_max: float = 3.0,
    ) -> None:
        self.robot_neutral_feature_path = Path(robot_neutral_feature_path)
        self.feature_normalizer_path = Path(feature_normalizer_path) if feature_normalizer_path else None
        self.feature_dim_total = int(feature_dim_total)
        self.abs_dim = int(abs_dim)
        self.rel_dim = int(rel_dim)
        self.pose_dim = int(pose_dim)
        self.device = str(device)
        self.config = dict(config) if config else {}
        self.clip_rel = bool(clip_rel)
        self.rel_clip_min = float(rel_clip_min)
        self.rel_clip_max = float(rel_clip_max)

        self._warnings: List[str] = []
        self._normalizer_stats: Dict[str, Any] = self._load_normalizer_stats()
        self._columns_json_path: Optional[Path] = self._resolve_columns_json_path()
        self._feature_columns = self._load_feature_columns_optional()
        self.robot_neutral: Dict[str, np.ndarray] = self.load_robot_neutral_feature()
        self._fa_model = None
        self._abs_mod = None
        adapter_cfg = self.config.get("adapter", {}) if isinstance(self.config, Mapping) else {}
        self._aligned_size = int(adapter_cfg.get("aligned_size", 256))
        self._crop_expand = float(adapter_cfg.get("crop_expand", 1.25))
        self._scale_warn_ratio = float(adapter_cfg.get("scale_warn_ratio", 0.2))
        self._apply_scale_ratio_correction = bool(adapter_cfg.get("apply_scale_ratio_correction", False))
        self._last_mapping_info: Dict[str, Any] = {
            "rel_to_abs_mapping_used": False,
            "rel_to_abs_mapping_type": "unset",
            "global_abs_kept_neutral": True,
        }
        self._last_abs_delta: np.ndarray | None = None
        self._last_rel_missing_columns: List[str] = []
        self._rel190_order_name: str = "unknown"
        self._rel190_order_checked_by_columns_json: bool = bool(self._feature_columns)

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)
        print(f"[ADAPTER][WARN] {msg}")

    def _ensure_abs_extractor(self) -> None:
        """Lazy init for single-image ABS extractor from preprocess pipeline."""
        if self._fa_model is not None and self._abs_mod is not None:
            return

        project_dir = Path(__file__).resolve().parent.parent
        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))

        try:
            import preprocess.extract_abs_input_vec_gpu as abs_mod  # type: ignore
        except Exception as exc:  # pragma: no cover - import error depends on env
            raise RuntimeError(
                "Failed to import preprocess.extract_abs_input_vec_gpu. "
                "Please ensure project root is on PYTHONPATH."
            ) from exc

        torch_home = (
            self.config.get("adapter", {}).get("torch_home")
            if isinstance(self.config, Mapping)
            else None
        )
        if not torch_home:
            torch_home = r"D:/torch_cache"
        os.environ["TORCH_HOME"] = str(torch_home)
        Path(torch_home).mkdir(parents=True, exist_ok=True)

        try:
            import face_alignment as fa  # type: ignore
            import torch
        except Exception as exc:  # pragma: no cover - import error depends on env
            raise RuntimeError(
                "Missing dependency for face feature extraction. Install face-alignment and torch."
            ) from exc

        requested_device = str(self.device).strip().lower()
        detector = "sfd"
        if isinstance(self.config, Mapping):
            detector = str(self.config.get("adapter", {}).get("detector", "sfd"))

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self._warn("CUDA requested for adapter extractor but unavailable; fallback to CPU.")
            requested_device = "cpu"

        self._fa_model = fa.FaceAlignment(
            fa.LandmarksType.THREE_D,
            device=requested_device,
            face_detector=detector,
            flip_input=False,
            verbose=False,
        )
        self._abs_mod = abs_mod

    def _load_normalizer_stats(self) -> Dict[str, Any]:
        if self.feature_normalizer_path is None:
            return {}
        p = self.feature_normalizer_path
        if not p.exists():
            self._warn(f"feature_normalizer_path not found: {p}")
            return {}
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        if p.suffix.lower() in {".npz"}:
            npz = np.load(p, allow_pickle=True)
            out: Dict[str, Any] = {}
            for k in npz.files:
                out[k] = np.asarray(npz[k])
            return out
        self._warn(f"unsupported normalizer format: {p.suffix}")
        return {}

    def _resolve_columns_json_path(self) -> Optional[Path]:
        cfg = self.config if isinstance(self.config, Mapping) else {}
        adapter_cfg = cfg.get("adapter", {}) if isinstance(cfg.get("adapter", {}), Mapping) else {}

        candidates: List[Path] = []
        v = adapter_cfg.get("feature_columns_json", None)
        if v:
            candidates.append(Path(str(v)))
        v = cfg.get("feature_columns_json", None)
        if v:
            candidates.append(Path(str(v)))
        feature_file = cfg.get("feature_file", None)
        if feature_file:
            candidates.append(Path(str(feature_file) + ".columns.json"))

        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_feature_columns_optional(self) -> Optional[List[Dict[str, Any]]]:
        if self._columns_json_path is None:
            self._warn("feature columns json not found in config; REL190 order check fallback will be used.")
            return None
        try:
            return load_feature_columns(self._columns_json_path)
        except Exception as exc:
            self._warn(f"failed to load feature columns json: {self._columns_json_path} ({exc})")
            return None

    def _split_feature385(self, feature385: np.ndarray) -> Dict[str, np.ndarray]:
        f = _to_np_1d(feature385, "feature385")
        if f.shape[0] != self.feature_dim_total:
            raise ValueError(f"feature385 dim mismatch: got {f.shape[0]}, expect {self.feature_dim_total}")
        if self.feature_dim_total == 385:
            return unpack_feature385_interleaved(f)
        raise ValueError("Cannot split robot neutral feature. Please provide abs/rel/pose or schema.")

    def load_robot_neutral_feature(self) -> Dict[str, np.ndarray]:
        """Load robot neutral feature from .json/.npy/.npz/.csv."""
        p = self.robot_neutral_feature_path
        if not p.exists():
            raise FileNotFoundError(f"robot neutral feature not found: {p}")

        raw: Dict[str, Any] = {}
        suffix = p.suffix.lower()
        if suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
        elif suffix == ".npy":
            arr = np.load(p, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
                obj = arr.item()
                if isinstance(obj, dict):
                    raw = obj
                else:
                    raw = {"feature385": np.asarray(obj, dtype=np.float32).reshape(-1)}
            else:
                raw = {"feature385": np.asarray(arr, dtype=np.float32).reshape(-1)}
        elif suffix == ".npz":
            npz = np.load(p, allow_pickle=True)
            raw = {k: npz[k] for k in npz.files}
        elif suffix == ".csv":
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                raise ValueError(f"empty csv: {p}")
            row = rows[0]
            if "feature385" in row:
                raw = {"feature385": [float(x) for x in row["feature385"].split(",") if x.strip()]}
            else:
                feat_cols = [k for k in row.keys() if k.startswith("feat_")]
                if feat_cols:
                    feat_cols.sort()
                    raw = {"feature385": [float(row[c]) for c in feat_cols]}
                else:
                    raise ValueError("csv must contain feature385 or feat_000... columns")
        else:
            raise ValueError(f"unsupported robot neutral format: {suffix}")

        out: Dict[str, np.ndarray]
        has_abs_rel_pose = all(k in raw for k in ("abs", "rel", "pose"))
        if has_abs_rel_pose:
            abs_vec = _to_np_1d(raw["abs"], "robot_neutral.abs")
            rel_vec = _to_np_1d(raw["rel"], "robot_neutral.rel")
            pose = _to_np_1d(raw["pose"], "robot_neutral.pose")
            out = {"abs": abs_vec, "rel": rel_vec, "pose": pose}
            out["feature385"] = self.build_feature385(abs_vec, rel_vec, pose)
        elif "feature385" in raw:
            out = self._split_feature385(_to_np_1d(raw["feature385"], "robot_neutral.feature385"))
        else:
            raise ValueError("Cannot split robot neutral feature. Please provide abs/rel/pose or schema.")

        if out["abs"].shape[0] != self.abs_dim or out["rel"].shape[0] != self.rel_dim or out["pose"].shape[0] != self.pose_dim:
            raise ValueError(
                f"robot neutral dim mismatch: abs={out['abs'].shape[0]} rel={out['rel'].shape[0]} pose={out['pose'].shape[0]}, "
                f"expect abs={self.abs_dim} rel={self.rel_dim} pose={self.pose_dim}"
            )
        return out

    def _pick_best_face(self, bboxes: list[np.ndarray], landmarks_all: list[np.ndarray], w: int, h: int) -> int:
        center = np.array([w * 0.5, h * 0.5], dtype=np.float64)
        best_i = 0
        best_score = -1e18
        for i, (bb, lmk) in enumerate(zip(bboxes, landmarks_all)):
            x1, y1, x2, y2 = bb[:4]
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            area = bw * bh
            area_score = area / max(float(w * h), 1.0)
            conf = float(bb[4]) if bb.shape[0] > 4 else 0.5
            c = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            center_dist = float(np.linalg.norm(c - center))
            center_score = 1.0 - center_dist / max(float(np.hypot(w, h)), 1.0)
            try:
                _, yaw, pitch, roll = self._abs_mod.estimate_pose(np.asarray(lmk, dtype=np.float64)[:, :2], w, h)
                pose_penalty = (abs(yaw) + abs(pitch) + abs(roll)) / 180.0
            except Exception:
                pose_penalty = 0.5
            score = 0.45 * area_score + 0.30 * center_score + 0.25 * conf - 0.20 * pose_penalty
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def preprocess_face_image(self, image_path: str | Path) -> tuple[np.ndarray, Dict[str, Any]]:
        """Face preprocess before AU/landmark extraction.

        Pipeline:
        read -> detect faces -> pick best -> align -> crop ROI -> fixed resize.
        """
        self._ensure_abs_extractor()
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise RuntimeError(f"imread failed: {p}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = self._fa_model.get_landmarks_from_image(
            img_rgb,
            return_bboxes=True,
            return_landmark_score=True,
        )
        if out is None:
            raise RuntimeError(f"no face output for image: {p}")
        landmarks_all, scores_all, bboxes_all = out
        if landmarks_all is None or len(landmarks_all) == 0:
            raise RuntimeError(f"no face detected for image: {p}")

        bboxes = [np.asarray(bb, dtype=np.float64) for bb in bboxes_all]
        landmarks = [np.asarray(lmk, dtype=np.float64) for lmk in landmarks_all]
        best_i = self._pick_best_face(bboxes=bboxes, landmarks_all=landmarks, w=w, h=h)
        bb = bboxes[best_i]
        lmk = landmarks[best_i]

        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        conf = float(bb[4]) if bb.shape[0] > 4 else 0.0
        try:
            _, yaw, pitch, roll = self._abs_mod.estimate_pose(lmk[:, :2], w, h)
        except Exception:
            yaw, pitch, roll = 0.0, 0.0, 0.0

        inter_eye_distance = float(np.linalg.norm(lmk[45, :2] - lmk[36, :2])) if lmk.shape[0] >= 46 else 0.0

        # Rotate around face center to reduce roll.
        M = cv2.getRotationMatrix2D((cx, cy), -roll, 1.0)
        rotated = cv2.warpAffine(
            img_bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Expand bbox and crop ROI from aligned image.
        half_w = 0.5 * bw * self._crop_expand
        half_h = 0.5 * bh * self._crop_expand
        rx1 = int(max(0, min(w - 1, round(cx - half_w))))
        ry1 = int(max(0, min(h - 1, round(cy - half_h))))
        rx2 = int(max(0, min(w, round(cx + half_w))))
        ry2 = int(max(0, min(h, round(cy + half_h))))
        if rx2 <= rx1 or ry2 <= ry1:
            raise RuntimeError("invalid ROI after alignment crop")
        face_roi = rotated[ry1:ry2, rx1:rx2]
        aligned = cv2.resize(face_roi, (self._aligned_size, self._aligned_size), interpolation=cv2.INTER_LINEAR)

        scale_used = inter_eye_distance if inter_eye_distance > 1e-6 else max(bw, bh)
        if scale_used <= 1e-6:
            scale_used = 1.0

        face_meta: Dict[str, Any] = {
            "bbox": [int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))],
            "face_center": [float(cx), float(cy)],
            "face_width": float(bw),
            "face_height": float(bh),
            "inter_eye_distance": float(inter_eye_distance),
            "scale_used": float(scale_used),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "detection_confidence": float(conf),
            "image_size": [int(w), int(h)],
            "aligned_size": [int(self._aligned_size), int(self._aligned_size)],
        }
        return aligned, face_meta

    def normalize_landmarks(self, landmarks: np.ndarray, face_meta: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        """Normalize landmarks to reduce resolution/camera-distance effects.

        x_norm = (x - cx) / s
        y_norm = (y - cy) / s
        z_norm = z / s
        """
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim != 2 or lm.shape[1] < 3:
            raise ValueError(f"landmarks must be [N,3], got shape={lm.shape}")
        cx = float(face_meta.get("face_center", [0.0, 0.0])[0])
        cy = float(face_meta.get("face_center", [0.0, 0.0])[1])
        inter_eye = float(face_meta.get("inter_eye_distance", 0.0) or 0.0)
        fw = float(face_meta.get("face_width", 0.0) or 0.0)
        fh = float(face_meta.get("face_height", 0.0) or 0.0)
        if inter_eye > 1e-6:
            scale = inter_eye
        elif fw > 1e-6:
            scale = fw
        elif max(fw, fh) > 1e-6:
            scale = max(fw, fh)
        else:
            scale = 1.0
            self._warn("normalize_landmarks fallback scale=1.0 (missing face scale).")

        out = lm.copy()
        out[:, 0] = (out[:, 0] - cx) / scale
        out[:, 1] = (out[:, 1] - cy) / scale
        out[:, 2] = out[:, 2] / scale
        return out.astype(np.float32), float(scale)

    def extract_face_feature(self, image_path: str | Path) -> Dict[str, Any]:
        """Extract normalized face feature for REL computation."""
        self._ensure_abs_extractor()
        aligned_bgr, face_meta = self.preprocess_face_image(image_path)
        ah, aw = aligned_bgr.shape[:2]
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        out = self._fa_model.get_landmarks_from_image(
            aligned_rgb,
            return_bboxes=True,
            return_landmark_score=True,
        )
        if out is None:
            raise RuntimeError(f"no landmarks after preprocess: {image_path}")
        landmarks_all, scores_all, bboxes_all = out
        if landmarks_all is None or len(landmarks_all) == 0:
            raise RuntimeError(f"no face after preprocess: {image_path}")
        # best on aligned image: highest detector confidence.
        best_i = 0
        best_conf = -1.0
        for i, bb in enumerate(bboxes_all):
            s = float(bb[4]) if len(bb) > 4 else 0.0
            if s > best_conf:
                best_conf = s
                best_i = i
        lmk68 = np.asarray(landmarks_all[best_i], dtype=np.float64)
        bb = np.asarray(bboxes_all[best_i], dtype=np.float64)
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        inter_eye = float(np.linalg.norm(lmk68[45, :2] - lmk68[36, :2])) if lmk68.shape[0] >= 46 else 0.0
        _, yaw, pitch, roll = self._abs_mod.estimate_pose(lmk68[:, :2], aw, ah)
        face_meta_aligned = dict(face_meta)
        face_meta_aligned.update(
            {
                "bbox": [int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))],
                "face_center": [float(cx), float(cy)],
                "face_width": float(bw),
                "face_height": float(bh),
                "inter_eye_distance": float(inter_eye),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "detection_confidence": float(best_conf),
                "aligned_size": [int(aw), int(ah)],
            }
        )

        lmk_norm68, norm_scale = self.normalize_landmarks(lmk68[:, :3], face_meta_aligned)
        face_meta_aligned["scale_used"] = float(norm_scale)
        idx50 = np.asarray(self._abs_mod.SELECTED_68_50, dtype=np.int32)
        pts50 = lmk_norm68[idx50]
        flat_lmk = pts50.reshape(-1)
        dists, aux = self._abs_mod.compute_distances(lmk_norm68.astype(np.float64))
        aus = self._abs_mod.compute_au_from_geometry(dists, aux)
        au_names = list(self._abs_mod.AU_NAMES)
        dist_names = list(self._abs_mod.DIST_NAMES)
        au_vec = np.array([float(aus[a]) for a in au_names], dtype=np.float32)
        dist_vec = np.array([float(dists[d]) for d in dist_names], dtype=np.float32)
        pose = np.array([float(yaw), float(pitch), float(roll)], dtype=np.float32)
        abs_vec = np.concatenate([pose, au_vec, flat_lmk.astype(np.float32), dist_vec], axis=0).astype(np.float32)
        if abs_vec.shape[0] != self.abs_dim:
            raise RuntimeError(f"ABS vector size mismatch: got {abs_vec.shape[0]}, expect {self.abs_dim}")
        return {
            "abs": abs_vec,
            "pose": pose,
            "au": au_vec,
            "lmk_norm": flat_lmk.astype(np.float32),
            "dist": dist_vec,
            "face_meta": face_meta_aligned,
        }

    def compute_human_rel(
        self,
        human_neutral_feature: Mapping[str, Any],
        human_expr_feature: Mapping[str, Any],
    ) -> np.ndarray:
        """Compute REL190 from normalized feature components.

        IMPORTANT:
        REL must be computed from normalized expression-neutral deltas,
        not raw pixel coordinates.
        """
        for k in ("au", "lmk_norm", "dist"):
            if k not in human_neutral_feature or k not in human_expr_feature:
                raise ValueError(f"extract_face_feature output must contain '{k}' for REL computation.")
        au_neu = _to_np_1d(human_neutral_feature["au"], "human_neutral.au")
        au_exp = _to_np_1d(human_expr_feature["au"], "human_expr.au")
        lmk_neu = _to_np_1d(human_neutral_feature["lmk_norm"], "human_neutral.lmk_norm")
        lmk_exp = _to_np_1d(human_expr_feature["lmk_norm"], "human_expr.lmk_norm")
        dist_neu = _to_np_1d(human_neutral_feature["dist"], "human_neutral.dist")
        dist_exp = _to_np_1d(human_expr_feature["dist"], "human_expr.dist")
        if au_neu.shape != au_exp.shape or lmk_neu.shape != lmk_exp.shape or dist_neu.shape != dist_exp.shape:
            raise ValueError("neutral/expr normalized feature shapes are inconsistent.")
        au_rel = (au_exp - au_neu).astype(np.float32)
        lmk_rel = (lmk_exp - lmk_neu).astype(np.float32)
        dist_rel = (dist_exp - dist_neu).astype(np.float32)
        energy_rel = float(np.sum(np.abs(dist_rel)))

        # Build rel source-column map first, then pack REL190 by columns.json order when available.
        au_names = list(self._abs_mod.AU_NAMES)
        dist_names = list(self._abs_mod.DIST_NAMES)
        rel_map: Dict[str, float] = {}
        for i, au in enumerate(au_names):
            rel_map[f"{au}_rel"] = float(au_rel[i])

        lmk_rel_50x3 = lmk_rel.reshape(50, 3)
        for i in range(50):
            rel_map[f"lmk_rel_{i:02d}_x"] = float(lmk_rel_50x3[i, 0])
            rel_map[f"lmk_rel_{i:02d}_y"] = float(lmk_rel_50x3[i, 1])
            rel_map[f"lmk_rel_{i:02d}_z"] = float(lmk_rel_50x3[i, 2])

        for i, dname in enumerate(dist_names):
            rel_map[f"{dname}_rel"] = float(dist_rel[i])
        rel_map["ENERGY_rel"] = float(energy_rel)

        self._last_rel_missing_columns = []
        if self._feature_columns:
            rel_values: List[float] = []
            for item in self._feature_columns:
                if str(item.get("source", "")).strip().lower() != "rel":
                    continue
                source_col = str(item.get("source_column", "")).strip()
                if source_col in rel_map:
                    rel_values.append(float(rel_map[source_col]))
                else:
                    self._last_rel_missing_columns.append(source_col)
                    rel_values.append(0.0)
            rel190 = np.asarray(rel_values, dtype=np.float32)
            if rel190.shape[0] != self.rel_dim:
                raise RuntimeError(
                    f"REL190 dim mismatch after columns-json packing: got {rel190.shape[0]}, expect {self.rel_dim}"
                )
            if self._last_rel_missing_columns:
                self._warn(
                    f"REL190 columns missing from rel_map, filled zeros: {len(self._last_rel_missing_columns)} columns"
                )
            self._rel190_order_name = "columns_json_rel_by_feat_idx"
            self._rel190_order_checked_by_columns_json = True
            return rel190

        # Fallback (kept for robustness when columns json missing).
        rel190 = np.concatenate([au_rel, lmk_rel, dist_rel, np.array([energy_rel], dtype=np.float32)], axis=0).astype(np.float32)
        if rel190.shape[0] != 190:
            raise RuntimeError(f"REL190 dim mismatch, got {rel190.shape[0]}")
        self._rel190_order_name = "au_lmk_dist_energy_fallback"
        self._rel190_order_checked_by_columns_json = False
        return rel190

    def normalize_human_rel(
        self,
        human_rel: np.ndarray,
        human_neutral_feature: Mapping[str, Any],
        human_expr_feature: Mapping[str, Any],
    ) -> np.ndarray:
        """Normalize REL by human face scale to reduce distance/size effect.

        REL already comes from normalized landmarks/geometry.
        Default is no-op. Optional scale-ratio correction can be enabled by config.
        """
        rel = _to_np_1d(human_rel, "human_rel")
        neu_meta = human_neutral_feature.get("face_meta", {}) if isinstance(human_neutral_feature, Mapping) else {}
        exp_meta = human_expr_feature.get("face_meta", {}) if isinstance(human_expr_feature, Mapping) else {}
        s_neu = float(neu_meta.get("scale_used", 1.0) or 1.0)
        s_exp = float(exp_meta.get("scale_used", 1.0) or 1.0)
        if s_neu <= 1e-8 or s_exp <= 1e-8:
            self._warn("invalid face scale in normalize_human_rel, fallback no-op.")
            return rel.astype(np.float32)
        if not self._apply_scale_ratio_correction:
            return rel.astype(np.float32)
        ratio = s_exp / s_neu
        corrected = rel / max(ratio, 1e-6)
        return corrected.astype(np.float32)

    def map_human_rel_to_robot_scale(self, human_rel_norm: np.ndarray) -> np.ndarray:
        """Map normalized human REL to robot-style REL."""
        robot_face_scale = 1.0
        scale_factor = float(robot_face_scale)
        robot_style_rel = human_rel_norm.astype(np.float32) * scale_factor

        if self._normalizer_stats:
            vmin = self._normalizer_stats.get("rel_clip_min", None)
            vmax = self._normalizer_stats.get("rel_clip_max", None)
            if vmin is not None and vmax is not None:
                robot_style_rel = np.clip(robot_style_rel, np.asarray(vmin), np.asarray(vmax)).astype(np.float32)
                return robot_style_rel

        if self.clip_rel:
            robot_style_rel = np.clip(robot_style_rel, self.rel_clip_min, self.rel_clip_max).astype(np.float32)
        return robot_style_rel

    def map_rel_to_abs_delta(self, robot_style_rel: np.ndarray) -> np.ndarray:
        """Map REL delta to ABS delta.

        If mapping is unknown, safe fallback is zeros(abs_dim), i.e. keep robot neutral ABS.
        """
        rel = _to_np_1d(robot_style_rel, "robot_style_rel")
        self._last_mapping_info = {
            "rel_to_abs_mapping_used": False,
            "rel_to_abs_mapping_type": "fallback_zero_delta",
            "global_abs_kept_neutral": True,
        }

        # Preferred default schema for compare6 FEATURE385:
        # ABS192 = [brow37, eye45, mouth79, jaw28, global3]
        # REL190 = [brow37, eye45, mouth79, jaw28, global1]
        # Map first 189 REL dims region-wise to ABS first 189 dims;
        # keep ABS global(189:192) unchanged (delta=0).
        if self.abs_dim == 192 and self.rel_dim == 190:
            if rel.shape[0] != 190:
                raise ValueError(f"robot_style_rel dim mismatch: got {rel.shape[0]}, expect 190")
            out = np.zeros((192,), dtype=np.float32)
            out[0:37] = rel[0:37]       # brow
            out[37:82] = rel[37:82]     # eye
            out[82:161] = rel[82:161]   # mouth
            out[161:189] = rel[161:189] # jaw
            # out[189:192] keeps 0.0 (global_abs kept neutral)
            self._last_mapping_info = {
                "rel_to_abs_mapping_used": True,
                "rel_to_abs_mapping_type": "default_regionwise_189_to_192",
                "global_abs_kept_neutral": True,
            }
            return out

        schema = self.config.get("schema", {}) if isinstance(self.config, Mapping) else {}
        if isinstance(schema, Mapping):
            mat = schema.get("rel_to_abs_matrix", None)
            if mat is not None:
                m = np.asarray(mat, dtype=np.float32)
                if m.shape == (self.abs_dim, self.rel_dim):
                    self._last_mapping_info = {
                        "rel_to_abs_mapping_used": True,
                        "rel_to_abs_mapping_type": "schema_matrix",
                        "global_abs_kept_neutral": False,
                    }
                    return (m @ rel).astype(np.float32)

            index_map = schema.get("rel_to_abs_index", None)
            if index_map is not None and isinstance(index_map, list) and len(index_map) == self.abs_dim:
                out = np.zeros((self.abs_dim,), dtype=np.float32)
                for i, ridx in enumerate(index_map):
                    j = int(ridx)
                    if 0 <= j < self.rel_dim:
                        out[i] = rel[j]
                self._last_mapping_info = {
                    "rel_to_abs_mapping_used": True,
                    "rel_to_abs_mapping_type": "schema_index",
                    "global_abs_kept_neutral": False,
                }
                return out

        self._warn("REL to ABS mapping is not configured. Using robot neutral ABS only for ABS channel.")
        return np.zeros((self.abs_dim,), dtype=np.float32)

    def build_robot_style_abs(self, robot_style_rel: np.ndarray) -> np.ndarray:
        """Build robot-style ABS from robot neutral ABS + mapped REL delta.

        Why not use human ABS directly:
        - Human ABS carries subject-specific morphology that differs from robot.
        - We anchor ABS on robot neutral face, then inject adapted expression change.
        """
        abs_delta = self.map_rel_to_abs_delta(robot_style_rel)
        self._last_abs_delta = abs_delta.copy()
        neutral_abs = self.robot_neutral["abs"]
        if abs_delta.shape != neutral_abs.shape:
            raise ValueError(
                f"abs_delta shape mismatch: delta={abs_delta.shape}, neutral_abs={neutral_abs.shape}"
            )
        robot_style_abs = neutral_abs + abs_delta
        if robot_style_abs.shape[0] != self.abs_dim:
            raise ValueError(
                f"robot_style_abs dim mismatch: got {robot_style_abs.shape[0]}, expect {self.abs_dim}"
            )
        return robot_style_abs.astype(np.float32)

    def build_feature385(self, robot_style_abs: np.ndarray, robot_style_rel: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Build FEATURE385_adapted in compare6 interleaved layout."""
        abs_vec = _to_np_1d(robot_style_abs, "robot_style_abs")
        rel_vec = _to_np_1d(robot_style_rel, "robot_style_rel")
        pose_vec = _to_np_1d(pose, "pose")
        if abs_vec.shape[0] != self.abs_dim or rel_vec.shape[0] != self.rel_dim or pose_vec.shape[0] != self.pose_dim:
            raise ValueError(
                f"build_feature385 dim mismatch: abs={abs_vec.shape[0]} rel={rel_vec.shape[0]} pose={pose_vec.shape[0]}, "
                f"expect abs={self.abs_dim} rel={self.rel_dim} pose={self.pose_dim}"
            )
        if self.abs_dim == 192 and self.rel_dim == 190 and self.pose_dim == 3:
            feat = pack_feature385_interleaved(abs_vec, rel_vec, pose_vec)
        else:
            feat = np.concatenate([abs_vec, rel_vec, pose_vec], axis=0).astype(np.float32)
        if feat.shape[0] != self.feature_dim_total:
            raise ValueError(
                f"feature385 dim mismatch: got {feat.shape[0]}, expect {self.feature_dim_total}; "
                f"abs_dim={self.abs_dim}, rel_dim={self.rel_dim}, pose_dim={self.pose_dim}"
            )
        return feat

    def validate_feature_layout(
        self,
        feature385: np.ndarray,
        robot_style_abs: np.ndarray,
        robot_style_rel: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        f = _to_np_1d(feature385, "feature385")
        a = _to_np_1d(robot_style_abs, "robot_style_abs")
        r = _to_np_1d(robot_style_rel, "robot_style_rel")
        p = _to_np_1d(pose, "pose")
        if f.shape[0] != 385:
            raise ValueError(f"feature385 dim mismatch: {f.shape}")
        if a.shape[0] != 192:
            raise ValueError(f"robot_style_abs dim mismatch: {a.shape}")
        if r.shape[0] != 190:
            raise ValueError(f"robot_style_rel dim mismatch: {r.shape}")
        if p.shape[0] != 3:
            raise ValueError(f"pose dim mismatch: {p.shape}")

        checks = [
            np.allclose(f[0:37], a[0:37]),
            np.allclose(f[37:74], r[0:37]),
            np.allclose(f[74:119], a[37:82]),
            np.allclose(f[119:164], r[37:82]),
            np.allclose(f[164:243], a[82:161]),
            np.allclose(f[243:322], r[82:161]),
            np.allclose(f[322:350], a[161:189]),
            np.allclose(f[350:378], r[161:189]),
            np.allclose(f[378:381], a[189:192]),
            np.allclose(f[381:382], r[189:190]),
            np.allclose(f[382:385], p),
        ]
        if not all(checks):
            raise AssertionError("feature385 interleaved layout check failed")

    def adapt(self, human_neutral_image_path: str | Path, human_expr_image_path: str | Path) -> Dict[str, Any]:
        """Run full adaptation pipeline."""
        self._warnings = []
        h_neu = self.extract_face_feature(human_neutral_image_path)
        h_exp = self.extract_face_feature(human_expr_image_path)
        neu_meta = h_neu.get("face_meta", {}) if isinstance(h_neu, Mapping) else {}
        exp_meta = h_exp.get("face_meta", {}) if isinstance(h_exp, Mapping) else {}
        s_neu = float(neu_meta.get("scale_used", 1.0) or 1.0)
        s_exp = float(exp_meta.get("scale_used", 1.0) or 1.0)
        scale_ratio = s_exp / max(s_neu, 1e-6)
        large_scale_change = abs(scale_ratio - 1.0) > self._scale_warn_ratio
        if large_scale_change:
            self._warn(
                "Large face scale change detected between neutral and expression image. "
                "Please keep camera distance stable or rely on normalized landmarks."
            )

        human_rel = self.compute_human_rel(h_neu, h_exp)
        human_rel_norm = self.normalize_human_rel(human_rel, h_neu, h_exp)

        # If extracted REL dim differs from model REL dim, use safe adjustment.
        if human_rel_norm.shape[0] != self.rel_dim:
            self._warn(
                f"human_rel dim ({human_rel_norm.shape[0]}) != rel_dim ({self.rel_dim}); using safe resize fallback."
            )
            rel_tmp = np.zeros((self.rel_dim,), dtype=np.float32)
            n = min(self.rel_dim, human_rel_norm.shape[0])
            rel_tmp[:n] = human_rel_norm[:n]
            human_rel_norm = rel_tmp

        robot_style_rel = self.map_human_rel_to_robot_scale(human_rel_norm)
        robot_style_abs = self.build_robot_style_abs(robot_style_rel)
        abs_delta = self._last_abs_delta if self._last_abs_delta is not None else np.zeros((self.abs_dim,), dtype=np.float32)

        pose = _to_np_1d(h_exp.get("pose", self.robot_neutral["pose"]), "human_expr.pose")
        if pose.shape[0] != self.pose_dim:
            self._warn(f"pose dim mismatch ({pose.shape[0]}), fallback to robot neutral pose")
            pose = self.robot_neutral["pose"].copy()

        feature385 = self.build_feature385(robot_style_abs, robot_style_rel, pose)
        feature_layout_check_passed = True
        try:
            self.validate_feature_layout(feature385, robot_style_abs, robot_style_rel, pose)
        except Exception:
            feature_layout_check_passed = False
            raise

        if float(np.max(np.abs(robot_style_abs))) > 10.0:
            self._warn("robot_style_abs has large magnitude (>10).")
        if float(np.max(np.abs(robot_style_rel))) > 10.0:
            self._warn("robot_style_rel has large magnitude (>10).")

        debug = {
            "human_rel_mean_abs": float(np.mean(np.abs(human_rel))),
            "robot_style_rel_mean_abs": float(np.mean(np.abs(robot_style_rel))),
            "robot_style_abs_min": float(np.min(robot_style_abs)),
            "robot_style_abs_max": float(np.max(robot_style_abs)),
            "feature_layout": FEATURE_LAYOUT_NAME,
            "feature385_pack_order": FEATURE385_PACK_ORDER,
            "rel190_order": self._rel190_order_name,
            "rel190_order_checked_by_columns_json": bool(self._rel190_order_checked_by_columns_json),
            "rel190_missing_columns": list(self._last_rel_missing_columns),
            "feature_columns_json_path": None if self._columns_json_path is None else str(self._columns_json_path),
            "feature_layout_check_passed": bool(feature_layout_check_passed),
            "rel_to_abs_mapping_used": bool(self._last_mapping_info.get("rel_to_abs_mapping_used", False)),
            "rel_to_abs_mapping_type": str(self._last_mapping_info.get("rel_to_abs_mapping_type", "unknown")),
            "global_abs_kept_neutral": bool(self._last_mapping_info.get("global_abs_kept_neutral", True)),
            "abs_delta_mean_abs": float(np.mean(np.abs(abs_delta))),
            "abs_delta_min": float(np.min(abs_delta)),
            "abs_delta_max": float(np.max(abs_delta)),
            "robot_neutral_abs_mean_abs": float(np.mean(np.abs(self.robot_neutral["abs"]))),
            "robot_style_abs_mean_abs": float(np.mean(np.abs(robot_style_abs))),
            "neutral_face_scale": float(s_neu),
            "expr_face_scale": float(s_exp),
            "neutral_bbox": neu_meta.get("bbox", None),
            "expr_bbox": exp_meta.get("bbox", None),
            "neutral_image_size": neu_meta.get("image_size", None),
            "expr_image_size": exp_meta.get("image_size", None),
            "aligned_size": exp_meta.get("aligned_size", [self._aligned_size, self._aligned_size]),
            "scale_ratio_expr_to_neutral": float(scale_ratio),
            "warning_if_scale_change_too_large": bool(large_scale_change),
            "feature385_shape": list(feature385.shape),
            "warnings": list(self._warnings),
        }
        return {
            "feature385": feature385,
            "robot_style_abs": robot_style_abs,
            "robot_style_rel": robot_style_rel,
            "pose": pose,
            "debug": debug,
        }
