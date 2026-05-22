#!/usr/bin/env python3
"""Regional regressors for compare6 / compare7.

compare6:
  RegionalMotorRegressor (original B1vnext)

compare7-A:
  RegionalMotorRegressorContextNose (B1vnextContextNose)
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Two-layer region encoder: in -> hidden -> out."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseEncoder(nn.Module):
    """Pose encoder for yaw/pitch/roll."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateFusionWithPose(nn.Module):
    """Gate fusion with pose context.

    alpha = sigmoid(W * [h_abs, h_rel, h_pose])
    h = alpha * h_abs + (1 - alpha) * h_rel
    """

    def __init__(self, feat_dim: int, pose_dim: int):
        super().__init__()
        self.gate = nn.Linear(feat_dim * 2 + pose_dim, feat_dim)

    def forward(self, h_abs: torch.Tensor, h_rel: torch.Tensor, h_pose: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(torch.cat([h_abs, h_rel, h_pose], dim=1)))
        return alpha * h_abs + (1.0 - alpha) * h_rel


class PreLNResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block.

    x -> LN -> Linear(up) -> GELU -> Dropout -> Linear(down) -> add(x)
    """

    def __init__(self, hidden_dim: int = 96, expand_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, expand_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(expand_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class RegionalMotorRegressor(nn.Module):
    """compare6 B1vnext model (FEATURE385 -> z24 -> motor30)."""

    def __init__(self):
        super().__init__()
        self.model_name = "B1vnext"
        self.latent_dim = 24
        self.nose_residual_enabled = False
        self.brow_residual_input_dim = 102
        self.mouth_residual_input_dim = 108
        self.nose_residual_input_dim = 0

        self.register_buffer("brow_out_idx", torch.tensor([0, 1, 2, 3], dtype=torch.long), persistent=False)
        self.register_buffer(
            "mouth_out_idx",
            torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("nose_out_idx", torch.tensor([29], dtype=torch.long), persistent=False)

        pose_dim = 8
        self.pose_encoder = PoseEncoder(input_dim=3, hidden_dim=16, output_dim=pose_dim)

        self.brow_abs_encoder = MLPEncoder(37, 32, 16)
        self.brow_rel_encoder = MLPEncoder(37, 32, 16)
        self.eye_abs_encoder = MLPEncoder(45, 32, 16)
        self.eye_rel_encoder = MLPEncoder(45, 32, 16)
        self.mouth_abs_encoder = MLPEncoder(79, 64, 32)
        self.mouth_rel_encoder = MLPEncoder(79, 64, 32)
        self.jaw_abs_encoder = MLPEncoder(28, 32, 16)
        self.jaw_rel_encoder = MLPEncoder(28, 32, 16)
        self.global_abs_encoder = MLPEncoder(3, 8, 8)
        self.global_rel_encoder = MLPEncoder(1, 8, 8)

        self.brow_fusion = GateFusionWithPose(feat_dim=16, pose_dim=pose_dim)
        self.eye_fusion = GateFusionWithPose(feat_dim=16, pose_dim=pose_dim)
        self.mouth_fusion = GateFusionWithPose(feat_dim=32, pose_dim=pose_dim)
        self.jaw_fusion = GateFusionWithPose(feat_dim=16, pose_dim=pose_dim)
        self.global_fusion = GateFusionWithPose(feat_dim=8, pose_dim=pose_dim)

        self.brow_projector = nn.Linear(16, 4)
        self.eye_projector = nn.Linear(16, 4)
        self.mouth_projector = nn.Linear(32, 10)
        self.jaw_projector = nn.Linear(16, 4)
        self.global_projector = nn.Linear(8, 2)

        # Input projection: 24 -> 96
        self.input_proj = nn.Linear(24, 96)
        # Trunk: 2 pre-LN residual MLP blocks
        self.trunk_blocks = nn.ModuleList(
            [
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
            ]
        )

        # Base head: LN -> 96 -> GELU -> 30
        self.base_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.Linear(96, 30),
        )

        # Brow residual: [h_shared(96), z_brow(4), z_global(2)] -> 102 -> 4
        self.brow_residual_head = nn.Sequential(
            nn.Linear(102, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 4),
        )

        # Mouth residual: [h_shared(96), z_mouth(10), z_global(2)] -> 108 -> 13
        self.mouth_residual_head = nn.Sequential(
            nn.Linear(108, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 13),
        )

        # Learnable residual scales.
        self.alpha_b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha_m = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def encode(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose = x["pose"]
        h_pose = self.pose_encoder(pose)

        h_brow_abs = self.brow_abs_encoder(x["brow_abs"])
        h_brow_rel = self.brow_rel_encoder(x["brow_rel"])
        h_eye_abs = self.eye_abs_encoder(x["eye_abs"])
        h_eye_rel = self.eye_rel_encoder(x["eye_rel"])
        h_mouth_abs = self.mouth_abs_encoder(x["mouth_abs"])
        h_mouth_rel = self.mouth_rel_encoder(x["mouth_rel"])
        h_jaw_abs = self.jaw_abs_encoder(x["jaw_abs"])
        h_jaw_rel = self.jaw_rel_encoder(x["jaw_rel"])
        h_global_abs = self.global_abs_encoder(x["global_abs"])
        h_global_rel = self.global_rel_encoder(x["global_rel"])

        h_brow = self.brow_fusion(h_brow_abs, h_brow_rel, h_pose)
        h_eye = self.eye_fusion(h_eye_abs, h_eye_rel, h_pose)
        h_mouth = self.mouth_fusion(h_mouth_abs, h_mouth_rel, h_pose)
        h_jaw = self.jaw_fusion(h_jaw_abs, h_jaw_rel, h_pose)
        h_global = self.global_fusion(h_global_abs, h_global_rel, h_pose)

        z_brow = self.brow_projector(h_brow)
        z_eye = self.eye_projector(h_eye)
        z_mouth = self.mouth_projector(h_mouth)
        z_jaw = self.jaw_projector(h_jaw)
        z_global = self.global_projector(h_global)
        latent24 = torch.cat([z_brow, z_eye, z_mouth, z_jaw, z_global], dim=1)

        return {
            "z_brow": z_brow,
            "z_eye": z_eye,
            "z_mouth": z_mouth,
            "z_jaw": z_jaw,
            "z_global": z_global,
            "latent24": latent24,
        }

    def _forward_heads_from_latent(self, latent24: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_brow = latent24[:, 0:4]
        z_mouth = latent24[:, 8:18]
        z_global = latent24[:, 22:24]

        h_shared = self.input_proj(latent24)
        for block in self.trunk_blocks:
            h_shared = block(h_shared)

        y_base = self.base_head(h_shared)

        brow_in = torch.cat([h_shared, z_brow, z_global], dim=1)
        mouth_in = torch.cat([h_shared, z_mouth, z_global], dim=1)
        delta_brow = self.brow_residual_head(brow_in)
        delta_mouth = self.mouth_residual_head(mouth_in)

        y = y_base.clone()
        y[:, self.brow_out_idx] = y[:, self.brow_out_idx] + self.alpha_b * delta_brow
        y[:, self.mouth_out_idx] = y[:, self.mouth_out_idx] + self.alpha_m * delta_mouth

        return {
            "h_shared": h_shared,
            "y_base": y_base,
            "delta_brow": delta_brow,
            "delta_mouth": delta_mouth,
            "alpha_b": self.alpha_b,
            "alpha_m": self.alpha_m,
            "y": y,
        }

    def forward_from_latent(self, latent24: torch.Tensor) -> torch.Tensor:
        return self._forward_heads_from_latent(latent24)["y"]

    def forward(self, x: Mapping[str, torch.Tensor], return_latent: bool = False, return_parts: bool = False):
        encoded = self.encode(x)
        parts = self._forward_heads_from_latent(encoded["latent24"])
        pred = parts["y"]

        if return_parts:
            if return_latent:
                return pred, encoded["latent24"], parts
            return pred, parts
        if return_latent:
            return pred, encoded["latent24"]
        return pred


class RegionalMotorRegressorContextNose(RegionalMotorRegressor):
    """compare7-A / B1vnext-ContextNose.

    Keep FEATURE385 input, z24 latent, and trunk unchanged.
    Only enhance residual head contexts and add a nose-specific residual head.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "B1vnextContextNose"
        self.nose_residual_enabled = True
        self.brow_residual_input_dim = 106
        self.mouth_residual_input_dim = 112
        self.nose_residual_input_dim = 116

        # context brow residual: [h(96), z_brow(4), z_eye(4), z_global(2)] -> 4
        self.brow_residual_head = nn.Sequential(
            nn.Linear(106, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 4),
        )

        # context mouth residual: [h(96), z_mouth(10), z_jaw(4), z_global(2)] -> 13
        self.mouth_residual_head = nn.Sequential(
            nn.Linear(112, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 13),
        )

        # nose residual: [h(96), z_brow(4), z_eye(4), z_mouth(10), z_global(2)] -> 1
        self.nose_residual_head = nn.Sequential(
            nn.Linear(116, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1),
        )

        self.alpha_n = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def _forward_heads_from_latent(self, latent24: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_brow = latent24[:, 0:4]
        z_eye = latent24[:, 4:8]
        z_mouth = latent24[:, 8:18]
        z_jaw = latent24[:, 18:22]
        z_global = latent24[:, 22:24]

        h_shared = self.input_proj(latent24)
        for block in self.trunk_blocks:
            h_shared = block(h_shared)

        y_base = self.base_head(h_shared)

        brow_in = torch.cat([h_shared, z_brow, z_eye, z_global], dim=1)
        mouth_in = torch.cat([h_shared, z_mouth, z_jaw, z_global], dim=1)
        nose_in = torch.cat([h_shared, z_brow, z_eye, z_mouth, z_global], dim=1)

        delta_brow = self.brow_residual_head(brow_in)
        delta_mouth = self.mouth_residual_head(mouth_in)
        delta_nose = self.nose_residual_head(nose_in)  # [B, 1]

        y = y_base.clone()
        y[:, self.brow_out_idx] = y[:, self.brow_out_idx] + self.alpha_b * delta_brow
        y[:, self.mouth_out_idx] = y[:, self.mouth_out_idx] + self.alpha_m * delta_mouth
        # motor29 receives both mouth residual and nose residual in this compare7-A design.
        y[:, 29] = y[:, 29] + self.alpha_n * delta_nose.squeeze(-1)

        return {
            "h_shared": h_shared,
            "y_base": y_base,
            "delta_brow": delta_brow,
            "delta_mouth": delta_mouth,
            "delta_nose": delta_nose,
            "alpha_b": self.alpha_b,
            "alpha_m": self.alpha_m,
            "alpha_n": self.alpha_n,
            "y": y,
        }


B1vnextContextNose = RegionalMotorRegressorContextNose


def build_model(model_name: str | None = None) -> nn.Module:
    name = (model_name or "").strip().lower()
    if name in {"b1vnextcontextnose", "regionalmotorregressorcontextnose", "compare7-a", "compare7_context_nose"}:
        return RegionalMotorRegressorContextNose()
    return RegionalMotorRegressor()


def build_model_from_config(cfg: Mapping[str, object]) -> nn.Module:
    return build_model(model_name=str(cfg.get("model_name", "")))


# Compatibility alias for old imports/scripts.
MotorRegressorMLP = RegionalMotorRegressor
