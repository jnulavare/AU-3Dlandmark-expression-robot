#!/usr/bin/env python3
"""Regional gated regressor (compare5 output head):
[ABS, REL, Pose] -> z24 (regional latent) -> shared/base + residual heads -> y30.
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


class RegionalMotorRegressor(nn.Module):
    """
    [ABS, REL, Pose] -> regional encoders -> gated fusion -> z24
    z24 -> h_shared -> y_base + local residuals (brow + jaw/mouth joint) -> y30
    """

    def __init__(self):
        super().__init__()
        # Motor output indices used by local residual heads.
        self.register_buffer("brow_out_idx", torch.tensor([0, 1, 2, 3], dtype=torch.long), persistent=False)
        self.register_buffer(
            # jaw region (7): 10,11,12,13,14,27,28
            # mouth region (13): 15..26,29
            "jawmouth_out_idx",
            torch.tensor([10, 11, 12, 13, 14, 27, 28, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29], dtype=torch.long),
            persistent=False,
        )

        # A shared pose embedding is injected into all region fusion gates.
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

        # Shared trunk: z24 -> h_shared (R^64)
        self.shared_trunk = nn.Sequential(
            nn.Linear(24, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        # Base head predicts all 30 motors from global hidden representation.
        self.base_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 30),
        )

        # Brow residual: [h_shared(64), z_brow(4), z_global(2)] -> delta(4)
        self.brow_residual_head = nn.Sequential(
            nn.Linear(70, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 4),
        )

        # Jaw+mouth joint residual:
        # [h_shared(64), z_jaw(4), z_mouth(10), z_global(2)] -> delta(20)
        self.jawmouth_residual_head = nn.Sequential(
            nn.Linear(80, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 20),
        )

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
        # Split latent24 into region chunks.
        z_brow = latent24[:, 0:4]
        z_jaw = latent24[:, 18:22]
        z_mouth = latent24[:, 8:18]
        z_global = latent24[:, 22:24]

        h_shared = self.shared_trunk(latent24)
        y_base = self.base_head(h_shared)

        brow_in = torch.cat([h_shared, z_brow, z_global], dim=1)
        jawmouth_in = torch.cat([h_shared, z_jaw, z_mouth, z_global], dim=1)
        delta_brow = self.brow_residual_head(brow_in)
        delta_jawmouth = self.jawmouth_residual_head(jawmouth_in)

        y = y_base.clone()
        y[:, self.brow_out_idx] = y[:, self.brow_out_idx] + delta_brow
        y[:, self.jawmouth_out_idx] = y[:, self.jawmouth_out_idx] + delta_jawmouth

        return {
            "h_shared": h_shared,
            "y_base": y_base,
            "delta_brow": delta_brow,
            "delta_jawmouth": delta_jawmouth,
            "y": y,
        }

    def forward_from_latent(self, latent24: torch.Tensor) -> torch.Tensor:
        return self._forward_heads_from_latent(latent24)["y"]

    def forward(self, x: Mapping[str, torch.Tensor], return_latent: bool = False):
        encoded = self.encode(x)
        pred = self._forward_heads_from_latent(encoded["latent24"])["y"]
        if return_latent:
            return pred, encoded["latent24"]
        return pred


# Keep this alias so old imports continue to work.
MotorRegressorMLP = RegionalMotorRegressor
