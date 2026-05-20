#!/usr/bin/env python3
"""B1vnext model variants for REL190 and REL193-Context."""

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


class PreLNResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

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


class RegionalMotorRegressorREL190(nn.Module):
    """REL190 B1vnext model (z24)."""

    def __init__(self):
        super().__init__()
        self.model_name = "B1vnextREL190"
        self.latent_dim = 24
        self.brow_residual_input_dim = 102
        self.mouth_residual_input_dim = 108

        self.register_buffer("brow_out_idx", torch.tensor([0, 1, 2, 3], dtype=torch.long), persistent=False)
        self.register_buffer(
            "mouth_out_idx",
            torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29], dtype=torch.long),
            persistent=False,
        )

        self.brow_rel_encoder = MLPEncoder(37, 32, 16)
        self.eye_rel_encoder = MLPEncoder(45, 32, 16)
        self.mouth_rel_encoder = MLPEncoder(79, 64, 32)
        self.jaw_rel_encoder = MLPEncoder(28, 32, 16)
        self.global_rel_encoder = MLPEncoder(1, 8, 8)

        self.brow_projector = nn.Linear(16, 4)
        self.eye_projector = nn.Linear(16, 4)
        self.mouth_projector = nn.Linear(32, 10)
        self.jaw_projector = nn.Linear(16, 4)
        self.global_projector = nn.Linear(8, 2)

        self.input_proj = nn.Linear(24, 96)
        self.trunk_blocks = nn.ModuleList(
            [
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
            ]
        )

        self.base_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.Linear(96, 30),
        )

        self.brow_residual_head = nn.Sequential(
            nn.Linear(102, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 4),
        )
        self.mouth_residual_head = nn.Sequential(
            nn.Linear(108, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 13),
        )

        self.alpha_b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha_m = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def encode(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_brow = self.brow_rel_encoder(x["brow_rel"])
        h_eye = self.eye_rel_encoder(x["eye_rel"])
        h_mouth = self.mouth_rel_encoder(x["mouth_rel"])
        h_jaw = self.jaw_rel_encoder(x["jaw_rel"])
        h_global = self.global_rel_encoder(x["global_rel"])

        z_brow = self.brow_projector(h_brow)
        z_eye = self.eye_projector(h_eye)
        z_mouth = self.mouth_projector(h_mouth)
        z_jaw = self.jaw_projector(h_jaw)
        z_global = self.global_projector(h_global)
        latent = torch.cat([z_brow, z_eye, z_mouth, z_jaw, z_global], dim=1)

        return {
            "z_brow": z_brow,
            "z_eye": z_eye,
            "z_mouth": z_mouth,
            "z_jaw": z_jaw,
            "z_global": z_global,
            "latent": latent,
            "latent24": latent,
        }

    def _forward_heads_from_latent(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_brow = latent[:, 0:4]
        z_mouth = latent[:, 8:18]
        z_global = latent[:, 22:24]

        h_shared = self.input_proj(latent)
        for block in self.trunk_blocks:
            h_shared = block(h_shared)

        y_base = self.base_head(h_shared)
        delta_brow = self.brow_residual_head(torch.cat([h_shared, z_brow, z_global], dim=1))
        delta_mouth = self.mouth_residual_head(torch.cat([h_shared, z_mouth, z_global], dim=1))

        y = y_base.clone()
        y[:, self.brow_out_idx] = y[:, self.brow_out_idx] + self.alpha_b * delta_brow
        y[:, self.mouth_out_idx] = y[:, self.mouth_out_idx] + self.alpha_m * delta_mouth
        return {"y": y}

    def forward_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self._forward_heads_from_latent(latent)["y"]

    def forward(self, x: Mapping[str, torch.Tensor], return_latent: bool = False):
        encoded = self.encode(x)
        y = self._forward_heads_from_latent(encoded["latent"])["y"]
        if return_latent:
            return y, encoded["latent"]
        return y


class RegionalMotorRegressorREL193Context(nn.Module):
    """REL193 + context-aware residual B1vnext model (z26)."""

    def __init__(self):
        super().__init__()
        self.model_name = "B1vnextREL193Context"
        self.latent_dim = 26
        self.brow_residual_input_dim = 108
        self.mouth_residual_input_dim = 114

        self.register_buffer("brow_out_idx", torch.tensor([0, 1, 2, 3], dtype=torch.long), persistent=False)
        self.register_buffer(
            "mouth_out_idx",
            torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29], dtype=torch.long),
            persistent=False,
        )

        self.brow_rel_encoder = MLPEncoder(37, 32, 16)
        self.eye_rel_encoder = MLPEncoder(45, 32, 16)
        self.mouth_rel_encoder = MLPEncoder(79, 64, 32)
        self.jaw_rel_encoder = MLPEncoder(28, 32, 16)
        self.global_rel_encoder = MLPEncoder(1, 8, 8)
        self.pose_encoder = MLPEncoder(3, 16, 8)

        self.brow_projector = nn.Linear(16, 4)
        self.eye_projector = nn.Linear(16, 4)
        self.mouth_projector = nn.Linear(32, 10)
        self.jaw_projector = nn.Linear(16, 4)
        self.global_projector = nn.Linear(8, 2)
        self.pose_projector = nn.Linear(8, 2)

        self.input_proj = nn.Linear(26, 96)
        self.trunk_blocks = nn.ModuleList(
            [
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
                PreLNResidualMLPBlock(hidden_dim=96, expand_dim=192, dropout=0.1),
            ]
        )

        self.base_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.Linear(96, 30),
        )

        self.brow_residual_head = nn.Sequential(
            nn.Linear(108, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 4),
        )
        self.mouth_residual_head = nn.Sequential(
            nn.Linear(114, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 13),
        )

        self.alpha_b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha_m = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def encode(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_brow = self.brow_rel_encoder(x["brow_rel"])
        h_eye = self.eye_rel_encoder(x["eye_rel"])
        h_mouth = self.mouth_rel_encoder(x["mouth_rel"])
        h_jaw = self.jaw_rel_encoder(x["jaw_rel"])
        h_global = self.global_rel_encoder(x["global_rel"])
        h_pose = self.pose_encoder(x["pose"])

        z_brow = self.brow_projector(h_brow)
        z_eye = self.eye_projector(h_eye)
        z_mouth = self.mouth_projector(h_mouth)
        z_jaw = self.jaw_projector(h_jaw)
        z_global = self.global_projector(h_global)
        z_pose = self.pose_projector(h_pose)
        latent = torch.cat([z_brow, z_eye, z_mouth, z_jaw, z_global, z_pose], dim=1)

        return {
            "z_brow": z_brow,
            "z_eye": z_eye,
            "z_mouth": z_mouth,
            "z_jaw": z_jaw,
            "z_global": z_global,
            "z_pose": z_pose,
            "latent": latent,
            "latent26": latent,
        }

    def _forward_heads_from_latent(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_brow = latent[:, 0:4]
        z_eye = latent[:, 4:8]
        z_mouth = latent[:, 8:18]
        z_jaw = latent[:, 18:22]
        z_global = latent[:, 22:24]
        z_pose = latent[:, 24:26]

        h_shared = self.input_proj(latent)
        for block in self.trunk_blocks:
            h_shared = block(h_shared)

        y_base = self.base_head(h_shared)
        delta_brow = self.brow_residual_head(torch.cat([h_shared, z_brow, z_eye, z_global, z_pose], dim=1))
        delta_mouth = self.mouth_residual_head(torch.cat([h_shared, z_mouth, z_jaw, z_global, z_pose], dim=1))

        y = y_base.clone()
        y[:, self.brow_out_idx] = y[:, self.brow_out_idx] + self.alpha_b * delta_brow
        y[:, self.mouth_out_idx] = y[:, self.mouth_out_idx] + self.alpha_m * delta_mouth
        return {"y": y}

    def forward_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self._forward_heads_from_latent(latent)["y"]

    def forward(self, x: Mapping[str, torch.Tensor], return_latent: bool = False):
        encoded = self.encode(x)
        y = self._forward_heads_from_latent(encoded["latent"])["y"]
        if return_latent:
            return y, encoded["latent"]
        return y


def build_model(model_name: str | None = None, feature_mode: str | None = None) -> nn.Module:
    name = (model_name or "").strip().lower()
    mode = (feature_mode or "").strip().upper()
    if name in {"b1vnextrel193context", "rel193context"} or mode == "REL193":
        return RegionalMotorRegressorREL193Context()
    return RegionalMotorRegressorREL190()


def build_model_from_config(cfg: Mapping[str, object]) -> nn.Module:
    return build_model(model_name=str(cfg.get("model_name", "")), feature_mode=str(cfg.get("feature_mode", "REL190")))


# Backward compatibility default alias (REL190)
MotorRegressorMLP = RegionalMotorRegressorREL190
