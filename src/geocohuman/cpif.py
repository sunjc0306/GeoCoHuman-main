from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SkipPointMLP(nn.Module):
    """Point-wise MLP with input skips at the specified hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        skip_layers: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.skip_layers = set(map(int, skip_layers))
        layers = []
        previous = self.input_dim
        for layer_index, hidden in enumerate(hidden_dims):
            if layer_index in self.skip_layers:
                previous += self.input_dim
            layers.append(nn.Linear(previous, hidden))
            previous = hidden
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(previous, output_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original = features
        hidden = features
        for layer_index, layer in enumerate(self.hidden_layers):
            if layer_index in self.skip_layers:
                hidden = torch.cat((hidden, original), dim=-1)
            hidden = F.silu(layer(hidden))
        return self.output_layer(hidden), hidden


class CollaborativeImplicitFunction(nn.Module):
    """CPIF with two domain-specific prior heads and one collaborative latent head."""

    def __init__(
        self,
        visual_dim: int,
        body_dim: int,
        clothes_dim: int,
        prior_hidden_dims: Sequence[int] = (512, 256, 128, 128),
        latent_hidden_dims: Sequence[int] = (512, 256, 128, 128),
        skip_layers: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        body_input = visual_dim + body_dim + 3
        clothes_input = visual_dim + clothes_dim + 3
        self.body_head = SkipPointMLP(body_input, prior_hidden_dims, 1, skip_layers)
        self.clothes_head = SkipPointMLP(clothes_input, prior_hidden_dims, 1, skip_layers)
        body_latent = int(prior_hidden_dims[-1])
        clothes_latent = int(prior_hidden_dims[-1])
        latent_input = visual_dim + body_latent + clothes_latent + 3
        self.latent_head = SkipPointMLP(latent_input, latent_hidden_dims, 1, skip_layers)

    def forward(
        self,
        visual_features: torch.Tensor,
        body_features: torch.Tensor,
        clothes_features: torch.Tensor,
        query_points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        body_logit, body_latent = self.body_head(
            torch.cat((visual_features, body_features, query_points), dim=-1)
        )
        clothes_logit, clothes_latent = self.clothes_head(
            torch.cat((visual_features, clothes_features, query_points), dim=-1)
        )
        sdf, fused_latent = self.latent_head(
            torch.cat(
                (visual_features, body_latent, clothes_latent, query_points), dim=-1
            )
        )
        return {
            "body_occupancy_logit": body_logit,
            "clothes_occupancy_logit": clothes_logit,
            "body_occupancy": body_logit.sigmoid(),
            "clothes_occupancy": clothes_logit.sigmoid(),
            "body_latent": body_latent,
            "clothes_latent": clothes_latent,
            "fused_latent": fused_latent,
            "sdf": sdf,
        }


class CPIFLoss(nn.Module):
    """Paper losses: two prior BCE terms plus latent SDF L1 and normal MSE."""

    def __init__(
        self,
        occupancy_weight: float = 1.0,
        sdf_weight: float = 1.0,
        normal_weight: float = 1.0,
        positive_balance: float = 0.5,
    ) -> None:
        super().__init__()
        self.occupancy_weight = float(occupancy_weight)
        self.sdf_weight = float(sdf_weight)
        self.normal_weight = float(normal_weight)
        self.positive_balance = float(positive_balance)

    def _balanced_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            target >= 0.5,
            torch.as_tensor(self.positive_balance, device=target.device, dtype=target.dtype),
            torch.as_tensor(1.0 - self.positive_balance, device=target.device, dtype=target.dtype),
        )
        values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        occupancy: torch.Tensor,
        sdf: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        normal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        body_bce = self._balanced_bce(prediction["body_occupancy_logit"], occupancy)
        clothes_bce = self._balanced_bce(prediction["clothes_occupancy_logit"], occupancy)
        occupancy_loss = body_bce + clothes_bce
        sdf_loss = F.l1_loss(prediction["sdf"], sdf)
        normal_loss = prediction["sdf"].sum() * 0.0
        if normals is not None and "normal" in prediction:
            error = (prediction["normal"] - normals).square().sum(dim=-1, keepdim=True)
            if normal_mask is None:
                normal_loss = error.mean()
            else:
                normal_loss = (error * normal_mask).sum() / normal_mask.sum().clamp_min(1.0)
        total = (
            self.occupancy_weight * occupancy_loss
            + self.sdf_weight * sdf_loss
            + self.normal_weight * normal_loss
        )
        return total, {
            "loss": float(total.detach()),
            "body_bce": float(body_bce.detach()),
            "clothes_bce": float(clothes_bce.detach()),
            "sdf_l1": float(sdf_loss.detach()),
            "normal_mse": float(normal_loss.detach()),
        }

