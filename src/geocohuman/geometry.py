from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def safe_normalize(vectors: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return vectors / vectors.norm(dim=dim, keepdim=True).clamp_min(eps)


def transform_points(points: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Apply batched 4x4 transforms to points shaped [B,N,3]."""
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    transformed = torch.bmm(homogeneous, matrices.transpose(1, 2))
    return transformed[..., :3]


def project_points(
    points: torch.Tensor, calibration: torch.Tensor, perspective: bool = False
) -> torch.Tensor:
    """Project world points and retain camera-space z; xy is expected in [-1,1]."""
    camera = transform_points(points, calibration)
    if perspective:
        xy = camera[..., :2] / camera[..., 2:3].clamp_min(1e-8)
        camera = torch.cat((xy, camera[..., 2:3]), dim=-1)
    return camera


def sample_image_features(feature_map: torch.Tensor, projected_points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B,C,H,W] at normalized [B,N,3] projected points."""
    grid = projected_points[..., :2].unsqueeze(2)
    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled.squeeze(-1).transpose(1, 2)


def default_inverse_calibrations(
    batch: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Fallback front/back orthographic cameras for normalized canonical coordinates."""
    front = torch.eye(4, device=device, dtype=dtype)
    back = torch.eye(4, device=device, dtype=dtype)
    back[0, 0] = -1.0
    back[2, 2] = -1.0
    return torch.stack((front, back), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)


def _depth_grid_to_world(
    depth: torch.Tensor, inverse_calibration: torch.Tensor
) -> torch.Tensor:
    batch, views, height, width = depth.shape
    xs = torch.linspace(-1.0, 1.0, width, device=depth.device, dtype=depth.dtype)
    ys = torch.linspace(1.0, -1.0, height, device=depth.device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).view(1, 1, height, width, 2)
    xy = xy.expand(batch, views, -1, -1, -1)
    camera_points = torch.cat((xy, depth[..., None]), dim=-1)
    flat = camera_points.reshape(batch * views, height * width, 3)
    matrices = inverse_calibration.reshape(batch * views, 4, 4)
    return transform_points(flat, matrices).reshape(batch, views, height, width, 3)


def _grid_normals(points: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[..., 1:-1, :] = points[..., 2:, :] - points[..., :-2, :]
    dx[..., 0, :] = points[..., 1, :] - points[..., 0, :]
    dx[..., -1, :] = points[..., -1, :] - points[..., -2, :]
    dy[:, :, 1:-1, :, :] = points[:, :, 2:, :, :] - points[:, :, :-2, :, :]
    dy[:, :, 0, :, :] = points[:, :, 1, :, :] - points[:, :, 0, :, :]
    dy[:, :, -1, :, :] = points[:, :, -1, :, :] - points[:, :, -2, :, :]
    return safe_normalize(torch.cross(dx, dy, dim=-1))


def depth_maps_to_oriented_pointcloud(
    depth_maps: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    inverse_calibrations: Optional[torch.Tensor] = None,
    max_points: Optional[int] = 40_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift front/back depth maps into padded oriented point clouds.

    Args:
        depth_maps: Metric front/back depth [B,2,H,W], zero for background.
        masks: Optional validity mask with the same shape.
        inverse_calibrations: Camera-to-world matrices [B,2,4,4].
        max_points: Uniformly subsample this many valid points per item.

    Returns:
        positions [B,P,3], normals [B,P,3], valid_mask [B,P].
    """
    if depth_maps.ndim != 4 or depth_maps.shape[1] != 2:
        raise ValueError("depth_maps must have shape [B,2,H,W]")
    batch = depth_maps.shape[0]
    valid = depth_maps > 0 if masks is None else masks.bool() & (depth_maps > 0)
    if inverse_calibrations is None:
        inverse_calibrations = default_inverse_calibrations(
            batch, depth_maps.device, depth_maps.dtype
        )
    world = _depth_grid_to_world(depth_maps, inverse_calibrations)
    normals = _grid_normals(world)

    counts = valid.flatten(1).sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each sample must contain at least one valid depth pixel")
    target_count = int(counts.max().item())
    if max_points is not None:
        target_count = min(target_count, int(max_points))
    position_batch, normal_batch, mask_batch = [], [], []
    for batch_index in range(batch):
        item_valid = valid[batch_index].reshape(-1)
        item_positions = world[batch_index].reshape(-1, 3)[item_valid]
        item_normals = normals[batch_index].reshape(-1, 3)[item_valid]
        take = min(target_count, item_positions.shape[0])
        indices = torch.linspace(
            0, item_positions.shape[0] - 1, take, device=depth_maps.device
        ).long()
        selected_positions = item_positions[indices]
        selected_normals = item_normals[indices]
        padding = target_count - take
        if padding:
            selected_positions = F.pad(selected_positions, (0, 0, 0, padding))
            selected_normals = F.pad(selected_normals, (0, 0, 0, padding))
        position_batch.append(selected_positions)
        normal_batch.append(selected_normals)
        mask_batch.append(
            torch.arange(target_count, device=depth_maps.device) < take
        )
    return (
        torch.stack(position_batch),
        torch.stack(normal_batch),
        torch.stack(mask_batch),
    )


class OrthographicProjector(nn.Module):
    def forward(self, points: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        return project_points(points, calibration, perspective=False)

