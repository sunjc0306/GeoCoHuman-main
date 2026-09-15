from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .geometry import safe_normalize


class PointwiseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _face_vertices(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    feature_dim = vertices.shape[-1]
    if faces.ndim == 2:
        return vertices[:, faces.long()]
    batch, face_count, _ = faces.shape
    flat = faces.reshape(batch, face_count * 3).long()
    gathered = torch.gather(vertices, 1, flat[..., None].expand(-1, -1, feature_dim))
    return gathered.reshape(batch, face_count, 3, feature_dim)


def _closest_point_on_triangle_pairs(
    points: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    """Closest triangle points with broadcast-compatible leading dimensions."""
    a, b, c = triangles.unbind(dim=-2)

    def segment_closest(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        edge = end - start
        amount = ((points - start) * edge).sum(dim=-1, keepdim=True)
        amount = amount / edge.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return start + amount.clamp(0.0, 1.0) * edge

    ab_closest = segment_closest(a, b)
    bc_closest = segment_closest(b, c)
    ca_closest = segment_closest(c, a)
    edge_candidates = torch.stack((ab_closest, bc_closest, ca_closest), dim=-2)
    edge_distance = (edge_candidates - points.unsqueeze(-2)).square().sum(dim=-1)
    edge_choice = edge_distance.argmin(dim=-1)
    edge_closest = torch.gather(
        edge_candidates,
        -2,
        edge_choice[..., None, None].expand(*edge_choice.shape, 1, 3),
    ).squeeze(-2)

    ab, ac = b - a, c - a
    normal = torch.cross(ab, ac, dim=-1)
    plane_distance = ((points - a) * normal).sum(dim=-1, keepdim=True)
    projection = points - plane_distance * normal / normal.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    relative = projection - a
    d00 = (ab * ab).sum(dim=-1)
    d01 = (ab * ac).sum(dim=-1)
    d11 = (ac * ac).sum(dim=-1)
    d20 = (relative * ab).sum(dim=-1)
    d21 = (relative * ac).sum(dim=-1)
    denominator = d00 * d11 - d01.square()
    bary_v = (d11 * d20 - d01 * d21) / denominator.clamp_min(1e-12)
    bary_w = (d00 * d21 - d01 * d20) / denominator.clamp_min(1e-12)
    bary_u = 1.0 - bary_v - bary_w
    inside = (
        (denominator.abs() > 1e-12)
        & (bary_u >= 0)
        & (bary_v >= 0)
        & (bary_w >= 0)
    )
    return torch.where(inside[..., None], projection, edge_closest)


def closest_faces_exact(
    query_points: torch.Tensor,
    triangles: torch.Tensor,
    query_chunk: int = 64,
    face_chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact closest-face lookup with bounded-memory pure PyTorch chunks."""
    batch, query_count, _ = query_points.shape
    face_count = triangles.shape[1]
    all_indices, all_closest = [], []
    for batch_index in range(batch):
        batch_indices, batch_closest = [], []
        for query_start in range(0, query_count, query_chunk):
            points = query_points[batch_index, query_start : query_start + query_chunk]
            best_distance = torch.full(
                (points.shape[0],), float("inf"), device=points.device, dtype=points.dtype
            )
            best_index = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
            best_point = torch.zeros_like(points)
            for face_start in range(0, face_count, face_chunk):
                face_group = triangles[
                    batch_index, face_start : face_start + face_chunk
                ]
                closest = _closest_point_on_triangle_pairs(
                    points[:, None, :], face_group[None, :, :, :]
                )
                distances = (closest - points[:, None, :]).square().sum(dim=-1)
                local_distance, local_index = distances.min(dim=1)
                local_point = closest[
                    torch.arange(points.shape[0], device=points.device), local_index
                ]
                improved = local_distance < best_distance
                best_distance = torch.where(improved, local_distance, best_distance)
                best_index = torch.where(improved, local_index + face_start, best_index)
                best_point = torch.where(improved[:, None], local_point, best_point)
            batch_indices.append(best_index)
            batch_closest.append(best_point)
        all_indices.append(torch.cat(batch_indices))
        all_closest.append(torch.cat(batch_closest))
    return torch.stack(all_indices), torch.stack(all_closest)


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    batch_indices = torch.arange(values.shape[0], device=values.device).reshape(batch_shape)
    return values[batch_indices, indices]


class SMPLMeshPFA(nn.Module):
    """Eq. (12): two-ring, inverse-square proximity aggregation on SMPL faces."""

    def __init__(
        self,
        semantic_dim: int = 3,
        aggregate_dim: int = 64,
        hidden_dim: int = 128,
        query_chunk: int = 64,
        face_chunk: int = 2048,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.raw_dim = 1 + 3 + self.semantic_dim
        self.output_dim = self.raw_dim + aggregate_dim
        self.query_chunk = int(query_chunk)
        self.face_chunk = int(face_chunk)
        self.aggregate = PointwiseMLP(3 + self.raw_dim, hidden_dim, aggregate_dim)

    def forward(
        self,
        query_points: torch.Tensor,
        smpl_vertices: torch.Tensor,
        smpl_faces: torch.Tensor,
        vertex_semantics: torch.Tensor,
        face_neighbors: torch.Tensor,
        face_neighbor_mask: torch.Tensor,
        nearest_face_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        triangles = _face_vertices(smpl_vertices, smpl_faces)
        face_normals = safe_normalize(
            torch.cross(
                triangles[:, :, 1] - triangles[:, :, 0],
                triangles[:, :, 2] - triangles[:, :, 0],
                dim=-1,
            )
        )
        semantic_triangles = _face_vertices(vertex_semantics, smpl_faces)
        face_semantics = semantic_triangles.mean(dim=2)
        if nearest_face_indices is None:
            nearest_face_indices, _ = closest_faces_exact(
                query_points, triangles, self.query_chunk, self.face_chunk
            )

        if face_neighbors.ndim == 2:
            neighbor_indices = face_neighbors.to(query_points.device)[nearest_face_indices]
            neighbor_mask = face_neighbor_mask.to(query_points.device)[nearest_face_indices]
        else:
            neighbor_indices = _batch_gather(face_neighbors.long(), nearest_face_indices)
            neighbor_mask = _batch_gather(face_neighbor_mask.bool(), nearest_face_indices)
        neighbor_triangles = _batch_gather(triangles, neighbor_indices)
        neighbor_normals = _batch_gather(face_normals, neighbor_indices)
        neighbor_semantics = _batch_gather(face_semantics, neighbor_indices)
        projection_points = _closest_point_on_triangle_pairs(
            query_points[:, :, None, :], neighbor_triangles
        )
        displacement = query_points[:, :, None, :] - projection_points
        signed_distance = (displacement * neighbor_normals).sum(dim=-1, keepdim=True)
        neighbor_features = torch.cat(
            (signed_distance, neighbor_normals, neighbor_semantics), dim=-1
        )
        distance_squared = displacement.square().sum(dim=-1).clamp_min(1e-8)
        weights = neighbor_mask.to(query_points.dtype) / distance_squared
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        aggregated = (weights[..., None] * neighbor_features).sum(dim=2)
        learned = self.aggregate(torch.cat((query_points, aggregated), dim=-1))
        closest_feature = neighbor_features[:, :, 0]
        return torch.cat((closest_feature, learned), dim=-1)


class PointCloudPFA(nn.Module):
    """Eq. (14): radius-ball aggregation on the oriented clothes point cloud."""

    def __init__(
        self,
        radius: float = 0.2,
        max_neighbors: int = 32,
        aggregate_dim: int = 64,
        hidden_dim: int = 128,
        query_chunk: int = 512,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.query_chunk = int(query_chunk)
        self.raw_dim = 1 + 3 + 3
        self.output_dim = self.raw_dim + aggregate_dim
        self.aggregate = PointwiseMLP(3 + self.raw_dim, hidden_dim, aggregate_dim)

    def forward(
        self,
        query_points: torch.Tensor,
        point_positions: torch.Tensor,
        point_normals: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        neighbor_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if point_mask is not None and (~point_mask.bool()).all(dim=1).any():
            raise ValueError("Every point-cloud prior needs at least one valid point")
        neighbor_count = min(self.max_neighbors, point_positions.shape[1])
        outputs = []
        for start in range(0, query_points.shape[1], self.query_chunk):
            query = query_points[:, start : start + self.query_chunk]
            if neighbor_indices is None:
                distances = torch.cdist(query, point_positions)
                if point_mask is not None:
                    distances = distances.masked_fill(~point_mask[:, None].bool(), float("inf"))
                neighbor_distance, chunk_indices = torch.topk(
                    distances, k=neighbor_count, dim=-1, largest=False, sorted=True
                )
            else:
                chunk_indices = neighbor_indices[:, start : start + self.query_chunk]
                chunk_indices = chunk_indices[..., :neighbor_count].long()
                neighbor_positions = _batch_gather(point_positions, chunk_indices)
                neighbor_distance = (
                    query[:, :, None, :] - neighbor_positions
                ).norm(dim=-1)
            neighbor_positions = _batch_gather(point_positions, chunk_indices)
            neighbor_normals = safe_normalize(_batch_gather(point_normals, chunk_indices))
            displacement = query[:, :, None, :] - neighbor_positions
            signed_distance = (displacement * neighbor_normals).sum(dim=-1, keepdim=True)
            neighbor_features = torch.cat(
                (signed_distance, displacement, neighbor_normals), dim=-1
            )
            ball_mask = neighbor_distance <= self.radius
            ball_mask[..., 0] = True  # nearest fallback for an empty radius query
            weights = ball_mask.to(query.dtype) / neighbor_distance.square().clamp_min(1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            aggregated = (weights[..., None] * neighbor_features).sum(dim=2)
            learned = self.aggregate(torch.cat((query, aggregated), dim=-1))
            outputs.append(torch.cat((neighbor_features[:, :, 0], learned), dim=-1))
        return torch.cat(outputs, dim=1)
