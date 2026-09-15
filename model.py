from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn

from .cpif import CollaborativeImplicitFunction
from .geometry import project_points, safe_normalize, sample_image_features
from .image_encoder import StackedHourglassEncoder
from .pfa import PointCloudPFA, SMPLMeshPFA


class GeoCoHumanImplicitModel(nn.Module):
    """Prior Feature Extraction + three-head CPIF stage of GeoCoHuman."""

    def __init__(
        self,
        image_feature_dim: int = 64,
        image_encoder_width: int = 128,
        image_stacks: int = 4,
        image_hourglass_depth: int = 3,
        smpl_semantic_dim: int = 3,
        body_aggregate_dim: int = 64,
        clothes_aggregate_dim: int = 64,
        pfa_hidden_dim: int = 128,
        point_radius: float = 0.2,
        point_max_neighbors: int = 32,
        prior_hidden_dims: Sequence[int] = (512, 256, 128, 128),
        latent_hidden_dims: Sequence[int] = (512, 256, 128, 128),
        skip_layers: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.smpl_semantic_dim = int(smpl_semantic_dim)
        self.image_encoder = StackedHourglassEncoder(
            width=image_encoder_width,
            feature_dim=image_feature_dim,
            num_stacks=image_stacks,
            hourglass_depth=image_hourglass_depth,
            blocks_per_stack=2,
        )
        self.body_pfa = SMPLMeshPFA(
            semantic_dim=smpl_semantic_dim,
            aggregate_dim=body_aggregate_dim,
            hidden_dim=pfa_hidden_dim,
        )
        self.clothes_pfa = PointCloudPFA(
            radius=point_radius,
            max_neighbors=point_max_neighbors,
            aggregate_dim=clothes_aggregate_dim,
            hidden_dim=pfa_hidden_dim,
        )
        self.cpif = CollaborativeImplicitFunction(
            visual_dim=image_feature_dim,
            body_dim=self.body_pfa.output_dim,
            clothes_dim=self.clothes_pfa.output_dim,
            prior_hidden_dims=prior_hidden_dims,
            latent_hidden_dims=latent_hidden_dims,
            skip_layers=skip_layers,
        )

    def _default_semantics(self, vertices: torch.Tensor) -> torch.Tensor:
        minimum = vertices.amin(dim=1, keepdim=True)
        maximum = vertices.amax(dim=1, keepdim=True)
        normalized = (vertices - minimum) / (maximum - minimum).clamp_min(1e-8)
        if self.smpl_semantic_dim == 3:
            return normalized
        if self.smpl_semantic_dim < 3:
            return normalized[..., : self.smpl_semantic_dim]
        padding = torch.zeros(
            *normalized.shape[:-1], self.smpl_semantic_dim - 3,
            device=vertices.device, dtype=vertices.dtype
        )
        return torch.cat((normalized, padding), dim=-1)

    def extract_image_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def query_from_features(
        self,
        image_features: torch.Tensor,
        query_points: torch.Tensor,
        calibration: torch.Tensor,
        smpl_vertices: torch.Tensor,
        smpl_faces: torch.Tensor,
        face_neighbors: torch.Tensor,
        face_neighbor_mask: torch.Tensor,
        point_positions: torch.Tensor,
        point_normals: torch.Tensor,
        smpl_vertex_semantics: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        nearest_face_indices: Optional[torch.Tensor] = None,
        point_neighbor_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if smpl_vertex_semantics is None:
            smpl_vertex_semantics = self._default_semantics(smpl_vertices)
        projected = project_points(query_points, calibration)
        visual_features = sample_image_features(image_features, projected)
        body_features = self.body_pfa(
            query_points,
            smpl_vertices,
            smpl_faces,
            smpl_vertex_semantics,
            face_neighbors,
            face_neighbor_mask,
            nearest_face_indices,
        )
        clothes_features = self.clothes_pfa(
            query_points, point_positions, point_normals, point_mask, point_neighbor_indices
        )
        prediction = self.cpif(
            visual_features, body_features, clothes_features, query_points
        )
        prediction.update(
            {
                "visual_features": visual_features,
                "body_features": body_features,
                "clothes_features": clothes_features,
            }
        )
        return prediction

    def forward(self, image: torch.Tensor, **query_inputs) -> Dict[str, torch.Tensor]:
        return self.query_from_features(self.extract_image_features(image), **query_inputs)

    def predict_with_normals(
        self,
        image: torch.Tensor,
        normal_epsilon: float = 1e-3,
        recompute_nearest_for_normals: bool = False,
        **query_inputs,
    ) -> Dict[str, torch.Tensor]:
        """Predict SDF normals using the paper's central finite differences."""
        image_features = self.extract_image_features(image)
        base = self.query_from_features(image_features, **query_inputs)
        points = query_inputs["query_points"]
        gradients = []
        for axis in range(3):
            offset = torch.zeros_like(points)
            offset[..., axis] = normal_epsilon
            plus_inputs = dict(query_inputs)
            minus_inputs = dict(query_inputs)
            plus_inputs["query_points"] = points + offset
            minus_inputs["query_points"] = points - offset
            if recompute_nearest_for_normals:
                plus_inputs["nearest_face_indices"] = None
                minus_inputs["nearest_face_indices"] = None
            plus = self.query_from_features(image_features, **plus_inputs)["sdf"]
            minus = self.query_from_features(image_features, **minus_inputs)["sdf"]
            gradients.append((plus - minus) / (2.0 * normal_epsilon))
        base["normal"] = safe_normalize(torch.cat(gradients, dim=-1))
        return base
