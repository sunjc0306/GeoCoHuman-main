from __future__ import annotations

from typing import Dict, Optional

import torch


def icon_batch_to_geocohuman(
    icon_batch: Dict[str, torch.Tensor],
    point_positions: torch.Tensor,
    point_normals: torch.Tensor,
    face_neighbors: torch.Tensor,
    face_neighbor_mask: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Map public ICON batch keys to this reproduction's model signature.

    ICON stores query samples as [B,3,N], while this project uses [B,N,3].
    SMPL colormap values are used as the paper's body semantic code.
    """
    required = ("image", "sample", "calib", "smpl_verts", "smpl_faces")
    missing = [key for key in required if key not in icon_batch]
    if missing:
        raise ValueError(f"ICON batch is missing keys: {missing}")
    samples = icon_batch["sample"]
    query_points = samples.transpose(1, 2) if samples.shape[1] == 3 else samples
    result = {
        "image": icon_batch["image"],
        "query_points": query_points.contiguous(),
        "calibration": icon_batch["calib"],
        "smpl_vertices": icon_batch["smpl_verts"],
        "smpl_faces": icon_batch["smpl_faces"],
        "face_neighbors": face_neighbors,
        "face_neighbor_mask": face_neighbor_mask,
        "point_positions": point_positions,
        "point_normals": point_normals,
    }
    if "smpl_cmap" in icon_batch:
        result["smpl_vertex_semantics"] = icon_batch["smpl_cmap"]
    if point_mask is not None:
        result["point_mask"] = point_mask
    return result

