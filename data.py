from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .topology import build_face_ring_neighbors


class ImplicitReconstructionDataset(Dataset):
    """Load fixed-size GeoCoHuman training samples from NPZ records."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: Tuple[int, int] = (512, 512),
        num_queries: int = 2048,
        num_prior_points: int = 40_000,
        face_rings: int = 2,
        max_face_neighbors: int = 32,
        training: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.image_size = tuple(map(int, image_size))
        self.num_queries = int(num_queries)
        self.num_prior_points = int(num_prior_points)
        self.face_rings = int(face_rings)
        self.max_face_neighbors = int(max_face_neighbors)
        self.training = bool(training)
        self.records = self._read_manifest()
        self._topology_cache: Dict[bytes, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _read_manifest(self) -> List[Mapping[str, str]]:
        records = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "image" not in record or "sample" not in record:
                    raise ValueError(
                        f"{self.manifest}:{line_number} requires image and sample"
                    )
                records.append(record)
        if not records:
            raise ValueError(f"Empty manifest: {self.manifest}")
        return records

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_image(self, path: Path) -> torch.Tensor:
        height, width = self.image_size
        image = Image.open(path).convert("RGB").resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)

    def _select(self, count: int, target: int) -> np.ndarray:
        if self.training:
            return np.random.choice(count, target, replace=count < target)
        if count >= target:
            return np.linspace(0, count - 1, target).astype(np.int64)
        return np.resize(np.arange(count, dtype=np.int64), target)

    def _point_prior(
        self, positions: np.ndarray, normals: np.ndarray, preserve_order: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(positions) == 0:
            raise ValueError("Oriented point-cloud prior is empty")
        if preserve_order and len(positions) > self.num_prior_points:
            raise ValueError(
                "Precomputed point_neighbor_indices require num_prior_points to cover the full point cloud"
            )
        count = min(len(positions), self.num_prior_points)
        indices = np.arange(count) if preserve_order else self._select(len(positions), count)
        selected_positions = torch.from_numpy(positions[indices]).float()
        selected_normals = torch.from_numpy(normals[indices]).float()
        valid = torch.ones(count, dtype=torch.bool)
        if count < self.num_prior_points:
            padding = self.num_prior_points - count
            selected_positions = torch.cat(
                (selected_positions, torch.zeros(padding, 3)), dim=0
            )
            selected_normals = torch.cat(
                (selected_normals, torch.zeros(padding, 3)), dim=0
            )
            valid = torch.cat((valid, torch.zeros(padding, dtype=torch.bool)))
        return selected_positions, selected_normals, valid

    def _topology(self, faces: np.ndarray, archive) -> Tuple[torch.Tensor, torch.Tensor]:
        if "face_neighbors" in archive and "face_neighbor_mask" in archive:
            return (
                torch.from_numpy(archive["face_neighbors"]).long(),
                torch.from_numpy(archive["face_neighbor_mask"]).bool(),
            )
        cache_key = faces.tobytes()
        if cache_key not in self._topology_cache:
            self._topology_cache[cache_key] = build_face_ring_neighbors(
                faces, self.face_rings, self.max_face_neighbors
            )
        return self._topology_cache[cache_key]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        archive = np.load(self._path(record["sample"]))
        required = (
            "query_points",
            "occupancy",
            "sdf",
            "smpl_vertices",
            "smpl_faces",
            "point_positions",
            "point_normals",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{record['sample']} missing arrays: {missing}")

        query_indices = self._select(len(archive["query_points"]), self.num_queries)
        faces = archive["smpl_faces"].astype(np.int64)
        face_neighbors, face_neighbor_mask = self._topology(faces, archive)
        point_positions, point_normals, point_mask = self._point_prior(
            archive["point_positions"].astype(np.float32),
            archive["point_normals"].astype(np.float32),
            preserve_order="point_neighbor_indices" in archive,
        )
        result: Dict[str, torch.Tensor | str] = {
            "id": str(record.get("id", index)),
            "image": self._load_image(self._path(record["image"])),
            "query_points": torch.from_numpy(archive["query_points"][query_indices]).float(),
            "occupancy": torch.from_numpy(archive["occupancy"][query_indices]).float().reshape(-1, 1),
            "sdf": torch.from_numpy(archive["sdf"][query_indices]).float().reshape(-1, 1),
            "smpl_vertices": torch.from_numpy(archive["smpl_vertices"]).float(),
            "smpl_faces": torch.from_numpy(faces).long(),
            "face_neighbors": face_neighbors,
            "face_neighbor_mask": face_neighbor_mask,
            "point_positions": point_positions,
            "point_normals": point_normals,
            "point_mask": point_mask,
            "calibration": torch.from_numpy(
                archive["calibration"] if "calibration" in archive else np.eye(4)
            ).float(),
        }
        optional_arrays = {
            "query_normals": "normal_target",
            "surface_mask": "normal_mask",
            "smpl_vertex_semantics": "smpl_vertex_semantics",
            "nearest_face_indices": "nearest_face_indices",
            "point_neighbor_indices": "point_neighbor_indices",
        }
        for source, destination in optional_arrays.items():
            if source in archive:
                values = archive[source]
                if source in {
                    "query_normals", "surface_mask", "nearest_face_indices",
                    "point_neighbor_indices",
                }:
                    values = values[query_indices]
                tensor = torch.from_numpy(values)
                if source in {"nearest_face_indices", "point_neighbor_indices"}:
                    tensor = tensor.long()
                elif source == "surface_mask":
                    tensor = tensor.float().reshape(-1, 1)
                else:
                    tensor = tensor.float()
                result[destination] = tensor
        return result
