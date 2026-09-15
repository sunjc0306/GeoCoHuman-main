from __future__ import annotations

from collections import defaultdict, deque
from typing import Tuple

import numpy as np
import torch


def build_face_ring_neighbors(
    faces: np.ndarray | torch.Tensor, rings: int = 2, max_neighbors: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute K-ring face neighbors using shared mesh edges.

    The closest face itself is always stored at column zero. Extra entries are
    padded with the face itself and disabled by the returned boolean mask.
    """
    face_array = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    face_array = face_array.astype(np.int64, copy=False)
    edge_to_faces = defaultdict(list)
    for face_index, (a, b, c) in enumerate(face_array):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted((int(u), int(v))))].append(face_index)
    adjacency = [set() for _ in range(len(face_array))]
    for attached in edge_to_faces.values():
        for source in attached:
            adjacency[source].update(index for index in attached if index != source)

    indices = np.zeros((len(face_array), max_neighbors), dtype=np.int64)
    mask = np.zeros((len(face_array), max_neighbors), dtype=np.bool_)
    for root in range(len(face_array)):
        ordered = [root]
        visited = {root}
        queue = deque((neighbor, 1) for neighbor in sorted(adjacency[root]))
        while queue and len(ordered) < max_neighbors:
            current, distance = queue.popleft()
            if current in visited or distance > rings:
                continue
            visited.add(current)
            ordered.append(current)
            if distance < rings:
                queue.extend((neighbor, distance + 1) for neighbor in sorted(adjacency[current]))
        indices[root] = root
        indices[root, : len(ordered)] = ordered
        mask[root, : len(ordered)] = True
    return torch.from_numpy(indices), torch.from_numpy(mask)

