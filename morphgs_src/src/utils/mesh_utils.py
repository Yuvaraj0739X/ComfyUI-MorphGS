# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
from typing import Tuple

import fpsample
import numpy as np
from PIL import Image
import trimesh


def load_mesh(mesh_path):
    """
    Load a mesh from the given path. If a texture.png sits next to the mesh,
    it is attached as the material image.
    """
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)

    texture_path = os.path.join(os.path.dirname(mesh_path), "texture.png")
    if os.path.exists(texture_path):
        mesh.visual.material.image = Image.open(texture_path).convert("RGBA")

    return mesh


def fps_pointcloud(
    p: np.ndarray,
    n_sample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples a point cloud using farthest point sampling.
    """
    inds = fpsample.bucket_fps_kdline_sampling(p, n_sample, h=3).astype(np.int32)
    return p[inds], inds
