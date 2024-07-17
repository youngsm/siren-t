from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LogNorm
from photonlib import PhotonLib


def visualize_plib(
    plib: PhotonLib | str, channel_id: int = None, cmap: str = "viridis"
) -> None:
    if isinstance(plib, str):
        plib = PhotonLib.load(plib)
    positions = plib.meta.voxel_to_coord(np.arange(len(plib.vis)))
    values = plib.vis.mean(axis=-1) if not channel_id else plib.vis[:, channel_id]
    pointcloud = trimesh.PointCloud(positions)
    norm = LogNorm()
    pointcloud.colors = getattr(plt.cm, cmap)(norm(values))
    scene = trimesh.Scene([pointcloud])
    return scene.show()