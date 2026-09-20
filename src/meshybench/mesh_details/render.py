"""Eight-view face-normal rasterization for the mesh details metric.

The mesh is normalized to a fixed extent, rotated so the view direction lands
on +z, and rasterized orthographically with nvdiffrast into an S x S buffer of
camera facing face normals plus a coverage mask. Requires torch + nvdiffrast
on a CUDA device; deterministic (fixed views, no sampling).
"""
from __future__ import annotations

import numpy as np
import trimesh

RASTER = 8192
W = 2.0            # normalized extent of the longest axis
XMAG = 1.02        # orthographic half width relative to W / 2 (a 2% margin)
_CTX: dict = {}

_R3 = 1.0 / np.sqrt(3.0)
# camera directions in the mesh frame (Y up); the camera looks along -d
VIEWS = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "top": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "diag_pfr": (_R3, _R3, _R3),
    "diag_pfl": (-_R3, _R3, _R3),
    "diag_pbr": (_R3, _R3, -_R3),
}


def _rot_to_z(d):
    """Rotation taking unit vector d onto +z (the raster's depth axis)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(d, z)
    c = float(d @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def raster_view(mesh: trimesh.Trimesh, d, S: int = RASTER):
    """(nrm, cover, tid) at S x S for the view from direction d: camera facing face
    normals per raster pixel, the coverage mask and the 1-based triangle id (0 = background)."""
    import torch
    import nvdiffrast.torch as dr

    if "nvdr" not in _CTX:
        _CTX["nvdr"] = dr.RasterizeCudaContext()
    dev = torch.device("cuda")
    R = _rot_to_z(d)
    mm = mesh.copy()
    mm.vertices = mm.vertices @ R.T
    if mm.extents.max() <= 0:
        raise ValueError("degenerate extents")
    mm.vertices = mm.vertices * (W / mm.extents.max())
    V = np.asarray(mm.vertices) - mm.bounds.mean(axis=0)
    F = np.asarray(mm.faces, np.int32)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(ln > 1e-14, fn / np.maximum(ln, 1e-30), 0.0)
    pos = torch.zeros((1, len(V), 4), dtype=torch.float32, device=dev)
    pos[0, :, 0] = torch.as_tensor(V[:, 0] / XMAG, dtype=torch.float32)
    pos[0, :, 1] = torch.as_tensor(V[:, 1] / XMAG, dtype=torch.float32)
    pos[0, :, 2] = torch.as_tensor(-V[:, 2] / 3.0, dtype=torch.float32)
    pos[0, :, 3] = 1.0
    rast, _ = dr.rasterize(_CTX["nvdr"], pos, torch.as_tensor(F, device=dev), resolution=[S, S])
    tid = rast[0, :, :, 3].long()
    del rast, pos
    cover = tid > 0
    fn_t = torch.as_tensor(np.vstack([[0, 0, 0], fn]), dtype=torch.float32, device=dev)
    nrm = fn_t[tid]
    nrm = torch.where(nrm[:, :, 2:3] < 0, -nrm, nrm)
    torch.cuda.empty_cache()
    return nrm, cover, tid
