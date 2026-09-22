"""Area-uniform surface sampling with base color, per geometry.

Points are allocated to geometries by area and drawn with a fixed seed per geometry. The base
color is the texture read bilinearly at the interpolated texture coordinate with glTF REPEAT
wrap, times the base color factor, returned as linear RGB in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .loading import load_scene_geoms

N_SAMPLE = 40_000


@dataclass
class Samples:
    p: np.ndarray   # (N,3) positions
    n: np.ndarray   # (N,3) unit face normals
    a: np.ndarray   # (N,3) albedo, linear RGB in [0,1]


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _texture_array(material):
    img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if img is None:
        return None
    # a palette-mode image decodes to palette indices unless converted
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float64) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3]


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample img (H,W,3) at trimesh UV (v-up); glTF wraps with REPEAT."""
    h, w = img.shape[:2]
    uvw = uv % 1.0
    u = uvw[:, 0] * (w - 1)
    v = (1.0 - uvw[:, 1]) * (h - 1)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _base_color_factor(mat) -> np.ndarray:
    """trimesh hands the factor back as bytes after a round trip through its material classes and
    as floats when read straight from the file; a factor above 1 can only be bytes."""
    fac = getattr(mat, "baseColorFactor", None)
    if fac is None:
        return np.ones(3)
    fac = np.asarray(fac, dtype=float).ravel()
    if fac.size < 3:
        return np.ones(3)
    return fac[:3] / 255.0 if fac.max() > 1.0 else fac[:3]


def _albedo(g, fidx, bary) -> np.ndarray:
    """glTF base color at sampled points: texture times factor, else vertex colors, else the
    factor alone."""
    vis = g.visual
    faces = g.faces[fidx]
    mat = getattr(vis, "material", None)
    uv = getattr(vis, "uv", None)
    img = _texture_array(mat) if mat is not None else None
    if img is not None:
        if uv is None or len(uv) != len(g.vertices):
            raise ValueError("geometry has a base-color texture but no matching UV array")
        uv_s = (bary[:, :, None] * np.asarray(uv)[faces]).sum(1)
        return srgb_to_linear(_bilinear(img, uv_s) * _base_color_factor(mat))
    vc = getattr(vis, "vertex_colors", None)
    if mat is None and vc is not None and len(vc) == len(g.vertices):
        col = (bary[:, :, None] * (np.asarray(vc)[:, :3] / 255.0)[faces]).sum(1)
        return srgb_to_linear(col)
    if mat is not None:
        return srgb_to_linear(np.tile(_base_color_factor(mat), (len(fidx), 1)))
    raise ValueError("geometry has neither a material nor vertex colors")


def sample_scene(glb: str, n: int = N_SAMPLE, seed: int = 0) -> Samples:
    """Area-uniform albedo samples over every geometry of the scene."""
    geoms = load_scene_geoms(glb)
    total = sum(g.area for g in geoms)
    P, N, A = [], [], []
    for gi, g in enumerate(geoms):
        k = max(int(n * g.area / total), 1)
        pts, fidx = trimesh.sample.sample_surface(g, k, seed=seed + gi)
        bary = trimesh.triangles.points_to_barycentric(g.triangles[fidx], pts)
        P.append(np.asarray(pts))
        N.append(np.asarray(g.face_normals)[fidx])
        A.append(_albedo(g, fidx, bary))
    return Samples(p=np.concatenate(P), n=np.concatenate(N), a=np.concatenate(A))
