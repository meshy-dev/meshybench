"""Unshaded base-color rendering: six axis-aligned orthographic views with nvdiffrast.

A view looks at the origin from distance 6 along an axis with a fixed half-width of XMAG RMS
radii, flat base color over white, textures sampled trilinearly with glTF REPEAT wrap, alpha
below 0.5 cut out, image row 0 at the top.
"""
from __future__ import annotations

import numpy as np

VIEWS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
XMAG = 2.0
CAMERA_DISTANCE = 6.0
ALPHA_CUTOFF = 0.5

_CTX = {}


def _look_at_rows(eye):
    eye = np.asarray(eye, dtype=float)
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd, eye


def _base_color_factor(m) -> np.ndarray:
    """trimesh hands the factor back as bytes after a round trip through its material classes
    and as floats when read straight from the file; a factor above 1 can only be bytes."""
    fac = np.asarray(m.baseColorFactor[:3], np.float32)
    return fac / 255.0 if fac.max() > 1.001 else fac


def prep_geoms(geoms):
    """GPU buffers per geometry: vertices, faces, texture coordinates, texture, vertex colors,
    base color factor. A texture is uploaded once however many geometries share it."""
    import torch

    dev = torch.device("cuda")
    out = []
    tex_cache = {}
    for g in geoms:
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        V = torch.as_tensor(np.asarray(g.vertices, np.float32), device=dev)
        F = torch.as_tensor(np.asarray(g.faces, np.int32), device=dev)
        uv = tex = vcol = None
        fac = np.array([1.0, 1.0, 1.0], np.float32)
        vis = g.visual
        m = getattr(vis, "material", None)
        if m is not None and getattr(m, "baseColorFactor", None) is not None:
            fac = _base_color_factor(m)
        t = getattr(m, "baseColorTexture", None) if m is not None else None
        u = getattr(vis, "uv", None)
        if t is not None:
            if u is None or len(u) != len(g.vertices):
                raise ValueError("geometry has a base-color texture but no matching UV array")
            uv = torch.as_tensor(np.asarray(u, np.float32), device=dev)
            if id(t) in tex_cache:
                tex = tex_cache[id(t)]
            else:
                # trimesh texture coordinates have their origin at the bottom left; dr.texture
                # indexes row 0 at the top
                timg = np.asarray(t.convert("RGBA"), np.float32)[::-1] / 255.0
                tex = torch.as_tensor(timg, device=dev).unsqueeze(0).contiguous()
                tex_cache[id(t)] = tex
        elif getattr(vis, "vertex_colors", None) is not None and len(vis.vertex_colors) == len(g.vertices):
            vcol = torch.as_tensor(np.asarray(vis.vertex_colors[:, :3], np.float32) / 255.0, device=dev)
        out.append({"V": V, "F": F, "uv": uv, "tex": tex, "vcol": vcol,
                    "fac": torch.as_tensor(fac, device=dev)})
    return out


def _mip_levels(h: int, w: int) -> int:
    """nvdiffrast cannot halve an odd extent above 1."""
    n = 0
    while h > 1 or w > 1:
        if (h > 1 and h % 2) or (w > 1 and w % 2):
            break
        h, w, n = max(h // 2, 1), max(w // 2, 1), n + 1
    return n


def render_views(prepped, res, transform):
    """[(color uint8 res x res x 3 over white, foreground mask float), ...] for VIEWS.
    transform = (R, c, t) places vertices as c * R @ x + t before the camera."""
    import torch

    import nvdiffrast.torch as dr

    dev = torch.device("cuda")
    if "c" not in _CTX:
        _CTX["c"] = dr.RasterizeCudaContext()
    ctx = _CTX["c"]
    R, c, t = transform
    Rm = torch.as_tensor(np.asarray(R, np.float32), device=dev)
    tv = torch.as_tensor(np.asarray(t, np.float32), device=dev)
    c = float(c)
    outs = []
    for v in VIEWS:
        right, up, fwd, eye = _look_at_rows(np.asarray(v, float) * CAMERA_DISTANCE)
        Rt = torch.as_tensor(np.stack([right, up, fwd]).astype(np.float32), device=dev)
        eye_t = torch.as_tensor(eye.astype(np.float32), device=dev)
        color = torch.ones((res, res, 3), device=dev)
        zbuf = torch.full((res, res), 1e9, device=dev)
        for gp in prepped:
            P = c * (gp["V"] @ Rm.T) + tv
            pc = (P - eye_t) @ Rt.T
            x = pc[:, 0] / XMAG
            y = pc[:, 1] / XMAG
            depth = pc[:, 2]
            z = (depth - CAMERA_DISTANCE) / 20.0
            pos = torch.stack([x, y, z, torch.ones_like(x)], -1).unsqueeze(0)
            rast, rast_db = dr.rasterize(ctx, pos.contiguous(), gp["F"], resolution=[res, res])
            hit = rast[0, :, :, 3] > 0
            if not hit.any():
                continue
            dpx, _ = dr.interpolate(depth.reshape(1, -1, 1).contiguous(), rast, gp["F"])
            dpx = dpx[0, :, :, 0]
            alpha = None
            if gp["tex"] is not None:
                uvpx, uv_da = dr.interpolate(gp["uv"].unsqueeze(0).contiguous(), rast, gp["F"],
                                             rast_db=rast_db, diff_attrs="all")
                rgba = dr.texture(gp["tex"], uvpx, uv_da, filter_mode="linear-mipmap-linear",
                                  boundary_mode="wrap",
                                  max_mip_level=_mip_levels(gp["tex"].shape[1], gp["tex"].shape[2]))[0]
                col, alpha = rgba[:, :, :3], rgba[:, :, 3]
            elif gp["vcol"] is not None:
                cpx, _ = dr.interpolate(gp["vcol"].unsqueeze(0).contiguous(), rast, gp["F"])
                col = cpx[0]
            else:
                col = torch.ones((res, res, 3), device=dev)
            col = col * gp["fac"]
            closer = hit & (dpx < zbuf)
            if alpha is not None:
                closer = closer & (alpha > ALPHA_CUTOFF)
            zbuf = torch.where(closer, dpx, zbuf)
            color = torch.where(closer.unsqueeze(-1), col, color)
        mask = zbuf < 1e8
        img = (color.clamp(0, 1) * 255).byte().cpu().numpy()
        outs.append((np.flipud(img).copy(), np.flipud(mask.float().cpu().numpy()).copy()))
    return outs
