"""Mesh-Texture Agreement: whether a model's texture boundaries sit on its geometry ridges,
reference free.

On SURF_N_SAMPLE area-uniform surface points scaled to a unit bounding sphere, each point's
SURF_K nearest neighbours give its curvature (mean 1 - |n.n'|) and, after SURF_LP_PASSES
neighbourhood averages of the albedo, its texture edge strength (largest Lab distance to a
neighbour). Points with curvature above SURF_CURV_RIDGE are ridges, points with edge strength
above SURF_TEDGE_STRONG are texture edges. The co-change domain is every ridge within
SURF_CAPTURE_R of a texture edge and every texture edge within SURF_CAPTURE_R of a ridge; each
member's offset d to the nearest point of the other kind scores exp(-max(d - SURF_OFFSET_TOL, 0)
/ SURF_OFFSET_S), and the score is the mean weighted by ridge curvature and edge strength, each
scaled by its own maximum. A uniform albedo or an empty domain has no defined score and raises.
"""
from __future__ import annotations

import numpy as np

from .sampling import sample_scene

SURF_N_SAMPLE = 150_000
SURF_K = 40
SURF_LP_PASSES = 2
SURF_CURV_RIDGE = 0.12
SURF_TEDGE_STRONG = 8.0
SURF_CAPTURE_R = 0.08
SURF_OFFSET_TOL = 0.012
SURF_OFFSET_S = 0.035


def _lab(a_lin):
    from skimage import color as skcolor

    a = np.clip(a_lin, 0.0, 1.0)
    srgb = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 1e-8, 1), 1 / 2.4) - 0.055)
    return skcolor.rgb2lab(srgb.reshape(-1, 1, 3)).reshape(-1, 3)


def geotex_self(glb: str, n: int = SURF_N_SAMPLE, seed: int = 0) -> float:
    from scipy.spatial import cKDTree

    s = sample_scene(glb, n, seed=seed)
    p = np.asarray(s.p, float)
    nn = np.asarray(s.n, float)
    nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    a = np.asarray(s.a, float)
    if a.std() < 1e-4:
        raise ValueError("mesh-texture agreement is undefined for a uniform albedo")
    c = (p.min(0) + p.max(0)) / 2.0
    rad = np.linalg.norm(p - c, axis=1).max()
    p = (p - c) / max(rad, 1e-9)

    tree = cKDTree(p)
    _, idx = tree.query(p, k=SURF_K)
    nb = idx[:, 1:]

    dots = np.abs(np.einsum("ij,ikj->ik", nn, nn[nb]))
    curv = (1.0 - dots).mean(1)

    a_s = a
    for _ in range(SURF_LP_PASSES):
        a_s = a_s[idx].mean(1)
    lab_s = _lab(a_s)
    dE = np.linalg.norm(lab_s[:, None, :] - lab_s[nb], axis=2)
    tedge = dE.max(1)

    ridge = curv > SURF_CURV_RIDGE
    strong = tedge > SURF_TEDGE_STRONG
    if not strong.any() or not ridge.any():
        raise ValueError("mesh-texture agreement is undefined: no texture edge or no geometry ridge")
    pr, pt = p[ridge], p[strong]
    rtree, ttree = cKDTree(pr), cKDTree(pt)
    d_t, _ = rtree.query(pt)
    d_r, _ = ttree.query(pr)
    in_t = d_t < SURF_CAPTURE_R
    in_r = d_r < SURF_CAPTURE_R
    if int(in_t.sum() + in_r.sum()) == 0:
        raise ValueError("mesh-texture agreement is undefined: no texture edge near a geometry ridge")
    off = np.concatenate([d_t[in_t], d_r[in_r]])
    wt = np.concatenate([tedge[strong][in_t] / max(tedge[strong][in_t].max(), 1e-9)
                         if in_t.any() else np.array([]),
                         curv[ridge][in_r] / max(curv[ridge][in_r].max(), 1e-9)
                         if in_r.any() else np.array([])])
    sim = np.exp(-np.maximum(0.0, off - SURF_OFFSET_TOL) / SURF_OFFSET_S)
    return float((wt * sim).sum() / max(wt.sum(), 1e-9))
