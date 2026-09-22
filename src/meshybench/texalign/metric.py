"""Texture alignment: Color, Texture Style and Semantic Details of a textured model against its
reference, all in the reference frame of the geometry alignment registration.

The candidate is registered to the reference by meshalign (RMS normalization once per mesh, 24
proper rotations, rigid ICP) and rendered in that frame. For an identical-mesh experiment the
input mesh is registered once and each painter's file is mapped onto the input by the exact
axis-aligned similarity that reproduces the input vertices.

  Color             color.channel_score on N_COLOR area-uniform albedo samples per side
  Texture Style     exp(-max(d - STYLE_D0, 0) / STYLE_SCALE), d the comparator style distance
  Semantic Details  comparator.pattern_score within TOL cells
  overall           unweighted mean of the three

Mesh-Texture Agreement is the separate reference-free metric in geotex and is not averaged in.
"""
from __future__ import annotations

import numpy as np

from . import comparator, descriptor, rendering
from .loading import load_scene_geoms

TOL = descriptor.TOL
STYLE_D0 = descriptor.STYLE_D0
STYLE_SCALE = descriptor.STYLE_SCALE
RES = descriptor.RES * descriptor.K
GRID = descriptor.GRID * descriptor.K
N_COLOR = 40_000
MATCH_TOL = 1e-4          # of the input's RMS radius: a vertex the painter left in place
N_FIT = 20_000
FIT_REFITS = 3
FIT_INLIER = 10 * MATCH_TOL
FIT_MIN_INLIERS = 100


def _geom_register(glb, gt):
    """(A, mu, s, b, residual, vertices) with y = A @ ((x - mu) / s) + b on raw coordinates."""
    from ..meshalign.loading import load_mesh
    from ..meshalign.registration import N_REG, norm_params, register
    from ..meshalign.sampling import dense_sample
    m = load_mesh(glb)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    raw = dense_sample(V, F)
    mu, s = norm_params(raw)
    n = (raw - mu) / s
    pts = n[np.linspace(0, len(n) - 1, N_REG).astype(int)]
    A, b = register(pts, gt.pts, gt.tree)
    resid = float(gt.tree.query(pts @ A.T + b, workers=-1)[0].mean())
    return A, mu, float(s), b, resid, V


def fit_to_input(painter_v, input_v) -> dict:
    """Axis-aligned similarity painter -> input on raw coordinates, x_in = s R x + t, fitted on
    N_FIT painter vertices; match_fraction is the share of them within MATCH_TOL RMS radii of an
    input vertex."""
    from scipy.spatial import cKDTree

    from ..meshalign.registration import ORI
    mu_i, mu_p = input_v.mean(0), painter_v.mean(0)
    rad_i = float(np.sqrt(((input_v - mu_i) ** 2).sum(1).mean()))
    rad_p = float(np.sqrt(((painter_v - mu_p) ** 2).sum(1).mean()))
    tree = cKDTree(input_v)
    rng = np.random.default_rng(0)
    sub = painter_v[rng.choice(len(painter_v), min(N_FIT, len(painter_v)), replace=False)]
    best = None
    for R in ORI:
        s = rad_i / rad_p
        d, _ = tree.query(s * ((sub - mu_p) @ R.T) + mu_i)
        med = float(np.median(d))
        if best is None or med < best[0]:
            best = (med, R, s)
    _, R, s = best
    t = mu_i - s * (R @ mu_p)
    for _ in range(FIT_REFITS):
        d, idx = tree.query(s * (sub @ R.T) + t)
        inl = d <= FIT_INLIER * rad_i
        if inl.sum() < FIT_MIN_INLIERS:
            break
        src, dst = sub[inl] @ R.T, input_v[idx[inl]]
        ms, md = src.mean(0), dst.mean(0)
        s = float(((dst - md) * (src - ms)).sum() / max(((src - ms) ** 2).sum(), 1e-12))
        t = md - s * ms
    d, _ = tree.query(s * (sub @ R.T) + t)
    return {"R": R, "s": float(s), "t": t, "match_fraction": float((d <= MATCH_TOL * rad_i).mean()),
            "median": float(np.median(d) / rad_i), "p99": float(np.quantile(d, 0.99) / rad_i),
            "identity": bool(np.allclose(R, np.eye(3)) and abs(s - 1) < 1e-6
                             and np.abs(t).max() < 1e-6 * max(1.0, rad_i))}


def _as_similarity(A, mu, s, b, fit=None):
    """One similarity y = c R x + t on raw coordinates, the painter -> input fit applied first."""
    if fit is None:
        return A, 1.0 / s, b - A @ mu / s
    Rf, sf, tf = np.asarray(fit["R"]), fit["s"], np.asarray(fit["t"])
    return A @ Rf, sf / s, A @ (tf - mu) / s + b


def _render_embed(glb, R, c, t):
    prepped = rendering.prep_geoms(load_scene_geoms(glb))
    return descriptor.feats_tiled(rendering.render_views(prepped, RES, (np.asarray(R), float(c), np.asarray(t))))


def _style_semantic(ref_feat, feat) -> dict:
    d_mu, d_cov = comparator.style_dist(ref_feat, feat)
    d = d_mu + d_cov
    return {"style": round(float(np.exp(-max(d - STYLE_D0, 0.0) / STYLE_SCALE)), 4),
            "style_raw": round(d, 4), "style_raw_mu": round(float(d_mu), 4), "style_raw_cov": round(float(d_cov), 4),
            "pattern": round(comparator.pattern_score(ref_feat, feat, GRID, TOL), 4)}


def _color(ref, cand_glb, R, c, t) -> dict:
    from .color import SUBW, channel_score, channels
    from .sampling import sample_scene
    cs = sample_scene(cand_glb, N_COLOR, seed=0)
    cand = channels(cs)
    cand_pn = c * (cs.p @ np.asarray(R).T) + t
    cand_lab = ((cand_pn[:, None, :] - ref.color_cent[None, :, :]) ** 2).sum(2).argmin(1)
    out = {sub: round(channel_score(ref.color_ref, cand, ref.color_lab, cand_lab, sub), 4) for sub in SUBW}
    out["color"] = round(sum(SUBW[s] * out[s] for s in SUBW), 4)
    out["uniform_albedo"] = bool(ref.color_ref["uniform"] or cand["uniform"])
    return out


class Reference:
    """A reference prepared once: meshalign ground truth, color regions and descriptor tokens."""

    def __init__(self, glb_path: str):
        from scipy.cluster.vq import kmeans2

        from ..meshalign.metric import prepare_gt
        from .color import K_REGIONS, channels
        from .sampling import sample_scene
        self.glb_path = glb_path
        self.gt = prepare_gt(glb_path)
        rs = sample_scene(glb_path, N_COLOR, seed=0)
        self.color_ref = channels(rs)
        pn = (rs.p - self.gt.mu) / self.gt.s
        rng = np.random.default_rng(0)
        init = pn[rng.choice(len(pn), K_REGIONS, replace=False)]
        self.color_cent, self.color_lab = kmeans2(pn, init, minit="matrix", seed=0)
        self.feat = _render_embed(glb_path, np.eye(3), 1.0 / self.gt.s, -self.gt.mu / self.gt.s)


def prepare_reference(glb_path: str) -> Reference:
    return Reference(glb_path)


class SharedInput:
    """The input mesh of an identical-mesh experiment, registered once to the reference."""

    def __init__(self, input_glb: str, ref: Reference):
        self.glb_path = input_glb
        self.A, self.mu, self.s, self.b, self.residual, self.vertices = _geom_register(input_glb, ref.gt)


def prepare_shared_input(input_glb: str, ref: Reference) -> SharedInput:
    return SharedInput(input_glb, ref)


def score(cand_glb: str, ref: Reference, shared_input: SharedInput | None = None) -> dict:
    """pattern (Semantic Details), style (Texture Style), color and overall, with the raw style
    distance, the color channels, the transform used and the registration residual."""
    out = {}
    if shared_input is None:
        A, mu, s, b, resid, _ = _geom_register(cand_glb, ref.gt)
        R, c, t = _as_similarity(A, mu, s, b)
        out["registration"] = {"residual": round(resid, 4)}
    else:
        from ..meshalign.loading import load_mesh
        fit = fit_to_input(np.asarray(load_mesh(cand_glb).vertices), shared_input.vertices)
        R, c, t = _as_similarity(shared_input.A, shared_input.mu, shared_input.s, shared_input.b, fit)
        out["registration"] = {"residual": round(shared_input.residual, 4),
                               "fit_to_input": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}}
    out["transform"] = {"R": np.asarray(R).tolist(), "c": float(c), "t": np.asarray(t).tolist()}
    out.update(_style_semantic(ref.feat, _render_embed(cand_glb, R, c, t)))
    col = _color(ref, cand_glb, R, c, t)
    out["color"] = col["color"]
    out["color_hue"], out["color_sat"], out["color_value"] = col["hue"], col["sat"], col["value"]
    out["uniform_albedo"] = col["uniform_albedo"]
    out["overall"] = round(float(np.mean([out["pattern"], out["style"], out["color"]])), 4)
    return out
