"""Semantic Details and Texture Style statistics on token grids.

Semantic Details: for each foreground reference cell, the best cosine among the candidate's
foreground tokens within +-tol cells, averaged over the union foreground; a cell only one side
covers counts 0. Texture Style: distance between pooled foreground token clouds, mean L2 distance
plus covariance Frobenius distance normalized by the geometric mean of the two covariance norms.
"""
from __future__ import annotations

import numpy as np


def pattern_score(ref_feat, cand_feat, grid: int, tol: int) -> float:
    import torch

    dev = torch.device("cuda")
    sims = []
    for v in range(len(ref_feat["tokens"])):
        A = torch.as_tensor(ref_feat["tokens"][v], device=dev).reshape(grid, grid, -1)
        B = torch.as_tensor(cand_feat["tokens"][v], device=dev).reshape(grid, grid, -1)
        FA = torch.as_tensor(ref_feat["fg"][v].reshape(grid, grid), device=dev)
        FB = torch.as_tensor(cand_feat["fg"][v].reshape(grid, grid), device=dev)
        best = torch.full((grid, grid), -1.0, device=dev)
        neg = torch.tensor(-1.0, device=dev)
        for di in range(-tol, tol + 1):
            for dj in range(-tol, tol + 1):
                Bs = torch.roll(B, (di, dj), (0, 1))
                FBs = torch.roll(FB, (di, dj), (0, 1))
                best = torch.maximum(best, torch.where(FBs, (A * Bs).sum(-1), neg))
        union = FA | FB
        if not bool(union.any()):
            continue
        c = torch.where(FA & (best > -1), best, torch.zeros_like(best))
        sims.append(float(c[union].mean()))
    if not sims:
        raise ValueError("no foreground in any view")
    return float(np.clip(np.mean(sims), 0, 1))


def style_dist(feat_a, feat_b) -> tuple[float, float]:
    """(mean distance, normalized covariance distance); the style distance is their sum."""
    A = feat_a["tokens"][feat_a["fg"]].astype(np.float64)
    B = feat_b["tokens"][feat_b["fg"]].astype(np.float64)
    if len(A) < 2 or len(B) < 2:
        raise ValueError("fewer than two foreground tokens on one side")
    d_mu = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    Ca = np.cov(A.T)
    Cb = np.cov(B.T)
    d_c = float(np.linalg.norm(Ca - Cb) / np.sqrt(np.linalg.norm(Ca) * np.linalg.norm(Cb) + 1e-9))
    return d_mu, d_c
