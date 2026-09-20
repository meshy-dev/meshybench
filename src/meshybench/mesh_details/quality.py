"""Detail quality DQ: the fixed-64 perpendicular-tilt entropy estimator of the release.

At base level N on the S x S normal buffer (b = S / N >= 8), every pixel block takes the same
stratified 8 x 8 sub-sample of its b x b raster samples, so the sampling geometry and the
entropy bias are identical across bases. A block is a detail block when all 64 samples are
covered and the circular spread of its normals lies in [2, 20) degrees. From up to n_sample
detail blocks per view (a seeded permutation), each block's normals are projected onto its
tangent plane, the principal direction of the projected normals is found by 2D PCA, and the
perpendicular tilt is histogrammed at Q degrees. The block score is 1 - H / Href with H the
histogram entropy and Href the entropy of the widest spread the block could show (bounded by
its triangle count, the 64 samples and a Gaussian of the block's own spread). Scores are
averaged inside three spread strata, then across the strata that hold at least 30 blocks; a
view's DQ is that mean, and the mesh's DQ is the mean over the eight views.
"""
from __future__ import annotations

import numpy as np

Q = 0.5                                  # tilt histogram bin, degrees
STRATA = [(2, 5), (5, 10), (10, 20)]     # spread strata, degrees
DQ_BASE = 512                            # base level of the published DQ (8192 raster, b = 16)
DQ_PER_VIEW = 1500                       # detail blocks scored per view
MIN_STRATUM = 30                         # blocks a stratum needs to count


def dq64_level(nrm, cover, tid, N: int, S: int, n_sample: int = DQ_PER_VIEW, seed: int = 0):
    """DQ of one view at base N from its S x S normal buffer, coverage mask and triangle ids.
    Returns (dq, per-stratum means); dq is nan when the view has no detail block or no stratum
    reaches MIN_STRATUM blocks."""
    import torch

    dev = nrm.device
    b = S // N
    if b < 8:
        return float("nan"), {}
    idx8 = torch.as_tensor(np.round((np.arange(8) + 0.5) * b / 8 - 0.5).astype(np.int64), device=dev)
    cv = cover.reshape(N, b, N, b).index_select(1, idx8).index_select(3, idx8)
    okb = cv.all(dim=3).all(dim=1)
    sums = torch.stack([(nrm[:, :, k].reshape(N, b, N, b)
                        .index_select(1, idx8).index_select(3, idx8) * cv).sum(dim=(1, 3))
                        for k in range(3)], -1)
    Rbar = sums.norm(dim=-1) / 64.0
    sig = torch.rad2deg(torch.sqrt(torch.clamp(2.0 * (1.0 - Rbar), min=0.0)))
    sig_np = sig.cpu().numpy()
    active = okb.cpu().numpy() & (sig_np > 2.0) & (sig_np < 20.0)
    ii, jj = np.where(active)
    if len(ii) == 0:
        return float("nan"), {}
    take = np.random.default_rng(seed).permutation(len(ii))[:n_sample]
    ii, jj = ii[take], jj[take]
    K = len(ii)
    idx8_np = idx8.cpu().numpy()
    rr = (ii[:, None] * b + idx8_np[None, :])                      # (K,8)
    cc = (jj[:, None] * b + idx8_np[None, :])                      # (K,8)
    R = np.repeat(rr[:, :, None], 8, axis=2).reshape(K, 64)        # rows
    C = np.repeat(cc[:, None, :], 8, axis=1).reshape(K, 64)        # cols
    Rt = torch.as_tensor(R.reshape(-1), device=dev)
    Ct = torch.as_tensor(C.reshape(-1), device=dev)
    n_sel = nrm[Rt, Ct].reshape(K, 64, 3)                          # (K,64,3)
    t_sel = tid[Rt, Ct].reshape(K, 64).cpu().numpy()
    mhat = n_sel.mean(dim=1)
    mhat = mhat / mhat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    up = torch.tensor([0., 0., 1.], device=dev).expand_as(mhat)
    alt = torch.tensor([1., 0., 0.], device=dev).expand_as(mhat)
    base = torch.where(mhat[:, 2:3].abs() > 0.9, alt, up)
    t1 = torch.cross(mhat, base, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    t2 = torch.cross(mhat, t1, dim=-1)
    x = (n_sel * t1.unsqueeze(1)).sum(-1).cpu().numpy()            # (K,64)
    y = (n_sel * t2.unsqueeze(1)).sum(-1).cpu().numpy()
    del n_sel, mhat, t1, t2, base, up, alt, sums, cv
    torch.cuda.empty_cache()
    qrad = np.radians(Q)
    strata_scores = {s: [] for s in STRATA}
    sig_sel = sig_np[ii, jj]
    for k in range(K):
        xc = x[k] - x[k].mean(); yc = y[k] - y[k].mean()
        Sxx, Syy, Sxy = (xc * xc).mean(), (yc * yc).mean(), (xc * yc).mean()
        th = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
        v = -np.sin(th) * xc + np.cos(th) * yc
        counts = np.unique(np.round(v / qrad), return_counts=True)[1]
        p = counts / counts.sum()
        H = -np.sum(p * np.log2(p))
        T = len(np.unique(t_sel[k]))
        st = np.radians(sig_sel[k])
        cap = np.sqrt(2 * np.pi * np.e) * max(st / np.sqrt(2), qrad) / qrad
        Href = np.log2(max(2.0, min(T, 64, cap)))
        sc = float(np.clip(1.0 - H / Href, 0, 1))
        for lo, hi in STRATA:
            if lo <= sig_sel[k] < hi:
                strata_scores[(lo, hi)].append(sc)
    parts = [np.mean(v) for v in strata_scores.values() if len(v) >= MIN_STRATUM]
    detail = {f"{lo}-{hi}": (round(float(np.mean(v)), 4) if len(v) >= MIN_STRATUM else None)
              for (lo, hi), v in strata_scores.items()}
    return (float(np.mean(parts)) if parts else float("nan")), detail
