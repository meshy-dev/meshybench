"""The three geometry alignment signals, computed on RMS-normalized, registered point samples.

Overall Proportion   IoU of 16^3 occupancy grids, each grid the max-pool of a 64^3 grid stamped by
                     the dense surface sample on a fixed window of +-2.6 RMS radii.
Spatial Distribution 1 - W1 / W1_D0 clipped to [0, 1], W1 the mean over 128 fixed unit directions
                     of the 1D Wasserstein distance between the projected occupied-voxel centers,
                     read from 512 evenly spaced quantiles of each projection.
Surface Details      1 - (F_0.10 - F_0.02) clipped to [0, 1], F_tau the F-score of the two-way
                     nearest-neighbor match at distance tau, on 6,000 query points per side
                     against the other side's full dense sample.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

KF = 64
NQ = 6000
TAU = {"coarse": 0.10, "fine": 0.02}
FIXED_LO = np.array([-2.6, -2.6, -2.6])
FIXED_SPAN = 5.2
PITCH = FIXED_SPAN / KF
W1_DIRS_N, W1_NQ, W1_SEED = 128, 512, 12345
W1_D0 = 0.18   # mean W1 between two different reference objects of the benchmark set

_w1_rng = np.random.default_rng(W1_SEED)
W1_DIRS = _w1_rng.normal(size=(W1_DIRS_N, 3))
W1_DIRS /= np.linalg.norm(W1_DIRS, axis=1, keepdims=True)
_W1_Q = (np.arange(W1_NQ) + 0.5) / W1_NQ


def occ_from_pts(pts: np.ndarray) -> np.ndarray:
    """64^3 occupancy on the fixed window; points outside it land in the boundary cell, so an
    outlying structure cannot move the grid or the registration it is scored in."""
    occ = np.zeros((KF, KF, KF), bool)
    idx = np.clip(np.floor((pts - FIXED_LO) / PITCH).astype(int), 0, KF - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def fscore(pr: float, rc: float) -> float:
    return float(2 * pr * rc / (pr + rc + 1e-9))


def detail_band(surf_coarse: float, surf_fine: float) -> float:
    return float(np.clip(1.0 - (surf_coarse - surf_fine), 0.0, 1.0))


def surface_fscore(gen_dense: np.ndarray, gen_tree: cKDTree,
                   gt_dense: np.ndarray, gt_tree: cKDTree) -> dict[str, float]:
    nq = min(NQ, len(gen_dense), len(gt_dense))
    qg = gen_dense[np.linspace(0, len(gen_dense) - 1, nq).astype(int)]
    qt = gt_dense[np.linspace(0, len(gt_dense) - 1, nq).astype(int)]
    d_g2t, _ = gt_tree.query(qg, workers=-1)
    d_t2g, _ = gen_tree.query(qt, workers=-1)
    return {k: fscore(float((d_g2t < tau).mean()), float((d_t2g < tau).mean()))
            for k, tau in TAU.items()}


def occ_iou_16(og: np.ndarray, ot: np.ndarray) -> float:
    f = KF // 16
    ag = og.reshape(16, f, 16, f, 16, f).any((1, 3, 5))
    at = ot.reshape(16, f, 16, f, 16, f).any((1, 3, 5))
    return float((ag & at).sum() / max((ag | at).sum(), 1))


def _occ_centers(occ: np.ndarray) -> np.ndarray:
    xs, ys, zs = np.where(occ)
    return FIXED_LO + (np.stack([xs, ys, zs], 1) + 0.5) * PITCH


def w1_spatial(og: np.ndarray, ot: np.ndarray) -> dict[str, float]:
    A, B = _occ_centers(og), _occ_centers(ot)
    tot = 0.0
    for u in W1_DIRS:
        pa = np.sort(A @ u)
        pb = np.sort(B @ u)
        qa = np.interp(_W1_Q, (np.arange(len(pa)) + 0.5) / len(pa), pa)
        qb = np.interp(_W1_Q, (np.arange(len(pb)) + 0.5) / len(pb), pb)
        tot += np.abs(qa - qb).mean()
    w1 = tot / len(W1_DIRS)
    return {"w1_raw": float(w1), "w1_spatial": float(np.clip(1.0 - w1 / W1_D0, 0.0, 1.0))}


def pair_signals(a_dense: np.ndarray, a_tree: cKDTree, a_occ: np.ndarray,
                 b_dense: np.ndarray, b_tree: cKDTree, b_occ: np.ndarray) -> dict[str, float]:
    sig = {f"surf_{k}": v for k, v in surface_fscore(a_dense, a_tree, b_dense, b_tree).items()}
    sig["surf_detail"] = detail_band(sig["surf_coarse"], sig["surf_fine"])
    sig["occ_iou_16"] = occ_iou_16(a_occ, b_occ)
    sig.update(w1_spatial(a_occ, b_occ))
    return sig
