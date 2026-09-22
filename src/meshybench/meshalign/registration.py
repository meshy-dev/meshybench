"""Registration: RMS normalization once per mesh, then rigid multi-start ICP.

Scale is fixed once from each point set's RMS radius and ICP is rigid: a per-iteration rescale
lets ICP shrink a bad reconstruction inside the reference and absorb the proportion error the
metric must charge. The coarse search ranks the 24 proper axis-aligned rotations (no
reflections) by symmetric nearest-neighbor distance on N_COARSE points; the best four are each
refined by ICP_ITERS rigid iterations on the N_REG-point registration subsample, and the lowest
residual wins.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

N_REG = 20_000
N_COARSE = 1500
N_REFINE = 4
ICP_ITERS = 30


def norm_params(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Centroid and RMS radius."""
    mu = pts.mean(0)
    s = float(np.sqrt(((pts - mu) ** 2).sum(1).mean()))
    return mu, s


def _proper_rotations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (c, s) in enumerate(zip(perm, signs)):
                M[i, c] = s
            if round(float(np.linalg.det(M))) == 1:
                mats.append(M)
    return mats


ORI = _proper_rotations()


def rigid(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation and translation mapping paired X onto Y."""
    mx = X.mean(0)
    my = Y.mean(0)
    H = (X - mx).T @ (Y - my)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = my - R @ mx
    return R, t


def register(gen: np.ndarray, gt: np.ndarray,
             gt_tree: cKDTree | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) with aligned = gen @ A.T + b for RMS-normalized point sets; A is a rotation."""
    if gt_tree is None:
        gt_tree = cKDTree(gt)
    gen_c = gen[np.linspace(0, len(gen) - 1, min(N_COARSE, len(gen))).astype(int)]
    gt_c = gt[np.linspace(0, len(gt) - 1, min(N_COARSE, len(gt))).astype(int)]
    gt_c_tree = cKDTree(gt_c)
    scored = []
    for i, M in enumerate(ORI):
        g = gen_c @ M.T
        da, _ = gt_c_tree.query(g, workers=-1)
        db, _ = cKDTree(g).query(gt_c, workers=-1)
        scored.append((0.5 * (da.mean() + db.mean()), i))
    scored.sort()
    best = None
    for _, i in scored[:N_REFINE]:
        A = ORI[i].copy()
        b = np.zeros(3)
        cur = gen @ A.T
        for _ in range(ICP_ITERS):
            _, idx = gt_tree.query(cur, workers=-1)
            R, t = rigid(cur, gt[idx])
            cur = cur @ R.T + t
            A = R @ A
            b = b @ R.T + t
        resid = gt_tree.query(cur, workers=-1)[0].mean()
        if best is None or resid < best[0]:
            best = (resid, A, b)
    return best[1], best[2]
