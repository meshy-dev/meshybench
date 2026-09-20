"""Deterministic area-uniform surface sampling. One fixed-seed sample per mesh makes an identical
mesh yield an identical sample, so every signal evaluates to exactly 1.0 against itself."""
from __future__ import annotations

import numpy as np

DEFAULT_SEED = 1234567
DEFAULT_N = 400_000


def dense_sample(verts: np.ndarray, faces: np.ndarray,
                 n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> np.ndarray:
    tri = np.asarray(verts)[np.asarray(faces)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tot = area.sum()
    if tot <= 0:
        raise ValueError("mesh has no surface area")
    rng = np.random.default_rng(seed)
    cum = np.cumsum(area)
    cum /= cum[-1]
    fid = np.searchsorted(cum, rng.random(n))
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    fl = (u + v > 1).ravel()
    u[fl] = 1 - u[fl]
    v[fl] = 1 - v[fl]
    T = tri[fid]
    return T[:, 0] + u * (T[:, 1] - T[:, 0]) + v * (T[:, 2] - T[:, 0])
