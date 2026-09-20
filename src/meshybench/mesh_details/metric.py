"""Mesh details: detail richness and detail quality of a mesh, reference free.

Richness A(N, t): the mesh is rendered as face normals into an S x S orthographic buffer from
eight fixed directions; at screen resolution N each pixel covers a b = S / N block and takes a
stratified 2 x 2 sub-sample of it; the pixel is active when the circular spread of the four
normals (m minus 1 degrees of freedom) exceeds t degrees. A(N, t) is the active fraction over
fully covered pixels, pooled over the eight views. The score of record is A(1024, 2 deg);
A(2048) and A(4096) ship as supporting values.

Quality DQ: the fixed-64 perpendicular-tilt entropy estimator at base 512 on the 8192 raster
(quality.dq64_level), reported in the release as DQ at 4096; the mean over the eight views of
the per-view DQ. Supporting value only; it never enters the score.

Requires torch + nvdiffrast on a CUDA device. Richness is deterministic (fixed views, ordered
stratified sampling); DQ draws its detail blocks from a permutation seeded by the view index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .quality import DQ_BASE, DQ_PER_VIEW, dq64_level
from .render import RASTER, VIEWS, raster_view

THRESHOLD_DEG = 2.0
LEVELS = (1024, 2048, 4096)
SCORE_LEVEL = 1024
MIN_ACTIVE = 50
QUALITY_KEY = "quality_4096"


def level_stats(nrm, cover, N: int, S: int = RASTER, t_deg: float = THRESHOLD_DEG):
    """Per-view counts at screen resolution N from an S x S normal buffer:
    (active pixels, fully covered pixels). Pure tensor math on the buffers' device."""
    import torch

    b = S // N
    if b < 2 or S % N:
        raise ValueError(f"raster {S} must be an even multiple of level {N}")
    if b == 2:
        idx = torch.as_tensor([0, 1], device=nrm.device)
    else:
        idx = torch.as_tensor(np.round((np.arange(2) + 0.5) * b / 2 - 0.5).astype(np.int64),
                              device=nrm.device)
    cv = cover.reshape(N, b, N, b).index_select(1, idx).index_select(3, idx)
    okb = cv.all(dim=3).all(dim=1)
    n4 = nrm.reshape(N, b, N, b, 3).index_select(1, idx).index_select(3, idx)
    s = (n4 * cv.unsqueeze(-1)).sum(dim=(1, 3))
    rbar = s.norm(dim=-1) / 4.0
    sig = torch.rad2deg(torch.sqrt(torch.clamp(2.0 * (1.0 - rbar), min=0.0)))
    # the m - 1 degrees of freedom correction: raw spread compared at t / sqrt(m / (m - 1))
    active = okb & (sig > t_deg / np.sqrt(4.0 / 3.0))
    return int(active.sum()), int(okb.sum())


@dataclass
class DetailResult:
    """The detail metric for one mesh. Fractions in [0, 1]; ``score`` is ``richness_1024``.
    ``status`` is ``ok`` or ``too_small`` (fewer than MIN_ACTIVE active pixels at the score
    level; the score is withheld and the values are kept for diagnosis)."""

    status: str
    score: float | None
    values: dict = field(default_factory=dict)
    input: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "values": dict(self.values),
            "input": dict(self.input),
            "provenance": dict(self.provenance),
        }


def _provenance() -> dict:
    import torch
    import nvdiffrast

    return {
        "raster": RASTER,
        "threshold_deg": THRESHOLD_DEG,
        "views": len(VIEWS),
        "levels": list(LEVELS),
        "score_level": SCORE_LEVEL,
        "dq_base": DQ_BASE,
        "dq_blocks_per_view": DQ_PER_VIEW,
        "torch": torch.__version__,
        "nvdiffrast": getattr(nvdiffrast, "__version__", None),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }


def evaluate(mesh) -> DetailResult:
    """Score one mesh (path or trimesh.Trimesh, glTF frame with Y up) on the current CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the detail metric renders with nvdiffrast and needs a CUDA device")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load(mesh, force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    acc = {N: [0, 0] for N in LEVELS}
    dq_views = []
    for vi, d in enumerate(VIEWS.values()):
        nrm, cover, tid = raster_view(mesh, d, RASTER)
        for N in LEVELS:
            n_active, n_ok = level_stats(nrm, cover, N, RASTER, THRESHOLD_DEG)
            acc[N][0] += n_active
            acc[N][1] += n_ok
        dq, _ = dq64_level(nrm, cover, tid, DQ_BASE, RASTER, n_sample=DQ_PER_VIEW, seed=vi)
        if dq == dq:
            dq_views.append(dq)
        del nrm, cover, tid
        torch.cuda.empty_cache()
    values: dict = {}
    for N in LEVELS:
        n_active, n_ok = acc[N]
        values[f"richness_{N}"] = (n_active / n_ok) if n_ok else None
    values[QUALITY_KEY] = float(np.mean(dq_views)) if dq_views else None
    info = {"faces": int(len(mesh.faces))}
    if acc[SCORE_LEVEL][0] < MIN_ACTIVE:
        return DetailResult(status="too_small", score=None, values=values, input=info,
                            provenance=_provenance())
    return DetailResult(status="ok", score=values[f"richness_{SCORE_LEVEL}"], values=values,
                        input=info, provenance=_provenance())

