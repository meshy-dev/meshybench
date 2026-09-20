"""Geometry alignment: register a generated mesh to its reference and score three dimensions.

  proportion  Overall Proportion    occ_iou_16
  spatial     Spatial Distribution  w1_spatial
  semantic    Surface Details       surf_detail
  overall     unweighted mean of the three

A mesh scored against itself gives 1.0 in every dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .loading import load_mesh
from .registration import N_REG, norm_params, register
from .sampling import dense_sample
from .signals import occ_from_pts, pair_signals

DIMS = {"proportion": "occ_iou_16", "spatial": "w1_spatial", "semantic": "surf_detail"}


@dataclass
class GroundTruth:
    """A reference prepared once: normalised vertices, the registration subsample with its
    KD-tree, the dense sample with its KD-tree, and the occupancy grid."""
    verts: np.ndarray
    faces: np.ndarray
    pts: np.ndarray
    tree: cKDTree
    dense: np.ndarray
    dtree: cKDTree = field(repr=False, default=None)
    occ: np.ndarray = field(repr=False, default=None)
    mu: np.ndarray = field(repr=False, default=None)
    s: float = field(repr=False, default=1.0)


def prepare_gt(mesh: trimesh.Trimesh | str) -> GroundTruth:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    dense_raw = dense_sample(V, F)
    mu, s = norm_params(dense_raw)
    Vn = (V - mu) / s
    dense = (dense_raw - mu) / s
    pts = dense[np.linspace(0, len(dense) - 1, N_REG).astype(int)]
    gt = GroundTruth(verts=Vn, faces=F, pts=pts, tree=cKDTree(pts), dense=dense)
    gt.mu, gt.s = mu, s
    gt.dtree = cKDTree(dense)
    gt.occ = occ_from_pts(dense)
    return gt


def align_gen(gt: GroundTruth, gen_mesh: trimesh.Trimesh) -> np.ndarray:
    """Dense sample of the generated mesh in the reference's normalised frame."""
    Vg = np.asarray(gen_mesh.vertices)
    Fg = np.asarray(gen_mesh.faces)
    gen_raw = dense_sample(Vg, Fg)
    mu, s = norm_params(gen_raw)
    gen_n = (gen_raw - mu) / s
    reg_pts = gen_n[np.linspace(0, len(gen_n) - 1, N_REG).astype(int)]
    A, b = register(reg_pts, gt.pts, gt.tree)
    Vgn = ((Vg - mu) / s) @ A.T + b
    return dense_sample(Vgn, Fg)


def all_signals(gt: GroundTruth, gen_mesh: trimesh.Trimesh) -> dict[str, float]:
    gd = align_gen(gt, gen_mesh)
    og = occ_from_pts(gd)
    return pair_signals(gd, cKDTree(gd), og, gt.dense, gt.dtree, gt.occ)


def score(gen: trimesh.Trimesh | str, gt: trimesh.Trimesh | str | GroundTruth) -> dict:
    """{"proportion", "spatial", "semantic", "overall", "signals"}, all in [0, 1]."""
    if not isinstance(gt, GroundTruth):
        gt = prepare_gt(gt if not isinstance(gt, str) else load_mesh(gt))
    if isinstance(gen, str):
        gen = load_mesh(gen)
    sig = all_signals(gt, gen)
    dims = {d: float(sig[k]) for d, k in DIMS.items()}
    dims["overall"] = float(np.mean(list(dims.values())))
    dims["signals"] = sig
    return dims
