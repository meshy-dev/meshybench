"""Patch tokens of DINOv2-base on tiled renders.

A view rendered at RES * K pixels is cut into K x K tiles of RES pixels, each tile embedded on
its own, and the token grids stitched into one (GRID * K)^2 grid per view. The style floor and
scale and the comparator window are distances in this descriptor's embedding and cells of its
token grid, so they are bound to the model here.
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
PATCH = 14
RES = 518
K = 3
DROP = 1                     # the CLS token precedes the patch tokens
GRID = RES // PATCH
TOL = 3
STYLE_D0 = 0.60
STYLE_SCALE = 0.70
TOKEN_BATCH = 16

_M = {}


def _model():
    if "m" not in _M:
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        # the processor's own default resize would discard the render resolution the grid assumes
        proc.crop_size = {"height": RES, "width": RES}
        proc.size = {"shortest_edge": RES}
        mdl = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).eval().cuda()
        _M["m"] = (proc, mdl)
    return _M["m"]


def tokens(imgs):
    """(N, GRID*GRID, C) L2-normalised patch tokens for uint8 RGB images of RES x RES pixels."""
    import torch
    from PIL import Image

    proc, mdl = _model()
    outs = []
    for i in range(0, len(imgs), TOKEN_BATCH):
        pil = [Image.fromarray(im) for im in imgs[i:i + TOKEN_BATCH]]
        px = proc(images=pil, return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            h = mdl(pixel_values=px).last_hidden_state[:, DROP:, :]
        if h.shape[1] != GRID * GRID:
            raise RuntimeError(f"expected {GRID * GRID} patch tokens, got {h.shape[1]}")
        outs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, 0)


def patch_fg(mask):
    """(GRID*GRID,) bool: any foreground pixel in each PATCH x PATCH cell."""
    m = mask[:GRID * PATCH, :GRID * PATCH].reshape(GRID, PATCH, GRID, PATCH)
    return m.any(axis=(1, 3)).reshape(-1)


def feats_tiled(rendered):
    """{"tokens": (V, (GRID*K)^2, C), "fg": (V, (GRID*K)^2)} for views rendered at RES * K."""
    tiles, keep = [], []
    for col, d in rendered:
        if col.shape[0] != RES * K or col.shape[1] != RES * K:
            raise ValueError(f"view must be {RES * K} x {RES * K} pixels, got {col.shape[:2]}")
        for i in range(K):
            for j in range(K):
                tiles.append(col[i * RES:(i + 1) * RES, j * RES:(j + 1) * RES])
                keep.append((d > 0)[i * RES:(i + 1) * RES, j * RES:(j + 1) * RES])
    toks = tokens(tiles)
    V = len(rendered)
    G = GRID
    big_t = np.zeros((V, G * K, G * K, toks.shape[-1]), dtype=toks.dtype)
    big_f = np.zeros((V, G * K, G * K), dtype=bool)
    n = 0
    for v in range(V):
        for i in range(K):
            for j in range(K):
                big_t[v, i * G:(i + 1) * G, j * G:(j + 1) * G] = toks[n].reshape(G, G, -1)
                big_f[v, i * G:(i + 1) * G, j * G:(j + 1) * G] = patch_fg(keep[n]).reshape(G, G)
                n += 1
    return {"tokens": big_t.reshape(V, (G * K) ** 2, -1), "fg": big_f.reshape(V, (G * K) ** 2)}
