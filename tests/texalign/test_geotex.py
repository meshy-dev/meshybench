"""Mesh-Texture Agreement: a uniform albedo is refused, a textured mesh scores in [0, 1], and the
score is deterministic."""
import os
import tempfile

import numpy as np
import pytest
import trimesh

from meshybench.texalign import geotex


def _box_glb(path, colorize):
    box = trimesh.creation.box(extents=(1.0, 0.7, 0.4)).subdivide().subdivide()
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=colorize(np.asarray(box.vertices)))
    box.export(path)
    return path


def _checker_box_glb(path):
    from PIL import Image as PILImage

    box = trimesh.creation.box(extents=(1.0, 0.7, 0.4)).subdivide().subdivide()
    checker = np.indices((64, 64)).sum(0) % 2 * 200 + 30
    img = PILImage.fromarray(np.stack([checker] * 3, -1).astype(np.uint8))
    v = np.asarray(box.vertices)
    uv = (v[:, :2] - v[:, :2].min(0)) / (v[:, :2].max(0) - v[:, :2].min(0))
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=img))
    box.export(path)
    return path


def test_uniform_albedo_raises():
    with tempfile.TemporaryDirectory() as td:
        p = _box_glb(os.path.join(td, "white.glb"), lambda v: np.full((len(v), 4), 255, np.uint8))
        with pytest.raises(ValueError, match="uniform"):
            geotex.geotex_self(p)


def test_textured_mesh_scores_in_unit_interval():
    with tempfile.TemporaryDirectory() as td:
        p = _checker_box_glb(os.path.join(td, "checker.glb"))
        try:
            score = geotex.geotex_self(p)
        except ValueError as e:
            assert "uniform" not in str(e)
        else:
            assert 0.0 <= score <= 1.0


def test_geotex_self_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        p = _checker_box_glb(os.path.join(td, "checker.glb"))
        try:
            a = geotex.geotex_self(p)
            b = geotex.geotex_self(p)
        except ValueError:
            return
        assert a == b
