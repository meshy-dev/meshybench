"""Reflexivity, similarity invariance, proportion sensitivity, tessellation invariance."""
import numpy as np
import pytest
import trimesh

from meshybench import meshalign


def _asset():
    return trimesh.util.concatenate([
        trimesh.creation.box(extents=(1.0, 0.4, 0.6)),
        trimesh.creation.icosphere(subdivisions=3, radius=0.35).apply_translation((0.8, 0.1, 0.05)),
        trimesh.creation.cylinder(radius=0.08, height=0.9).apply_translation((-0.5, 0.3, 0)),
    ])


@pytest.fixture(scope="module")
def asset():
    return _asset()


@pytest.fixture(scope="module")
def gt(asset):
    return meshalign.prepare_gt(asset)


def test_reflexivity_exact(asset, gt):
    res = meshalign.score(asset, gt)
    for d in ("proportion", "spatial", "semantic"):
        assert res[d] == pytest.approx(1.0, abs=1e-6), d


def test_similarity_invariance(asset, gt):
    m = asset.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    m.apply_scale(2.7)
    m.apply_translation((5.0, -3.0, 1.5))
    res = meshalign.score(m, gt)
    for d in ("proportion", "spatial", "semantic"):
        assert res[d] > 0.999, (d, res[d])


def test_proportion_error_survives_registration(asset, gt):
    m = asset.copy()
    m.apply_transform(np.diag([1.6, 1.0, 1.0, 1.0]))
    res = meshalign.score(m, gt)
    assert res["proportion"] < 0.95


def test_tessellation_invariance(asset, gt):
    m = asset.copy().subdivide()
    res = meshalign.score(m, gt)
    for d in ("proportion", "spatial", "semantic"):
        assert res[d] > 0.98, (d, res[d])


def test_degenerate_geometry_raises(tmp_path):
    tri = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=[[0, 1, 2]], process=False)
    p = tmp_path / "degenerate.glb"
    tri.export(p)
    with pytest.raises(ValueError):
        meshalign.load_mesh(str(p))
