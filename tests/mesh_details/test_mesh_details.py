"""Detail richness and quality on procedural meshes. GPU tests skip visibly without a CUDA
device and nvdiffrast."""
import numpy as np
import pytest
import trimesh

from meshybench import mesh_details as detail


def _has_cuda():
    try:
        import torch
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


gpu = pytest.mark.skipif(not _has_cuda(), reason="needs torch + nvdiffrast on a CUDA device")


def test_level_stats_rejects_non_multiple_raster():
    torch = pytest.importorskip("torch")
    nrm = torch.zeros((100, 100, 3))
    cover = torch.ones((100, 100), dtype=torch.bool)
    with pytest.raises(ValueError):
        detail.level_stats(nrm, cover, 64, 100)


def _sphere(bumps: float, subdiv=6):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    if bumps:
        V = np.asarray(m.vertices)
        r = 1.0 + bumps * np.sin(24 * V[:, 0]) * np.sin(24 * V[:, 1]) * np.sin(24 * V[:, 2])
        m.vertices = V * r[:, None]
    return m


def _rough_sphere(sigma, seed=0, subdiv=6):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    V = np.asarray(m.vertices)
    n = np.random.default_rng(seed).normal(0, sigma, size=len(V))
    m.vertices = V * (1.0 + n)[:, None]
    return m


@gpu
def test_smooth_sphere_is_too_small():
    res = detail.evaluate(_sphere(0.0))
    assert res.status == "too_small" and res.score is None


@gpu
def test_richness_orders_bumpy_above_smooth():
    smooth = detail.evaluate(_sphere(0.005))
    bumpy = detail.evaluate(_sphere(0.03))
    assert smooth.status == "ok" and bumpy.status == "ok"
    for N in detail.LEVELS:
        assert bumpy.values[f"richness_{N}"] > smooth.values[f"richness_{N}"]
    assert bumpy.score == bumpy.values["richness_1024"]


@gpu
def test_quality_is_a_fraction():
    for m in (_sphere(0.03), _rough_sphere(0.004)):
        q = detail.evaluate(m).values[detail.QUALITY_KEY]
        assert q is not None and 0.0 <= q <= 1.0


@gpu
def test_evaluate_is_deterministic():
    a = detail.evaluate(_sphere(0.03)).to_dict()
    b = detail.evaluate(_sphere(0.03)).to_dict()
    assert a["score"] == b["score"]
    assert a["values"] == b["values"]


@gpu
def test_result_shape():
    res = detail.evaluate(_sphere(0.03))
    d = res.to_dict()
    assert set(d) == {"status", "score", "values", "input", "provenance"}
    assert d["input"] == {"faces": len(_sphere(0.03).faces)}
    assert set(d["values"]) == {"richness_1024", "richness_2048", "richness_4096", detail.QUALITY_KEY}
    assert d["status"] == "ok" and d["score"] == d["values"]["richness_1024"]
    assert d["provenance"]["dq_base"] == detail.DQ_BASE
