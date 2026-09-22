"""Comparator and color statistics. The Semantic Details comparator runs on CUDA and its tests
skip visibly without one."""
import numpy as np
import pytest

from meshybench.texalign.comparator import pattern_score, style_dist


def _has_cuda():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


gpu = pytest.mark.skipif(not _has_cuda(), reason="needs torch on a CUDA device")


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@gpu
def test_pattern_identity_is_one():
    rng = np.random.default_rng(0)
    g = 8
    f = {"tokens": _unit(rng.normal(size=(1, g * g, 16))), "fg": np.ones((1, g * g), bool)}
    assert pattern_score(f, f, g, 1) == pytest.approx(1.0, abs=1e-6)


@gpu
def test_pattern_window_recovers_shifted_texture():
    rng = np.random.default_rng(1)
    g = 8
    A = _unit(rng.normal(size=(g, g, 16)))
    B = np.roll(A, 1, axis=1)
    fa = {"tokens": A.reshape(1, g * g, 16), "fg": np.ones((1, g * g), bool)}
    fb = {"tokens": B.reshape(1, g * g, 16), "fg": np.ones((1, g * g), bool)}
    assert pattern_score(fa, fb, g, 0) < 0.5
    assert pattern_score(fa, fb, g, 1) == pytest.approx(1.0, abs=1e-6)


@gpu
def test_pattern_union_charges_uncovered_reference_cells():
    rng = np.random.default_rng(2)
    g = 4
    toks = _unit(rng.normal(size=(1, g * g, 8)))
    fg_full = np.ones((1, g * g), bool)
    fg_half = fg_full.copy()
    fg_half[0, : g * g // 2] = False
    s = pattern_score({"tokens": toks, "fg": fg_full}, {"tokens": toks, "fg": fg_half}, g, 0)
    assert s == pytest.approx(0.75, abs=0.26)
    assert s < 1.0


def test_style_dist_zero_for_identical_and_positive_otherwise():
    rng = np.random.default_rng(3)
    A = _unit(rng.normal(size=(1, 64, 16)))
    B = _unit(rng.normal(size=(1, 64, 16)))
    fg = np.ones((1, 64), bool)
    fa, fb = {"tokens": A, "fg": fg}, {"tokens": B, "fg": fg}
    d_mu, d_cov = style_dist(fa, fa)
    assert d_mu == pytest.approx(0.0, abs=1e-9)
    assert d_cov == pytest.approx(0.0, abs=1e-9)
    d_mu2, d_cov2 = style_dist(fa, fb)
    assert d_mu2 > 0 and d_cov2 > 0


def test_style_mapping_constants():
    from meshybench.texalign.metric import STYLE_D0, STYLE_SCALE

    style = lambda d: np.exp(-max(d - STYLE_D0, 0.0) / STYLE_SCALE)
    assert (STYLE_D0, STYLE_SCALE) == (0.60, 0.70)
    assert style(0.0) == pytest.approx(1.0)
    assert style(STYLE_D0) == pytest.approx(1.0)
    assert style(0.8) > style(1.2) > style(1.8)


def test_color_circular_statistics():
    from meshybench.texalign.color import _circ_dist, _circ_mean, _circ_std

    w = np.ones(4)
    m = _circ_mean(np.array([350.0, 10.0, 355.0, 5.0]), w)
    assert _circ_dist(m, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert _circ_dist(350.0, 10.0) == pytest.approx(20.0)
    assert _circ_std(np.full(4, 123.0), w) == pytest.approx(0.0, abs=1e-3)
