"""The portable rasterizer: coverage rules, depth, barycentrics, viewport tiling, interpolation and
mipmapped texture sampling. Runs on CPU. The parity test against nvdiffrast skips visibly without
a CUDA device."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from meshybench import raster  # noqa: E402


def _clip(xy, z=0.0):
    xy = np.asarray(xy, np.float32)
    pos = np.zeros((len(xy), 4), np.float32)
    pos[:, :2] = xy
    pos[:, 2] = z
    pos[:, 3] = 1.0
    return torch.as_tensor(pos)


def _grid_mesh(n, lo=-0.9, hi=0.9, jitter=0.3, seed=0):
    """A jittered n x n grid of quads, each split into two triangles, inside [lo, hi]^2."""
    rng = np.random.default_rng(seed)
    t = np.linspace(lo, hi, n + 1)
    X, Y = np.meshgrid(t, t)
    h = (hi - lo) / n
    inner = (np.arange(n + 1) > 0) & (np.arange(n + 1) < n)
    m = inner[None, :] & inner[:, None]
    X = X + m * rng.uniform(-jitter, jitter, X.shape) * h
    Y = Y + m * rng.uniform(-jitter, jitter, Y.shape) * h
    V = np.stack([X.ravel(), Y.ravel()], 1)
    F = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, i * (n + 1) + j + 1
            c, d = a + n + 1, b + n + 1
            F += [[a, b, d], [a, d, c]] if (i + j) % 2 else [[a, b, c], [b, d, c]]
    return V, np.asarray(F, np.int64)


def _cover(pos, tri, res):
    rast, _ = raster.rasterize_torch(pos, torch.as_tensor(tri), [res, res])
    return rast[0, :, :, 3] > 0, rast


def test_backend_falls_back_to_torch_on_cpu():
    assert raster.backend("cpu") == "torch"


def test_full_screen_quad_has_no_holes_across_viewport_tiles():
    # 4096 is rendered as 2 x 2 viewport tiles of 2048, and the quad is clipped against each tile
    pos = _clip([[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]])
    tri = [[0, 1, 2], [0, 2, 3]]
    for res in (64, 1554, 4096):
        cov, _ = _cover(pos, tri, res)
        assert bool(cov.all()), res


def test_shared_edges_are_owned_by_exactly_one_triangle():
    # vertices on a coarse grid put many shared edges exactly through pixel centers
    V, F = _grid_mesh(12, jitter=0.0)
    V = np.round(V * 64) / 64
    for jitter_seed in (None, 3):
        if jitter_seed is not None:
            V, F = _grid_mesh(12, jitter=0.35, seed=jitter_seed)
        pos = _clip(V)
        even, odd = F[0::2], F[1::2]
        all_cov, _ = _cover(pos, F, 256)
        a, _ = _cover(pos, even, 256)
        b, _ = _cover(pos, odd, 256)
        assert not bool((a & b).any()), "a sample was covered by two triangles"
        assert bool(((a | b) == all_cov).all()), "a sample fell into a crack"


def test_top_left_rule_on_axis_aligned_edges():
    # pixel centers of an 8 x 8 raster sit at odd multiples of 1/8; this square's edges pass
    # exactly through the centers of columns 2 and 5 and rows 2 and 5
    s = [-3 / 8, 3 / 8]
    pos = _clip([[s[0], s[0]], [s[1], s[0]], [s[1], s[1]], [s[0], s[1]]])
    cov, _ = _cover(pos, [[0, 1, 2], [0, 2, 3]], 8)
    ys, xs = np.nonzero(cov.numpy())
    # the bottom edge (dx > 0) and the left edge (dy < 0) are inclusive, top and right exclusive
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (2, 4, 2, 4)


def test_nearest_triangle_wins_and_ties_go_to_the_later_triangle():
    xy = [[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8]]
    near = _clip(xy, z=-0.5)
    far = _clip(xy, z=0.5)
    pos = torch.cat([far, near])
    tri = torch.as_tensor([[0, 1, 2], [3, 4, 5]])
    for order in (tri, tri.flip(0)):
        rast, _ = raster.rasterize_torch(pos, order, [32, 32])
        ids = rast[0, :, :, 3]
        hit = ids > 0
        nearest = 2 if order[0, 0] == 0 else 1
        assert bool((ids[hit] == nearest).all())
    rast, _ = raster.rasterize_torch(torch.cat([near, near]), tri, [32, 32])
    ids = rast[0, :, :, 3]
    assert bool((ids[ids > 0] == 2).all())


def test_barycentrics_reconstruct_pixel_centers_and_derivatives():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.9, 0.9, (3, 2))
    pos = _clip(xy)
    pos[:, 2] = torch.as_tensor([0.1, -0.2, 0.3])
    pos[:, 3] = torch.as_tensor([1.0, 1.5, 2.0])          # perspective
    pos[:, :3] *= pos[:, 3:4]
    tri = torch.as_tensor([[0, 1, 2]])
    res = 128
    rast, db = raster.rasterize_torch(pos, tri, [res, res])
    yy, xx = torch.nonzero(rast[0, :, :, 3] > 0, as_tuple=True)
    assert yy.numel() > 100
    b0, b1 = rast[0, yy, xx, 0], rast[0, yy, xx, 1]
    b2 = 1 - b0 - b1
    # perspective-correct barycentrics interpolate clip coordinates; x / w is the pixel center
    clip = b0[:, None] * pos[0] + b1[:, None] * pos[1] + b2[:, None] * pos[2]
    cx = (2 * xx.float() + 1) / res - 1
    cy = (2 * yy.float() + 1) / res - 1
    assert torch.allclose(clip[:, 0] / clip[:, 3], cx, atol=1e-5)
    assert torch.allclose(clip[:, 1] / clip[:, 3], cy, atol=1e-5)
    assert torch.allclose(rast[0, yy, xx, 2], clip[:, 2] / clip[:, 3], atol=1e-5)
    # du/dX against a finite difference between horizontally adjacent covered pixels
    u = rast[0, :, :, 0]
    both = (rast[0, :, 1:, 3] > 0) & (rast[0, :, :-1, 3] > 0)
    fd = (u[:, 1:] - u[:, :-1])[both]
    an = (0.5 * (db[0, :, 1:, 0] + db[0, :, :-1, 0]))[both]
    assert torch.allclose(fd, an, atol=2e-4)


def test_interpolate_matches_barycentric_sum_and_differentials():
    V, F = _grid_mesh(6)
    pos = _clip(V)
    tri = torch.as_tensor(F)
    rast, db = raster.rasterize_torch(pos, tri, [96, 96])
    attr = torch.as_tensor(np.c_[V * 3.0 + 1.0, V[:, :1] ** 2], dtype=torch.float32)
    out, da = raster.interpolate(attr, rast, tri, rast_db=db, diff_attrs="all")
    hit = rast[0, :, :, 3] > 0
    yy, xx = torch.nonzero(hit, as_tuple=True)
    cx = (2 * xx.float() + 1) / 96 - 1
    # a linear attribute is reproduced exactly, except at the few edge pixels whose center falls
    # just outside the unsnapped triangle, where the barycentrics are clamped as nvdiffrast does
    err = (out[0, yy, xx, 0] - (cx * 3.0 + 1.0)).abs()
    assert float((err < 1e-5).float().mean()) > 0.99 and float(err.max()) < 2e-3
    assert torch.allclose(da[0, yy, xx, 0], torch.full_like(cx, 3.0 * 2 / 96), atol=1e-5)  # d/dX
    assert torch.allclose(da[0, yy, xx, 1], torch.zeros_like(cx), atol=1e-5)               # d/dY
    assert bool((out[0][~hit] == 0).all())


def test_texture_bilinear_wrap_and_mip_selection():
    # a one-texel checkerboard: level 0 alternates 0 and 1, level 1 is uniformly 0.5
    n = 16
    chk = ((np.indices((n, n)).sum(0) % 2).astype(np.float32))[..., None].repeat(3, -1)
    tex = torch.as_tensor(chk)[None]
    # texel centers sample the texel itself; u = 1 + center wraps to the same texel
    uv = torch.as_tensor([[[[0.5 / n, 0.5 / n], [1.5 / n, 0.5 / n], [1 + 1.5 / n, 0.5 / n]]]], dtype=torch.float32)
    zero = torch.zeros((1, 1, 3, 4))
    out = raster.texture(tex, uv, zero, max_mip_level=4)
    assert torch.allclose(out[0, 0, :, 0], torch.tensor([0.0, 1.0, 1.0]))
    # a footprint of two texels per pixel selects mip level 1
    two = torch.zeros((1, 1, 3, 4))
    two[..., 0] = 2.0 / n
    two[..., 3] = 2.0 / n
    out = raster.texture(tex, uv, two, max_mip_level=4)
    assert torch.allclose(out[0, 0, :, 0], torch.full((3,), 0.5))
    # halfway between texel centers on level 0
    mid = torch.as_tensor([[[[1.0 / n, 0.5 / n]]]], dtype=torch.float32)
    out = raster.texture(tex, mid, torch.zeros((1, 1, 1, 4)), max_mip_level=0)
    assert float(out[0, 0, 0, 0]) == pytest.approx(0.5)


def _has_nvdiffrast():
    try:
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_nvdiffrast(), reason="parity check needs nvdiffrast on a CUDA device")
@pytest.mark.parametrize("res", [512, 1554, 4096])
def test_parity_with_nvdiffrast(res):
    import nvdiffrast.torch as dr
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=6, radius=0.9)
    V = np.asarray(m.vertices)
    V = V * (1 + 0.03 * np.sin(20 * V[:, :1]) * np.cos(17 * V[:, 1:2]))
    pos = np.c_[V[:, 0], V[:, 1], -V[:, 2] / 3.0, np.ones(len(V))].astype(np.float32)
    F = np.asarray(m.faces, np.int32)
    p = torch.as_tensor(pos)
    ours, ours_db = raster.rasterize_torch(p, torch.as_tensor(F), [res, res])
    theirs, theirs_db = dr.rasterize(dr.RasterizeCudaContext(), p.cuda()[None], torch.as_tensor(F).cuda(),
                                     resolution=[res, res])
    theirs, theirs_db = theirs.cpu(), theirs_db.cpu()
    assert bool((ours[..., 3] == theirs[..., 3]).all())
    assert torch.allclose(ours, theirs, atol=1e-6)
    assert torch.allclose(ours_db, theirs_db, rtol=1e-5, atol=1e-7)
