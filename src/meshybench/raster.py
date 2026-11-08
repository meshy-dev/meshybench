"""Rasterization backends: nvdiffrast on CUDA, and a portable torch rasterizer for every other device.

``rasterize(pos, tri, resolution)`` returns what ``nvdiffrast.torch.rasterize`` returns for a single
image: ``(rast, rast_db)`` with ``rast`` of shape (1, H, W, 4) holding (u, v, z/w, triangle id + 1)
and ``rast_db`` holding the screen-space barycentric derivatives (du/dX, du/dY, dv/dX, dv/dY).
Row 0 is the bottom of the image (y = -1 in clip space), as in nvdiffrast.

The portable backend reimplements nvdiffrast's CUDA rasterizer (cudaraster) step by step so that
scores agree within the benchmark tolerance:

- resolutions above 2048 are split into the same viewport tiles, and each tile's clip-space
  transform is applied in float32 as cudaraster applies it;
- vertices snap to 1/16 pixel with round-half-to-even (snapTriangle); triangles that leave a tile's
  frustum are clipped and fan-triangulated as clipTriangleWithFrustum does;
- coverage is decided at pixel centers with exact integer edge functions on the snapped vertices
  and cudaraster's fill convention for samples on an edge, so a shared edge belongs to exactly one
  triangle;
- depth is cudaraster's 32-bit fixed-point plane equation (setupPleq); the nearest fragment wins
  and a depth tie goes to the later triangle, as its ROP resolves it;
- the output barycentrics, z/w and pixel differentials use the float32 formulas of nvdiffrast's
  forward shader kernel on the unsnapped clip-space positions.

It needs only torch and runs on CPU (or any torch device). It is slower than nvdiffrast but bounded
in memory: candidate pixels are generated per bounding-box tile in chunks.
"""
from __future__ import annotations

import os

import numpy as np

SUBPIXEL_LOG2 = 4
SUBPIXEL = 1 << SUBPIXEL_LOG2
TILE = 32                     # candidate block edge for a bounding box, pixels
CHUNK = 1 << 22               # candidate samples evaluated per step
_LERP_ERROR3 = 2200 << 3      # cudaraster CR_LERP_ERROR(3)
DEPTH_MIN = _LERP_ERROR3
DEPTH_MAX = (1 << 32) - 1 - _LERP_ERROR3
_TID_BITS = 31
_TID_MAX = (1 << _TID_BITS) - 1
MAX_VIEWPORT = 2048            # cudaraster CR_MAXVIEWPORT_SIZE
_EMPTY = (1 << 63) - 1


def backend(device=None) -> str:
    """``nvdiffrast`` when it is importable and the device is CUDA, else ``torch``.
    ``MESHYBENCH_RASTER=torch`` forces the portable backend everywhere."""
    import torch

    if os.environ.get("MESHYBENCH_RASTER", "").lower() == "torch":
        return "torch"
    dev = torch.device(device) if device is not None else default_device()
    if dev.type != "cuda":
        return "torch"
    try:
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        return "torch"
    return "nvdiffrast"


def default_device():
    """CUDA when available, else CPU."""
    import torch

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _edge_owns(dx, dy):
    """cudaraster's fill convention (cover8x8_exact_noLUT): for a counterclockwise triangle a sample
    exactly on an edge is covered when the edge has dy < 0, or dy == 0 and dx > 0. An interior edge
    is traversed in opposite directions by its two triangles, so exactly one of them owns it."""
    return (dy < 0) | ((dy == 0) & (dx > 0))


def _viewport_tiles(n: int):
    """nvdiffrast renders resolutions above 2048 in viewport tiles (torch_rasterize.cpp)."""
    count = (n + MAX_VIEWPORT - 1) // MAX_VIEWPORT
    size = ((n + count - 1) // count + 7) & -8
    return [(off, min(n - off, size)) for off in range(0, n, size)]


def _f32(x):
    import torch

    return torch.as_tensor(x, dtype=torch.float32)


def _fma32(a, b, c):
    """float32 fused multiply-add: the product and sum are exact in float64, rounded once."""
    import torch

    return (a.to(torch.float64) * b.to(torch.float64) + c.to(torch.float64)).to(torch.float32)


def _clip_with_plane(bary_in, v0, v1, v2):
    """clipPolygonWithPlane on one polygon given as a list of (b0, b1) float32 pairs."""
    out = []
    n = len(bary_in)
    if n < 3:
        return out
    f = np.float32
    a = bary_in[-1]
    av = _fma_s(v2, a[1], _fma_s(v1, a[0], v0))
    for bb in bary_in:
        bv = _fma_s(v2, bb[1], _fma_s(v1, bb[0], v0))
        if f(av * bv) < 0:
            bc = f(av / f(av - bv))
            ac = f(f(1.0) - bc)
            out.append((_fma_s(bb[0], bc, f(a[0] * ac)), _fma_s(bb[1], bc, f(a[1] * ac))))
        if bv >= 0:
            out.append(bb)
        a, av = bb, bv
    return out


def _fma_s(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def _clip_triangle(v0, v1, v2):
    """clipTriangleWithFrustum for one triangle (three float32 4-vectors); returns the clipped
    polygon's vertices reconstructed the way triangleSetupImpl does."""
    f = np.float32
    d1 = (v1 - v0).astype(np.float32)
    d2 = (v2 - v0).astype(np.float32)
    poly = [(f(0), f(0)), (f(1), f(0)), (f(0), f(1))]
    for k in range(3):
        if v0[3] < abs(v0[k]) or v1[3] < abs(v1[k]) or v2[3] < abs(v2[k]):
            poly = _clip_with_plane(poly, f(v0[3] + v0[k]), f(d1[3] + d1[k]), f(d2[3] + d2[k]))
            poly = _clip_with_plane(poly, f(v0[3] - v0[k]), f(d1[3] - d1[k]), f(d2[3] - d2[k]))
    return [np.array([_fma_s(d2[c], b1, _fma_s(d1[c], b0, v0[c])) for c in range(4)], np.float32)
            for b0, b1 in poly]


def _setup_and_cover(key, tids, V0, V1, V2, vpw, vph, offx, offy, W):
    """Triangle setup (snap, cull, orient, depth plane equation) and fine rasterization of the
    (sub)triangles V0, V1, V2 (float32 (n, 4) in tile clip space) into ``key``."""
    import torch

    dev = key.device
    if tids.numel() == 0:
        return
    one = _f32(1.0)
    scx, scy = _f32(float(vpw << (SUBPIXEL_LOG2 - 1))), _f32(float(vph << (SUBPIXEL_LOG2 - 1)))

    def snap(v):
        rw = one / v[:, 3]
        x = torch.round((v[:, 0] * rw) * scx).clamp(-2**31, 2**31 - 1).to(torch.int64)
        y = torch.round((v[:, 1] * rw) * scy).clamp(-2**31, 2**31 - 1).to(torch.int64)
        return x, y, rw

    x0, y0, r0 = snap(V0)
    x1, y1, r1 = snap(V1)
    x2, y2, r2 = snap(V2)
    z0, z1, z2 = V0[:, 2], V1[:, 2], V2[:, 2]
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    bx = (vpw << (SUBPIXEL_LOG2 - 1)) - (SUBPIXEL >> 1)
    by = (vph << (SUBPIXEL_LOG2 - 1)) - (SUBPIXEL >> 1)
    flip = area < 0
    x1, x2 = torch.where(flip, x2, x1), torch.where(flip, x1, x2)
    y1, y2 = torch.where(flip, y2, y1), torch.where(flip, y1, y2)
    z1, z2 = torch.where(flip, z2, z1), torch.where(flip, z1, z2)
    r1, r2 = torch.where(flip, r2, r1), torch.where(flip, r1, r2)
    area = area.abs()
    lox = torch.div(torch.minimum(torch.minimum(x0, x1), x2) + bx + SUBPIXEL - 1, SUBPIXEL, rounding_mode="floor")
    loy = torch.div(torch.minimum(torch.minimum(y0, y1), y2) + by + SUBPIXEL - 1, SUBPIXEL, rounding_mode="floor")
    hix = torch.div(torch.maximum(torch.maximum(x0, x1), x2) + bx, SUBPIXEL, rounding_mode="floor")
    hiy = torch.div(torch.maximum(torch.maximum(y0, y1), y2) + by, SUBPIXEL, rounding_mode="floor")
    lox, loy = lox.clamp(min=0), loy.clamp(min=0)
    hix, hiy = hix.clamp(max=vpw - 1), hiy.clamp(max=vph - 1)
    keep = (area != 0) & (lox <= hix) & (loy <= hiy)

    # depth plane equation, setupTriangle + setupPleq, in 32-bit integer arithmetic
    zcoef = _f32(float(np.float32(DEPTH_MAX - DEPTH_MIN)) * 0.5)
    zbias = _f32(float(np.float32(DEPTH_MAX + DEPTH_MIN)) * 0.5)
    zv = [_fma32(z * zcoef, r, zbias) for z, r in ((z0, r0), (z1, r1), (z2, r2))]
    mx = torch.maximum(torch.maximum(zv[0], zv[1]), zv[2])
    sh = ((mx.view(torch.int32).to(torch.int64) >> 23) - (127 + 22)).clamp(0, 8)
    zu = [z.to(torch.float64).clamp(0, 2.0**32 - 1).floor().to(torch.int64) for z in zv]
    t0 = zu[0] >> sh
    t1 = (zu[1] >> sh) - t0
    t2 = (zu[2] >> sh) - t0
    arcp = one / area.clamp(min=1).to(torch.float32)
    abits = arcp.view(torch.int32).to(torch.int64)
    rcp_mant = (abits & 0x007FFFFF) | 0x00800000
    rcp_shift = (23 + 127) - (abits >> 23)
    d1x, d1y, d2x, d2y = x1 - x0, y1 - y0, x2 - x0, y2 - y0
    xc = (t1 * d2y - t2 * d1y) * rcp_mant
    yc = (t2 * d1x - t1 * d2x) * rcp_mant
    M32 = 0xFFFFFFFF
    px_ = (xc >> (rcp_shift - (sh + SUBPIXEL_LOG2))) & M32
    py_ = (yc >> (rcp_shift - (sh + SUBPIXEL_LOG2))) & M32
    zx0 = x0 + (vpw << (SUBPIXEL_LOG2 - 1)) - (1 << (SUBPIXEL_LOG2 - 1))
    zy0 = y0 + (vph << (SUBPIXEL_LOG2 - 1)) - (1 << (SUBPIXEL_LOG2 - 1))
    zero = torch.zeros_like(d1x)
    lo_x = torch.minimum(torch.minimum(d1x, d2x), zero)
    hi_x = torch.maximum(torch.maximum(d1x, d2x), zero)
    lo_y = torch.minimum(torch.minimum(d1y, d2y), zero)
    hi_y = torch.maximum(torch.maximum(d1y, d2y), zero)
    cx = (zx0 * 2 + lo_x + hi_x) >> (SUBPIXEL_LOG2 + 1)
    cy = (zy0 * 2 + lo_y + hi_y) >> (SUBPIXEL_LOG2 + 1)
    vcx = zx0 - (cx << SUBPIXEL_LOG2)
    vcy = zy0 - (cy << SUBPIXEL_LOG2)
    pz_ = (t0 << sh) & M32
    pz_ = (pz_ - ((((xc >> 13) * vcx + (yc >> 13) * vcy) >> (rcp_shift - (sh + 13))) & M32)) & M32
    pz_ = (pz_ - px_ * cx - py_ * cy) & M32

    e01 = _edge_owns(x1 - x0, y1 - y0)
    e12 = _edge_owns(x2 - x1, y2 - y1)
    e20 = _edge_owns(x0 - x2, y0 - y2)

    # split bounding boxes into TILE x TILE candidate blocks, grouped by window size
    idx = torch.nonzero(keep).squeeze(1)
    ntx = torch.div(hix[idx] - lox[idx], TILE, rounding_mode="floor") + 1
    nty = torch.div(hiy[idx] - loy[idx], TILE, rounding_mode="floor") + 1
    nt = ntx * nty
    bt = torch.repeat_interleave(idx, nt)
    first = torch.cumsum(nt, 0) - nt
    local = torch.arange(int(nt.sum()), device=dev) - torch.repeat_interleave(first, nt)
    ntx_b = torch.repeat_interleave(ntx, nt)
    bxs = lox[bt] + (local % ntx_b) * TILE
    bys = loy[bt] + torch.div(local, ntx_b, rounding_mode="floor") * TILE
    ext = torch.maximum((hix[bt] - bxs).clamp(max=TILE - 1), (hiy[bt] - bys).clamp(max=TILE - 1)) + 1
    order = torch.argsort(ext, stable=True)
    bt, bxs, bys, ext = bt[order], bxs[order], bys[order], ext[order]
    sizes, counts = torch.unique_consecutive(ext, return_counts=True)
    start = 0
    for e, c in zip(sizes.tolist(), counts.tolist()):
        step = max(CHUNK // (e * e), 1)
        for a in range(start, start + c, step):
            b = min(a + step, start + c)
            t = bt[a:b]
            ox_ = torch.arange(e, device=dev)
            kx = (bxs[a:b, None, None] + ox_[None, None, :]).expand(-1, e, e)
            ky = (bys[a:b, None, None] + ox_[None, :, None]).expand(-1, e, e)
            cov = (kx <= hix[t, None, None]) & (ky <= hiy[t, None, None])
            qx = kx * SUBPIXEL - bx
            qy = ky * SUBPIXEL - by
            X0, Y0, X1, Y1 = x0[t, None, None], y0[t, None, None], x1[t, None, None], y1[t, None, None]
            X2, Y2 = x2[t, None, None], y2[t, None, None]
            w = (X1 - X0) * (qy - Y0) - (Y1 - Y0) * (qx - X0)
            cov &= (w > 0) | ((w == 0) & e01[t, None, None])
            w = (X2 - X1) * (qy - Y1) - (Y2 - Y1) * (qx - X1)
            cov &= (w > 0) | ((w == 0) & e12[t, None, None])
            w = (X0 - X2) * (qy - Y2) - (Y0 - Y2) * (qx - X2)
            cov &= (w > 0) | ((w == 0) & e20[t, None, None])
            sel = torch.nonzero(cov, as_tuple=True)
            if sel[0].numel():
                ts = t[sel[0]]
                kxs, kys = kx[sel], ky[sel]
                depth = (px_[ts] * kxs + py_[ts] * kys + pz_[ts]) & M32
                k = (depth << _TID_BITS) | (_TID_MAX - tids[ts])
                key.scatter_reduce_(0, (offy + kys) * W + (offx + kxs), k, reduce="amin")
        start += c


def rasterize_torch(pos, tri, resolution):
    """Portable rasterizer with nvdiffrast's output contract. ``pos`` is (1, V, 4) or (V, 4) clip
    space, ``tri`` is (F, 3) int; ``resolution`` is [H, W]."""
    import torch

    if pos.dim() == 3:
        if pos.shape[0] != 1:
            raise ValueError("the portable rasterizer takes a single image")
        pos = pos[0]
    dev = pos.device
    H, W = int(resolution[0]), int(resolution[1])
    tri = tri.to(device=dev, dtype=torch.int64)
    P = pos.to(torch.float32).contiguous()
    tid_all = torch.arange(tri.shape[0], device=dev, dtype=torch.int64)
    key = torch.full((H * W,), _EMPTY, dtype=torch.int64, device=dev)
    for offy, vph in _viewport_tiles(H):
        for offx, vpw in _viewport_tiles(W):
            xs = _f32(float(W)) / _f32(float(vpw))
            ys = _f32(float(H)) / _f32(float(vph))
            xo = _f32(float(W - vpw - 2 * offx)) / _f32(float(vpw))
            yo = _f32(float(H - vph - 2 * offy)) / _f32(float(vph))
            T = P.clone()
            T[:, 0] = _fma32(P[:, 0], xs, P[:, 3] * xo)
            T[:, 1] = _fma32(P[:, 1], ys, P[:, 3] * yo)
            V0, V1, V2 = T[tri[:, 0]], T[tri[:, 1]], T[tri[:, 2]]
            w0, w1, w2 = V0[:, 3], V1[:, 3], V2[:, 3]
            cull = torch.zeros(tri.shape[0], dtype=torch.bool, device=dev)
            for c in range(3):
                cull |= (w0 < V0[:, c]) & (w1 < V1[:, c]) & (w2 < V2[:, c])
                cull |= (w0 < -V0[:, c]) & (w1 < -V1[:, c]) & (w2 < -V2[:, c])
            inside = ((w0 >= V0[:, :3].abs().amax(1)) & (w1 >= V1[:, :3].abs().amax(1))
                      & (w2 >= V2[:, :3].abs().amax(1)))
            fast = inside & ~cull
            _setup_and_cover(key, tid_all[fast], V0[fast], V1[fast], V2[fast], vpw, vph, offx, offy, W)
            slow = torch.nonzero(~inside & ~cull).squeeze(1).tolist()
            if slow:
                A, B, C, S = [], [], [], []
                n0, n1, n2 = V0.cpu().numpy(), V1.cpu().numpy(), V2.cpu().numpy()
                for i in slow:
                    poly = _clip_triangle(n0[i], n1[i], n2[i])
                    for j in range(2, len(poly)):
                        A.append(poly[0])
                        B.append(poly[j - 1])
                        C.append(poly[j])
                        S.append(i)
                if S:
                    _setup_and_cover(key, torch.as_tensor(S, dtype=torch.int64, device=dev),
                                     torch.as_tensor(np.array(A), device=dev), torch.as_tensor(np.array(B), device=dev),
                                     torch.as_tensor(np.array(C), device=dev), vpw, vph, offx, offy, W)
    hit = key != _EMPTY
    tri_id = torch.where(hit, _TID_MAX - (key & _TID_MAX), torch.full_like(key, -1))
    return _shade(P, tri, tri_id.reshape(H, W), H, W)


def _shade(P, tri, tri_id, H, W):
    """nvdiffrast's RasterizeCudaFwdShaderKernel in float32 on the covered pixels."""
    import torch

    dev = P.device
    rast = torch.zeros((H, W, 4), dtype=torch.float32, device=dev)
    db = torch.zeros((H, W, 4), dtype=torch.float32, device=dev)
    yy, xx = torch.nonzero(tri_id >= 0, as_tuple=True)
    if yy.numel() == 0:
        return rast.unsqueeze(0), db.unsqueeze(0)
    t = tri_id[yy, xx]
    v = tri[t]
    p0, p1, p2 = P[v[:, 0]], P[v[:, 1]], P[v[:, 2]]
    xs = torch.tensor(2.0 / W, dtype=torch.float32)
    xo = torch.tensor(1.0 / W - 1.0, dtype=torch.float32)
    ys = torch.tensor(2.0 / H, dtype=torch.float32)
    yo = torch.tensor(1.0 / H - 1.0, dtype=torch.float32)
    fx = xs * xx.to(torch.float32) + xo
    fy = ys * yy.to(torch.float32) + yo
    p0x, p0y = p0[:, 0] - fx * p0[:, 3], p0[:, 1] - fy * p0[:, 3]
    p1x, p1y = p1[:, 0] - fx * p1[:, 3], p1[:, 1] - fy * p1[:, 3]
    p2x, p2y = p2[:, 0] - fx * p2[:, 3], p2[:, 1] - fy * p2[:, 3]
    a0 = p1x * p2y - p1y * p2x
    a1 = p2x * p0y - p2y * p0x
    a2 = p0x * p1y - p0y * p1x
    iw = 1.0 / (a0 + a1 + a2)
    b0 = a0 * iw
    b1 = a1 * iw
    z = p0[:, 2] * a0 + p1[:, 2] * a1 + p2[:, 2] * a2
    wv = p0[:, 3] * a0 + p1[:, 3] * a1 + p2[:, 3] * a2
    zw = (z / wv).clamp(-1.0, 1.0)
    b0c = b0.clamp(0.0, 1.0)
    b1c = b1.clamp(0.0, 1.0)
    bs = 1.0 / torch.clamp(b0c + b1c, min=1.0)
    b0c, b1c = b0c * bs, b1c * bs
    rast[yy, xx] = torch.stack([b0c, b1c, zw, (t + 1).to(torch.float32)], -1)
    dfxdx, dfydy = xs * iw, ys * iw
    da0dx = p2[:, 1] * p1[:, 3] - p1[:, 1] * p2[:, 3]
    da0dy = p1[:, 0] * p2[:, 3] - p2[:, 0] * p1[:, 3]
    da1dx = p0[:, 1] * p2[:, 3] - p2[:, 1] * p0[:, 3]
    da1dy = p2[:, 0] * p0[:, 3] - p0[:, 0] * p2[:, 3]
    da2dx = p1[:, 1] * p0[:, 3] - p0[:, 1] * p1[:, 3]
    da2dy = p0[:, 0] * p1[:, 3] - p1[:, 0] * p0[:, 3]
    datdx = da0dx + da1dx + da2dx
    datdy = da0dy + da1dy + da2dy
    db[yy, xx] = torch.stack([dfxdx * (b0c * datdx - da0dx), dfydy * (b0c * datdy - da0dy),
                              dfxdx * (b1c * datdx - da1dx), dfydy * (b1c * datdy - da1dy)], -1)
    return rast.unsqueeze(0), db.unsqueeze(0)


_NVDR: dict = {}


def rasterize(pos, tri, resolution):
    """(rast, rast_db) for one image with the backend chosen for ``pos.device``."""
    import torch

    if backend(pos.device) == "nvdiffrast":
        import nvdiffrast.torch as dr

        if "ctx" not in _NVDR:
            _NVDR["ctx"] = dr.RasterizeCudaContext()
        return dr.rasterize(_NVDR["ctx"], pos, tri.to(dtype=torch.int32), resolution=list(resolution))
    return rasterize_torch(pos, tri, resolution)


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """nvdiffrast.torch.interpolate for one image: ``attr`` (V, A) or (1, V, A). Returns (out, da)
    with ``da`` holding (d/dX, d/dY) per attribute when ``diff_attrs == "all"``."""
    import torch

    if rast.device.type == "cuda" and backend(rast.device) == "nvdiffrast":
        import nvdiffrast.torch as dr

        a = attr if attr.dim() == 3 else attr.unsqueeze(0)
        return dr.interpolate(a.contiguous(), rast, tri.to(dtype=torch.int32), rast_db=rast_db,
                              diff_attrs=diff_attrs)
    a = (attr[0] if attr.dim() == 3 else attr).to(torch.float32)
    r = rast[0]
    H, W = r.shape[:2]
    n_attr = a.shape[1]
    out = torch.zeros((H, W, n_attr), dtype=torch.float32, device=a.device)
    da = None
    if diff_attrs is not None:
        if diff_attrs != "all":
            raise NotImplementedError("the portable interpolate supports diff_attrs='all' only")
        da = torch.zeros((H, W, 2 * n_attr), dtype=torch.float32, device=a.device)
    yy, xx = torch.nonzero(r[:, :, 3] > 0, as_tuple=True)
    if yy.numel():
        t = r[yy, xx, 3].to(torch.int64) - 1
        v = tri.to(torch.int64)[t]
        a0, a1, a2 = a[v[:, 0]], a[v[:, 1]], a[v[:, 2]]
        b0, b1 = r[yy, xx, 0:1], r[yy, xx, 1:2]
        b2 = 1.0 - b0 - b1
        out[yy, xx] = b0 * a0 + b1 * a1 + b2 * a2
        if da is not None:
            db = rast_db[0][yy, xx]
            dsdu, dsdv = a0 - a2, a1 - a2
            dx = db[:, 0:1] * dsdu + db[:, 2:3] * dsdv
            dy = db[:, 1:2] * dsdu + db[:, 3:4] * dsdv
            da[yy, xx] = torch.stack([dx, dy], -1).reshape(-1, 2 * n_attr)
    return out.unsqueeze(0), (da.unsqueeze(0) if da is not None else None)


def _mip_pyramid(tex, max_level):
    """Box-filtered mip chain as nvdiffrast's MipBuildKernel builds it (2x2 average, or a 2-tap
    average once one side reaches 1)."""
    levels = [tex]
    for _ in range(max_level):
        t = levels[-1]
        h, w = t.shape[:2]
        if h == 1 and w == 1:
            break
        if h == 1:
            t = 0.5 * (t[:, 0:w - 1:2] + t[:, 1:w:2])
        elif w == 1:
            t = 0.5 * (t[0:h - 1:2] + t[1:h:2])
        else:
            t = 0.25 * (t[0:h - 1:2, 0:w - 1:2] + t[0:h - 1:2, 1:w:2] + t[1:h:2, 0:w - 1:2] + t[1:h:2, 1:w:2])
        levels.append(t.contiguous())
    return levels


def _bilinear_wrap(tex, u, v):
    """indexTextureLinear + bilerp with wrap boundary for one mip level; u, v in [0, 1) units."""
    import torch

    h, w = tex.shape[:2]
    u = u - torch.floor(u)
    v = v - torch.floor(v)
    u = u * float(w) - 0.5
    v = v * float(h) - 0.5
    iu0 = torch.floor(u).to(torch.int64)
    iv0 = torch.floor(v).to(torch.int64)
    fu = (u - iu0.to(torch.float32)).unsqueeze(-1)
    fv = (v - iv0.to(torch.float32)).unsqueeze(-1)
    iu1, iv1 = iu0 + 1, iv0 + 1
    iu0 = torch.where(iu0 < 0, iu0 + w, iu0)
    iv0 = torch.where(iv0 < 0, iv0 + h, iv0)
    iu1 = torch.where(iu1 >= w, iu1 - w, iu1)
    iv1 = torch.where(iv1 >= h, iv1 - h, iv1)
    a00, a10 = tex[iv0, iu0], tex[iv0, iu1]
    a01, a11 = tex[iv1, iu0], tex[iv1, iu1]
    top = a00 + fu * (a10 - a00)
    bot = a01 + fu * (a11 - a01)
    return top + fv * (bot - top)


def texture(tex, uv, uv_da, max_mip_level):
    """nvdiffrast.torch.texture(..., filter_mode="linear-mipmap-linear", boundary_mode="wrap") for
    one image. ``tex`` is (1, Ht, Wt, C), ``uv`` (1, H, W, 2), ``uv_da`` (1, H, W, 4)."""
    import torch

    if uv.device.type == "cuda" and backend(uv.device) == "nvdiffrast":
        import nvdiffrast.torch as dr

        return dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear", boundary_mode="wrap",
                          max_mip_level=max_mip_level)
    t = tex[0].to(torch.float32)
    th, tw = t.shape[:2]
    levels = _mip_pyramid(t, max_mip_level)
    level_max = len(levels) - 1
    H, W = uv.shape[1:3]
    u, v = uv[0, :, :, 0].reshape(-1), uv[0, :, :, 1].reshape(-1)
    d = uv_da[0].reshape(-1, 4)
    dsdx, dsdy = d[:, 0] * float(tw), d[:, 1] * float(tw)
    dtdx, dtdy = d[:, 2] * float(th), d[:, 3] * float(th)
    A = dsdx * dsdx + dtdx * dtdx
    B = dsdy * dsdy + dtdy * dtdy
    C = dsdx * dsdy + dtdx * dtdy
    l2b = 0.5 * (A + B)
    l2a = torch.sqrt(0.25 * (A - B) * (A - B) + C * C)
    flevel = 0.5 * torch.log2(l2b + l2a)
    flevel = torch.nan_to_num(flevel, nan=0.0, posinf=float(level_max), neginf=0.0)
    flevel = flevel.clamp(0.0, float(level_max))
    level0 = torch.floor(flevel).to(torch.int64)
    frac = torch.where(flevel > 0, flevel - level0.to(torch.float32), torch.zeros_like(flevel))
    out = torch.empty((u.shape[0], t.shape[2]), dtype=torch.float32, device=t.device)
    for lv in range(level_max + 1):
        m = level0 == lv
        if not bool(m.any()):
            continue
        a = _bilinear_wrap(levels[lv], u[m], v[m])
        l1 = int(min(lv + 1, level_max))
        f = frac[m].unsqueeze(-1)
        nz = f[:, 0] > 0
        if bool(nz.any()):
            b = _bilinear_wrap(levels[l1], u[m][nz], v[m][nz])
            a[nz] = a[nz] + f[nz] * (b - a[nz])
        out[m] = a
    return out.reshape(1, H, W, -1)
