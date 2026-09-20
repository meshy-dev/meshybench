"""Color: CIELCh agreement in hue, saturation and value between two albedo samples.

The reference surface is split into K_REGIONS k-means regions of its sample positions and a
candidate point takes the region of the nearest centre. Per channel, a mean term compares the
per-region means weighted by region size and a spread term compares the whole-surface spread;
both map a difference d to exp(-d / scale) in CIELCh units. Hue is circular and weighted by
chroma above CHROMA_FLOOR; its spread is the circular standard deviation. Saturation and value
spread is the 5th to 95th percentile span. The channel score is W_MEAN * mean + W_VAR * spread,
and Color is the SUBW-weighted sum of the three channels.
"""
from __future__ import annotations

import numpy as np

K_REGIONS = 12
W_MEAN, W_VAR = 0.7, 0.3
SUBW = {"hue": 0.5, "sat": 0.3, "value": 0.2}
SCALE_MEAN = {"hue": 20.0, "sat": 12.0, "value": 12.0}
SCALE_VAR = {"hue": 20.0, "sat": 12.0, "value": 12.0}
CHROMA_FLOOR = 8.0
MIN_REGION = 40
CHANNEL = {"hue": "h", "sat": "C", "value": "L"}


def _lch(albedo_lin):
    from skimage import color as skcolor

    a = np.clip(albedo_lin, 0.0, 1.0)
    srgb = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 1e-8, 1), 1 / 2.4) - 0.055)
    lab = skcolor.rgb2lab(srgb.reshape(-1, 1, 3)).reshape(-1, 3)
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    return L, np.hypot(A, B), np.degrees(np.arctan2(B, A)) % 360.0


def channels(samples) -> dict:
    """L, C, h arrays of a sample's albedo and whether the albedo is uniform."""
    L, C, h = _lch(samples.a)
    return {"L": L, "C": C, "h": h, "uniform": bool(samples.a.std() < 1e-4)}


def _circ_mean(h_deg, w):
    if w.sum() <= 1e-9:
        return np.nan
    r = np.radians(h_deg)
    return np.degrees(np.arctan2(np.sum(w * np.sin(r)), np.sum(w * np.cos(r)))) % 360.0


def _circ_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _circ_std(h_deg, w):
    if w.sum() <= 1e-9:
        return 0.0
    r = np.radians(h_deg)
    Rlen = np.hypot(np.sum(w * np.sin(r)), np.sum(w * np.cos(r))) / w.sum()
    Rlen = min(max(Rlen, 1e-6), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(Rlen))))


def _span(v):
    return float(np.quantile(v, 0.95) - np.quantile(v, 0.05))


def channel_score(ref, cand, ref_lab, cand_lab, sub) -> float:
    val_ref, val_cand = ref[CHANNEL[sub]], cand[CHANNEL[sub]]
    agrees, weights = [], []
    for r in range(K_REGIONS):
        gm, nm = ref_lab == r, cand_lab == r
        if gm.sum() < MIN_REGION or nm.sum() < MIN_REGION:
            continue
        if sub == "hue":
            wg = np.clip(ref["C"][gm] - CHROMA_FLOOR, 0, None)
            wn = np.clip(cand["C"][nm] - CHROMA_FLOOR, 0, None)
            a = _circ_mean(val_ref[gm], wg)
            b = _circ_mean(val_cand[nm], wn)
            d = _circ_dist(a, b) if not (np.isnan(a) or np.isnan(b)) else 0.0
        else:
            d = abs(float(val_ref[gm].mean()) - float(val_cand[nm].mean()))
        agrees.append(np.exp(-d / SCALE_MEAN[sub]))
        weights.append(gm.sum())
    mean_term = float(np.average(agrees, weights=weights)) if agrees else 1.0
    if sub == "hue":
        wg = np.clip(ref["C"] - CHROMA_FLOOR, 0, None)
        wn = np.clip(cand["C"] - CHROMA_FLOOR, 0, None)
        sg, sn = _circ_std(val_ref, wg), _circ_std(val_cand, wn)
    else:
        sg, sn = _span(val_ref), _span(val_cand)
    var_term = float(np.exp(-abs(sg - sn) / SCALE_VAR[sub]))
    return W_MEAN * mean_term + W_VAR * var_term
