"""Evidence table + per-curve formula check: the training-free formula baseline.

``compute_evidence(image)`` measures one frame and checks it against every candidate curve.  All code values
are 10-bit full-range floats: ``cv10 = v16 / 65535 * 1023``.  Statistics are
computed on the full 16-bit image; spatial statistics use non-overlapping
24x24 pixel blocks (160x90 blocks for UHD), whose means suppress per-pixel noise
by ~24x.

Field definitions
-----------------
channel_stats
    For R, G, B and Y (Y = 0.2126R + 0.7152G + 0.0722B of the *code values*, a
    curve-agnostic summary, not luminance): min, p0_1, p1, p50, p99, p99_9, max
    (exact percentiles from 16-bit histograms).
black
    estimate_cv: 0.1st percentile of the 24x24 block-mean min(R,G,B) over all
    blocks (the scene's dark floor).  fine_floor_cv: same on 4x4 block means
    (catches small deep shadows; per-pixel noise reduced 4x and 4:2:2 chroma
    reconstruction errors averaged) - used for the violation test.
    violation_tol_cv = max(5, 3 * noise_at_floor_cv / 4).  pixel_p0_1_cv: 0.1st
    percentile of per-pixel min(R,G,B) inside the darkest 1 % of blocks
    (noisiest).  floor_spread_cv: p1 - p0.1 of the 24x24 block-mean min channel.
    noise_at_floor_cv: median within-block std (mean of channels) of the darkest
    1 % of blocks (includes texture, so an upper bound on noise).
    has_true_black: floor_spread_cv <= 4 and estimate_cv <= TRUE_BLACK_MAX_CV
    (highest candidate black, Apple Log 154, + 10), i.e. a flat cluster of dark
    blocks at a plausible log black level.  When False the floor is a scene
    property and black-level evidence is one-sided.
highlight
    max_cv: largest code in any channel.  robust_max_cv: 99.995th percentile of
    per-pixel max(R,G,B).  plateau_cv: centre of a narrow histogram spike of
    per-pixel max(R,G,B) (1-CV bins) within 40 CV below robust_max: the peak
    bin must be >= 3x the median of the bins 8-30 CV below it, the 3 bins
    around it must hold >= max(300 px, 5e-5 of pixels), and at most 25 % of
    that count may lie > 2 CV above the peak (hard cutoff); null if none.
    plateau_rgb_cv: same test per channel.  clipped_fraction: share of pixels
    with max(R,G,B) >= plateau_cv - 3 (or >= 1020 without a plateau).
flat_patches
    Up to 24 low-variance blocks spread over 8 brightness quantiles (per bin:
    blocks below the bin's 10th std percentile, lowest std first, >= 3 blocks
    apart).  bbox [x, y, w, h] in pixels; mean_rgb_cv; std_cv (mean channel
    std inside the block); chroma = max - min of mean_rgb_cv (log-domain, curve
    agnostic).  Blocks touching the highlight plateau / code >= 1015 are skipped.
neutral_candidates_cv
    Mean code (of R, G, B) of flat patches with chroma <= 8 CV, ascending.
saturated_patches
    Up to 12 blocks with the largest chroma among blocks whose std is below the
    median and whose min channel is >= black.estimate_cv + 5 (avoids edges and
    noise), >= 3 blocks apart.
expected_table
    ``curves.expected_table_row`` per candidate curve: 10-bit black / grey18 /
    white90 / clip code, slope_per_stop (CV per stop at grey), clip_linear,
    default EI where relevant.
curve_consistency[curve_id]
    black_expected, black_delta = black.estimate_cv - black_expected.
    black_violation_cv = max(0, black_expected - violation_tol_cv - fine_floor_cv)
    (two-sided max(0, |black_delta| - violation_tol_cv) when has_true_black):
    the image contains averaged shadows darker than the curve can encode.
    decoded_negative_fraction: share of blocks with any channel decoding to
    < -0.001 linear (~5-6 CV below black for every curve here).
    imaginary_fraction: decode block means with the curve, convert with the
    curve's native gamut to XYZ, take blocks whose min channel is >= black + 10
    CV and not clipped; share whose xy lies outside the CIE 1931 spectral
    locus / purple line (relative purity > 1).  purity_p99: 99th percentile of
    relative purity of those blocks (1.0 = on the locus).
    exposure_key / exposure_offset_stops: log-average decoded luminance Y of
    non-clipped blocks (Reinhard et al. 2002 "key") and log2(key / 0.18).
    plateau_minus_clip_cv: plateau_cv - documented clip code at the default EI.
    plateau_matches_ei: EIs whose documented clip code is within +-6 CV of
    plateau_cv (LogC3 EI table, D-Log X9 EI table).
    plateau_stops_above_grey / max_stops_above_grey: log2(decode(code)/0.18).
    penalty = black_violation_cv / 10 + 20 * decoded_negative_fraction
              + 20 * imaginary_fraction + 0.5 * (exposure_offset_stops / 1.5)^2
              - 0.5 [if plateau_matches_ei].
    score = softmax(-penalty) over the 12 candidates (sums to 1).
    reference_fit (only when known reference patches are passed, e.g. a chart):
    neutral_rms_cv = RMS error of the neutral ladder with the exposure offset
    fitted freely (exposure_stops), chroma_dxy = mean |xy - reference xy| of
    known colour patches decoded with the curve + gamut, ei = best LogC3 EI.
    Adds penalty terms reference_neutral = 0.5 * (rms / 2 CV)^2 and
    reference_chroma = 0.5 * (dxy / 0.005)^2, and sets the exposure prior to 0
    (exposure is measured).  Without annotations nothing changes.
    These weights were fixed from first principles before looking at labels:
    the first three are physical-impossibility tests (a pixel darker than the
    curve's black; a real surface colour outside the spectral locus), the
    exposure term is a weak "normal key" prior (sigma 1.5 stops), and a
    plateau at a documented clip level is positive evidence.
classical_prediction
    ranked: curve ids by ascending penalty (ties keep taxonomy order);
    ruled_out: curves with any physical-impossibility term >= 1 (or a
    reference term >= 4.5, i.e. ~3 sigma, when reference patches exist);
    ranked_physical / tied_best_physical: same without the exposure prior
    (black + negative + imaginary - clip bonus + reference terms); most
    candidates tie at 0 when no test fires, so consumers should treat
    tied_best_physical as a set;
reference_patches
    Only with annotations: measured mean / std code per annotated patch.
    note: formula-based, no learning.
warnings
    Free-text caveats (no true black, no plateau, not 16-bit, ...).

Bit depth
---------
The paper's formula-baseline numbers were computed on 16-bit frames (``main16.png``) that are not part of
the public dataset.  The CLI below runs the same program on the released 8-bit lossless WebP frames
(8-bit values are scaled by 257 to the 16-bit range, so the code-value resolution is about 4 CV on the
10-bit scale); its numbers are close to but not identical with the paper's.

CLI::

    python -m cameralogbench.evidence --annotations data/CameraLogBench/annotations/real.jsonl \
        --data-root data/CameraLogBench --out runs/real__formula.jsonl [--patches synth_chart] [--workers 4]

writes one runner-format record per curve item (model ``formula-baseline``, tier ``formula``; score it with
``cameralogbench.eval.score``) and ``<out>.evidence.jsonl`` with the full evidence records.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cameralogbench import curves

BLOCK = 24
FINE_BLOCK = 4
BLACK_TOL_CV = 5.0
CLIP_MATCH_TOL_CV = 6.0
NEG_LIN = -0.001
EXPO_SIGMA_STOPS = 1.5
W_NEG = 20.0
W_IMAG = 20.0
CLIP_BONUS = 0.5
CANDIDATES = list(curves.CURVE_IDS)
EVIDENCE_VERSION = "v0.1"
# highest black level among the candidates (Apple Log 154) + margin
TRUE_BLACK_MAX_CV = max(curves.expected_code_values(c)["cv10_float"]["black"] for c in CANDIDATES) + 10.0
REF_NEUTRAL_SIGMA_CV = 2.0
REF_CHROMA_SIGMA_XY = 0.005
REF_RULE_OUT_TERM = 4.5          # ~3 sigma
REF_DXY_CAP = 0.2
# Photographed charts (reference_model="real"): real cameras reproduce chart colours with a colorimetric
# error of roughly 0.01 in xy, and veiling glare lifts the dark patches; see reference_fit.
REF_CHROMA_SIGMA_XY_REAL = 0.01
REF_FLARE_MAX = 0.03             # veiling glare, as a fraction of a 100 % reflector


# --------------------------------------------------------------------------- #
# Image loading and basic statistics
# --------------------------------------------------------------------------- #

def load_rgb16(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3][..., ::-1]
    return np.ascontiguousarray(img)


def _to_cv(v16) -> float:
    return float(v16) / 65535.0 * 1023.0


def _hist_percentiles(counts: np.ndarray, ps) -> dict:
    cdf = np.cumsum(counts)
    n = cdf[-1]
    nz = np.flatnonzero(counts)
    out = {"min": _to_cv(nz[0]), "max": _to_cv(nz[-1])}
    for name, p in ps:
        idx = int(np.searchsorted(cdf, p / 100.0 * n, side="left"))
        out[name] = _to_cv(min(idx, len(counts) - 1))
    return out


_PCTS = (("p0_1", 0.1), ("p1", 1.0), ("p50", 50.0), ("p99", 99.0), ("p99_9", 99.9))


def channel_stats(img16: np.ndarray) -> dict:
    res = {}
    for i, ch in enumerate("RGB"):
        res[ch] = _hist_percentiles(np.bincount(img16[..., i].ravel(), minlength=65536), _PCTS)
    y = (img16[..., 0] * np.float32(0.2126) + img16[..., 1] * np.float32(0.7152)
         + img16[..., 2] * np.float32(0.0722))
    y16 = np.clip(np.rint(y), 0, 65535).astype(np.uint16)
    res["Y"] = _hist_percentiles(np.bincount(y16.ravel(), minlength=65536), _PCTS)
    order = ("min", "p0_1", "p1", "p50", "p99", "p99_9", "max")
    return {k: {o: round(v[o], 2) for o in order} for k, v in res.items()}


def block_stats(cv: np.ndarray, b: int = BLOCK):
    h, w, _ = cv.shape
    nby, nbx = h // b, w // b
    blocks = cv[: nby * b, : nbx * b].reshape(nby, b, nbx, b, 3)
    mean = blocks.mean(axis=(1, 3), dtype=np.float64)
    dev = blocks - mean[:, None, :, None, :].astype(np.float32)
    std = np.sqrt((dev * dev).mean(axis=(1, 3), dtype=np.float64))
    return mean, std


# --------------------------------------------------------------------------- #
# Black / highlight
# --------------------------------------------------------------------------- #

def estimate_black(cv: np.ndarray, mean: np.ndarray, std: np.ndarray, b: int = BLOCK) -> dict:
    bmin = mean.min(axis=2)
    floor = float(np.percentile(bmin, 0.1))
    p1 = float(np.percentile(bmin, 1.0))
    dark = bmin <= p1
    ys, xs = np.nonzero(dark)
    pix = np.concatenate([cv[y * b:(y + 1) * b, x * b:(x + 1) * b].min(axis=2).ravel() for y, x in zip(ys, xs)])
    noise = float(np.median(std[dark].mean(axis=1)))
    spread = p1 - floor
    true_black = bool(spread <= 4.0 and floor <= TRUE_BLACK_MAX_CV)
    f = FINE_BLOCK
    h, w, _ = cv.shape
    fine = cv[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3)).min(axis=2)
    fine_floor = float(np.percentile(fine, 0.1))
    tol = max(BLACK_TOL_CV, 3.0 * noise / f)
    return {
        "estimate_cv": round(floor, 2),
        "method": "p0.1 of 24x24 block-mean min(R,G,B) over all blocks",
        "fine_floor_cv": round(fine_floor, 2),
        "fine_floor_method": "p0.1 of 4x4 block-mean min(R,G,B); used for the black-violation test",
        "violation_tol_cv": round(tol, 2),
        "pixel_p0_1_cv": round(float(np.percentile(pix, 0.1)), 2),
        "floor_spread_cv": round(spread, 2),
        "noise_at_floor_cv": round(noise, 2),
        "has_true_black": true_black,
    }


def _hist1024(values16: np.ndarray) -> np.ndarray:
    counts = np.bincount(values16.ravel(), minlength=65536)
    idx = (np.arange(65536) * (1023.0 / 65535.0)).astype(np.int64)
    return np.bincount(idx, weights=counts, minlength=1024)[:1024]


def detect_plateau(hist: np.ndarray, n: int) -> Optional[float]:
    cdf = np.cumsum(hist)
    r = int(np.searchsorted(cdf, 0.99995 * cdf[-1]))
    lo, hi = max(r - 40, 32), min(r + 3, 1023)
    if hi <= lo:
        return None
    m = lo + int(np.argmax(hist[lo:hi + 1]))
    a, z = max(m - 1, 0), min(m + 2, 1024)
    count = float(hist[a:z].sum())
    base = hist[max(m - 30, 0):max(m - 8, 1)]
    base_per_bin = float(np.median(base)) if base.size else 0.0
    above = float(hist[min(m + 3, 1024):].sum())
    if count < max(300.0, 5e-5 * n):
        return None
    if hist[m] < 3.0 * max(base_per_bin, 1.0):
        return None
    if above > 0.25 * count:
        return None
    centers = np.arange(a, z) + 0.5
    return float(np.sum(centers * hist[a:z]) / count)


def highlight_stats(img16: np.ndarray) -> dict:
    n = img16.shape[0] * img16.shape[1]
    mx16 = img16.max(axis=2)
    hist = _hist1024(mx16)
    cdf = np.cumsum(hist)
    robust = float(np.searchsorted(cdf, 0.99995 * cdf[-1])) + 0.5
    plateau = detect_plateau(hist, n)
    per_ch = {}
    for i, ch in enumerate("RGB"):
        p = detect_plateau(_hist1024(img16[..., i]), n)
        per_ch[ch] = round(p, 1) if p is not None else None
    thr = (plateau - 3.0) if plateau is not None else 1020.0
    thr16 = thr / 1023.0 * 65535.0
    clipped = float(np.count_nonzero(mx16 >= thr16)) / n
    if plateau is None:
        note = "no plateau detected (highlights may not reach sensor clip)"
    elif all(v is not None and abs(v - plateau) <= 4 for v in per_ch.values()):
        note = "plateau present in all three channels (white clip)"
    else:
        note = "plateau in max(R,G,B) only partially shared by channels"
    return {
        "max_cv": round(_to_cv(int(img16.max())), 2),
        "robust_max_cv": round(robust, 1),
        "plateau_cv": round(plateau, 1) if plateau is not None else None,
        "plateau_rgb_cv": per_ch,
        "clipped_fraction": round(clipped, 6),
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #

def _nms_pick(order, coords, k: int, radius: int = 2, taken=None) -> list[int]:
    taken = [] if taken is None else taken
    picked = []
    for i in order:
        y, x = coords[i]
        if any(abs(y - ty) <= radius and abs(x - tx) <= radius for ty, tx in taken):
            continue
        picked.append(i)
        taken.append((y, x))
        if len(picked) >= k:
            break
    return picked


def _patch(y, x, mean, std, b=BLOCK) -> dict:
    m = mean[y, x]
    return {"bbox": [int(x * b), int(y * b), b, b],
            "mean_rgb_cv": [round(float(v), 1) for v in m],
            "std_cv": round(float(std[y, x].mean()), 2),
            "chroma": round(float(m.max() - m.min()), 1)}


def find_patches(mean: np.ndarray, std: np.ndarray, black_cv: float, clip_cv: float):
    nby, nbx, _ = mean.shape
    coords = [(y, x) for y in range(nby) for x in range(nbx)]
    m = mean.reshape(-1, 3)
    s = std.reshape(-1, 3).mean(axis=1)
    luma = m @ np.array([0.2126, 0.7152, 0.0722])
    chroma = m.max(axis=1) - m.min(axis=1)
    usable = m.max(axis=1) < min(clip_cv - 3.0, 1015.0)

    flat_idx: list[int] = []
    taken: list = []
    edges = np.quantile(luma[usable], np.linspace(0, 1, 9)) if usable.any() else []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero(usable & (luma >= lo) & (luma <= hi))
        if sel.size == 0:
            continue
        thr = np.percentile(s[sel], 10)
        cand = sel[s[sel] <= thr]
        cand = cand[np.argsort(s[cand], kind="stable")]
        flat_idx += _nms_pick(cand, coords, 3, taken=taken)
    flat_idx = sorted(flat_idx, key=lambda i: luma[i])[:24]
    flat = [_patch(*coords[i], mean, std) for i in flat_idx]
    neutral = sorted(round(float(m[i].mean()), 1) for i in flat_idx if chroma[i] <= 8.0)

    sat_ok = usable & (s <= np.median(s)) & (m.min(axis=1) >= black_cv + 5.0)
    cand = np.flatnonzero(sat_ok)
    cand = cand[np.argsort(-chroma[cand], kind="stable")]
    sat_idx = _nms_pick(cand, coords, 12)
    sat = [_patch(*coords[i], mean, std) for i in sat_idx]
    return flat, neutral, sat


# --------------------------------------------------------------------------- #
# Known reference patches (calibration charts, synthetic charts)
# --------------------------------------------------------------------------- #

def measure_annotations(cv: np.ndarray, annotations: list[dict]) -> list[dict]:
    """Mean / std code value (cv10) over the central 60 % of each annotated bbox."""
    out = []
    for a in annotations:
        x, y, w, h = (int(v) for v in a["bbox"])
        ix, iy = int(round(w * 0.2)), int(round(h * 0.2))
        reg = cv[y + iy:y + h - iy, x + ix:x + w - ix].reshape(-1, 3).astype(np.float64)
        out.append({"name": a["name"], "kind": a["kind"], "bbox": [x, y, w, h],
                    "reflectance": a.get("reflectance"), "xy": a.get("xy"),
                    "measured_cv": [round(float(v), 3) for v in reg.mean(axis=0)],
                    "std_cv": round(float(reg.std(axis=0).mean()), 3)})
    return out


def _ref_variants(cid: str) -> list:
    return sorted(curves.LOGC3_PARAMS) if cid == "arri_logc3" else [None]


def reference_fit(cid: str, patches: list[dict], clip_obs: float, real: bool = False) -> Optional[dict]:
    """Exposure-free consistency of a curve hypothesis with known reference patches.

    neutral: patches of known (relative) reflectance; the exposure offset s (stops)
    is fitted, rms = RMS(mean measured code - encode(curve, reflectance * 2^s)).
    color: patches with a known D65 chromaticity; decode with curve + native gamut
    and take the mean xy distance (capped at REF_DXY_CAP), exposure-invariant.
    Patches touching code 0 or the observed clip are skipped.  For LogC3 the EI is
    a nuisance parameter: the EI with the lowest combined term is kept.

    real=True (photographed charts): the neutral fit also fits a veiling-glare
    term f in [0, REF_FLARE_MAX] (encode((reflectance + f) * 2^s)); colour patches
    are decoded, the glare is subtracted and the camera RGB is white-balanced on the
    chart's own mid neutrals (reflectance 0.05-0.7) before the xy comparison; the
    chroma sigma is REF_CHROMA_SIGMA_XY_REAL.  Default (synthetic charts) unchanged.
    """
    neutrals = [p for p in patches if p["kind"] == "neutral" and p.get("reflectance")
                and 2.0 < np.mean(p["measured_cv"]) < clip_obs - 2.0]
    colors = [p for p in patches if p["kind"] == "color" and p.get("xy")
              and min(p["measured_cv"]) > 2.0 and max(p["measured_cv"]) < clip_obs - 2.0]
    if len(neutrals) < 2 and not colors:
        return None
    rho = np.array([p["reflectance"] for p in neutrals], dtype=np.float64)
    m_neu = np.array([np.mean(p["measured_cv"]) for p in neutrals], dtype=np.float64)
    m_col = np.array([p["measured_cv"] for p in colors], dtype=np.float64) / 1023.0
    ref_xy = np.array([p["xy"] for p in colors], dtype=np.float64).reshape(-1, 2)
    M = curves.gamut_to_xyz(curves.CURVE_SPECS[cid].gamut_id)
    flares = np.linspace(0.0, REF_FLARE_MAX, 31) if real else np.array([0.0])
    wb_neu = [p for p in neutrals if 0.05 < p["reflectance"] < 0.7] if real else []
    best = None
    for ei in _ref_variants(cid):
        clip_h = curves.expected_code_values(cid, ei)["cv10_float"]["clip"]
        rms = s_hat = None
        flare = 0.0
        if len(neutrals) >= 2:
            grid = np.arange(-8.0, 8.0001, 0.02)
            fit = None
            for f in flares:
                pred = np.minimum(curves.encode(cid, (rho[None, :] + f) * 2.0 ** grid[:, None], ei) * 1023.0, clip_h)
                mse = np.mean((pred - m_neu[None, :]) ** 2, axis=1)
                k = int(np.argmin(mse))
                fine = np.linspace(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], 81)
                pred = np.minimum(curves.encode(cid, (rho[None, :] + f) * 2.0 ** fine[:, None], ei) * 1023.0, clip_h)
                mse_f = np.mean((pred - m_neu[None, :]) ** 2, axis=1)
                j = int(np.argmin(mse_f))
                if fit is None or mse_f[j] < fit[0]:
                    fit = (float(mse_f[j]), float(fine[j]), float(f))
            rms, s_hat, flare = math.sqrt(fit[0]), fit[1], fit[2]
        dxy = None
        if len(colors):
            lin = curves.decode(cid, m_col, ei)
            if real:
                glare = flare * 2.0 ** s_hat if s_hat is not None else 0.0
                lin = lin - glare
                if wb_neu:
                    g = curves.decode(cid, np.array([p["measured_cv"] for p in wb_neu]) / 1023.0, ei) - glare
                    g = g.mean(axis=0)
                    if np.all(g > 1e-9):
                        lin = lin * (g[1] / g)
            xyz = lin @ M.T
            ssum = xyz.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                xy = xyz[:, :2] / ssum[:, None]
            d = np.linalg.norm(xy - ref_xy, axis=1)
            d = np.where((ssum > 1e-9) & np.isfinite(d), np.minimum(d, REF_DXY_CAP), REF_DXY_CAP)
            dxy = float(d.mean())
        t_n = 0.5 * (rms / REF_NEUTRAL_SIGMA_CV) ** 2 if rms is not None else 0.0
        sig_c = REF_CHROMA_SIGMA_XY_REAL if real else REF_CHROMA_SIGMA_XY
        t_c = 0.5 * (dxy / sig_c) ** 2 if dxy is not None else 0.0
        cand = {"neutral_rms_cv": None if rms is None else round(rms, 3),
                "exposure_stops": None if s_hat is None else round(s_hat, 3),
                "chroma_dxy": None if dxy is None else round(dxy, 5),
                "ei": ei, "n_neutral": len(neutrals), "n_color": len(colors),
                **({"flare": round(flare, 4)} if real else {}),
                "terms": {"reference_neutral": t_n, "reference_chroma": t_c}}
        if best is None or t_n + t_c < sum(best["terms"].values()):
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Per-curve formula check
# --------------------------------------------------------------------------- #

def curve_consistency(mean: np.ndarray, black: dict, hl: dict, ref_patches: Optional[list] = None,
                      reference_model: str = "synthetic") -> dict:
    m = mean.reshape(-1, 3)
    bmin = m.min(axis=1)
    clip_cv = hl["plateau_cv"] if hl["plateau_cv"] is not None else 1020.0
    not_clipped = m.max(axis=1) < clip_cv - 3.0
    floor = black["estimate_cv"]
    fine_floor, tol = black["fine_floor_cv"], black["violation_tol_cv"]
    out = {}
    for cid in CANDIDATES:
        spec = curves.CURVE_SPECS[cid]
        exp = curves.expected_code_values(cid)
        b_exp = exp["cv10_float"]["black"]
        delta = floor - b_exp
        if black["has_true_black"]:
            viol = max(0.0, abs(delta) - tol)
        else:
            viol = max(0.0, b_exp - tol - fine_floor)
        lin = curves.decode(cid, m / 1023.0)
        neg = float(np.mean(np.any(lin < NEG_LIN, axis=1)))
        M = curves.gamut_to_xyz(spec.gamut_id)
        xyz = lin @ M.T
        ssum = xyz.sum(axis=1)
        valid = (bmin >= b_exp + 10.0) & not_clipped & (ssum > 1e-6)
        if valid.sum() >= 20:
            xy = xyz[valid, :2] / ssum[valid, None]
            pur = curves.relative_purity(xy)
            imag = float(np.mean(pur > 1.0))
            pur99 = float(np.percentile(pur, 99))
        else:
            imag, pur99 = 0.0, float("nan")
        Y = xyz[not_clipped, 1] if not_clipped.any() else xyz[:, 1]
        key = float(np.exp(np.mean(np.log(np.maximum(Y, 1e-4)))))
        offset = math.log2(key / 0.18)

        plateau = hl["plateau_cv"]
        entry = {
            "black_expected": round(b_exp, 1),
            "black_delta": round(delta, 1),
            "black_violation_cv": round(viol, 1),
            "decoded_negative_fraction": round(neg, 5),
            "imaginary_fraction": round(imag, 5),
            "purity_p99": None if math.isnan(pur99) else round(pur99, 3),
            "exposure_key": round(key, 4),
            "exposure_offset_stops": round(offset, 2),
            "clip_expected": exp["cv10"]["clip"],
            "plateau_minus_clip_cv": None,
            "plateau_matches_ei": [],
            "plateau_stops_above_grey": None,
            "max_stops_above_grey": None,
            "formula_confidence": spec.confidence,
            "gamut_used": spec.gamut_id + (" (provisional)" if spec.gamut_provisional else ""),
        }
        rmax_lin = float(curves.decode(cid, hl["robust_max_cv"] / 1023.0))
        if rmax_lin > 0:
            entry["max_stops_above_grey"] = round(math.log2(rmax_lin / 0.18), 2)
        if plateau is not None:
            entry["plateau_minus_clip_cv"] = round(plateau - exp["cv10_float"]["clip"], 1)
            pl_lin = float(curves.decode(cid, plateau / 1023.0))
            if pl_lin > 0:
                entry["plateau_stops_above_grey"] = round(math.log2(pl_lin / 0.18), 2)
            if spec.clip_cv10_by_ei:
                entry["plateau_matches_ei"] = sorted(
                    ei for ei, c in spec.clip_cv10_by_ei.items() if abs(plateau - c) <= CLIP_MATCH_TOL_CV)
        terms = {
            "black": viol / 10.0,
            "negative": W_NEG * neg,
            "imaginary": W_IMAG * imag,
            "exposure": 0.5 * (offset / EXPO_SIGMA_STOPS) ** 2,
            "clip_match": -CLIP_BONUS if entry["plateau_matches_ei"] else 0.0,
        }
        if ref_patches:
            rf = reference_fit(cid, ref_patches, clip_cv, real=reference_model == "real")
            if rf is not None:
                entry["reference_fit"] = {k: v for k, v in rf.items() if k != "terms"}
                terms.update(rf["terms"])
                if rf["exposure_stops"] is not None:
                    terms["exposure"] = 0.0     # exposure is measured, the prior no longer applies
        entry["penalty_terms"] = {k: round(v, 4) for k, v in terms.items()}
        entry["penalty"] = round(sum(terms.values()), 4)
        out[cid] = entry
    pen = np.array([out[c]["penalty"] for c in CANDIDATES])
    w = np.exp(-(pen - pen.min()))
    w /= w.sum()
    for c, s in zip(CANDIDATES, w):
        out[c]["score"] = round(float(s), 4)
    return out


def _physical_penalty(e: dict) -> float:
    t = e["penalty_terms"]
    return round(t["black"] + t["negative"] + t["imaginary"] + t["clip_match"]
                 + t.get("reference_neutral", 0.0) + t.get("reference_chroma", 0.0), 4)


def _ruled_out(e: dict) -> bool:
    t = e["penalty_terms"]
    return (max(t["black"], t["negative"], t["imaginary"]) >= 1.0
            or max(t.get("reference_neutral", 0.0), t.get("reference_chroma", 0.0)) >= REF_RULE_OUT_TERM)


def classical_prediction(cc: dict) -> dict:
    ranked = sorted(CANDIDATES, key=lambda c: cc[c]["penalty"])
    ruled = [c for c in CANDIDATES if _ruled_out(cc[c])]
    phys = {c: _physical_penalty(cc[c]) for c in CANDIDATES}
    ranked_phys = sorted(CANDIDATES, key=lambda c: phys[c])
    best = phys[ranked_phys[0]]
    uses_ref = any("reference_fit" in cc[c] for c in CANDIDATES)
    return {"ranked": ranked, "ruled_out": ruled,
            "ranked_physical": ranked_phys,
            "tied_best_physical": [c for c in ranked_phys if phys[c] == best],
            "uses_reference_patches": uses_ref,
            "note": "formula-based, no learning. ranked: penalty = black floor + negative decode + imaginary "
                    "colour + weak exposure-key prior - documented-clip bonus (+ reference-patch fits when "
                    "annotations are given, which replace the exposure prior). ranked_physical drops the "
                    "exposure prior; its ties keep taxonomy order, see tied_best_physical"}


# --------------------------------------------------------------------------- #
# Per-clip driver
# --------------------------------------------------------------------------- #

def compute_evidence(image_path: str, clip_id: str, batch_id: str = "", stem: str = "",
                     annotations: Optional[list[dict]] = None, reference_model: str = "synthetic") -> dict:
    """Evidence record for one image.  ``annotations`` (optional) = known reference
    patches: [{"name", "kind": "neutral"|"color"|"black"|"specular", "bbox": [x, y, w, h],
    "reflectance": relative reflectance for neutrals, "xy": D65 chromaticity for colors}].
    ``reference_model``: "synthetic" (rendered charts, default) or "real" (photographed charts:
    glare term, white balance on the chart's neutrals, real-camera chroma tolerance)."""
    t0 = time.time()
    img16 = load_rgb16(image_path)
    warnings: list[str] = []
    if img16.dtype != np.uint16:
        warnings.append(f"image dtype {img16.dtype}, expected uint16; values rescaled")
        img16 = (img16.astype(np.float64) * (65535.0 / np.iinfo(img16.dtype).max)).astype(np.uint16)
    sample = img16[::7, ::7]
    if np.all(sample % 257 == 0):
        warnings.append("all sampled values are multiples of 257: 8-bit source upsampled to 16-bit")
    h, w, _ = img16.shape
    cv = img16.astype(np.float32) * np.float32(1023.0 / 65535.0)

    stats = channel_stats(img16)
    mean, std = block_stats(cv)
    black = estimate_black(cv, mean, std)
    hl = highlight_stats(img16)
    ref_patches = measure_annotations(cv, annotations) if annotations else None
    del cv
    clip_cv = hl["plateau_cv"] if hl["plateau_cv"] is not None else 1020.0
    flat, neutral, sat = find_patches(mean, std, black["estimate_cv"], clip_cv)
    cc = curve_consistency(mean, black, hl, ref_patches, reference_model)
    pred = classical_prediction(cc)

    if not black["has_true_black"]:
        warnings.append("no true black detected: black-level evidence is one-sided (floor is a scene property)")
    if hl["plateau_cv"] is None:
        warnings.append("no highlight plateau: clip-level evidence unavailable")
    if not neutral:
        warnings.append("no low-chroma flat patch found")
    if len(pred["ruled_out"]) >= len(CANDIDATES) - 1:
        warnings.append("almost every candidate violates a physical test; check levels / processing")

    return {
        "clip_id": clip_id,
        "batch_id": batch_id,
        "stem": stem,
        "evidence_version": EVIDENCE_VERSION,
        "image": _rel(image_path),
        "width": int(w), "height": int(h), "bit_depth": 16,
        "channel_stats": stats,
        "black": black,
        "highlight": hl,
        "flat_patches": flat,
        "neutral_candidates_cv": neutral,
        "saturated_patches": sat,
        "expected_table": {c: curves.expected_table_row(c) for c in CANDIDATES},
        "curve_consistency": cc,
        "classical_prediction": pred,
        **({"reference_patches": ref_patches} if ref_patches else {}),
        "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _rel(p: str) -> str:
    try:
        return os.path.relpath(p)
    except ValueError:
        return p


def set_candidates(curve_ids: list[str]) -> None:
    """Change the candidate curves (module-wide; the default is the 12 benchmark curves)."""
    global CANDIDATES, TRUE_BLACK_MAX_CV
    CANDIDATES = list(curve_ids)
    TRUE_BLACK_MAX_CV = max(curves.expected_code_values(c)["cv10_float"]["black"] for c in CANDIDATES) + 10.0


def candidates_for(items: list[dict]) -> list[str]:
    """The 12 benchmark curves, plus the three extension encodings when the items use the 17-option set
    (the paper's choice: 15 candidates on real footage and charts, 12 on the synthetic natural scenes)."""
    from cameralogbench.taxonomy import EXT2_CURVE_IDS
    ext = [c for c in EXT2_CURVE_IDS if c not in curves.CURVE_IDS]
    uses_ext2 = any(set(ext) & set(it.get("options") or []) for it in items)
    return list(curves.CURVE_IDS) + (ext if uses_ext2 else [])


def predict(evidence: dict, known_patches: bool) -> str:
    """Top-ranked candidate; with known patch positions, "not_log" when every candidate is ruled out."""
    p = evidence["classical_prediction"]
    if known_patches and len(p["ruled_out"]) == len(CANDIDATES):
        return "not_log"
    return p["ranked"][0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _init(cands: list[str]) -> None:
    set_candidates(cands)


def _work(args) -> dict:
    image, clip_id, annotations = args
    return compute_evidence(image, clip_id, stem=clip_id, annotations=annotations)


def main(argv=None) -> int:
    from datetime import datetime

    ap = argparse.ArgumentParser(description="Formula baseline on annotated frames (see module docstring)")
    ap.add_argument("--annotations", nargs="+", required=True, help="annotation .jsonl file(s)")
    ap.add_argument("--data-root", required=True, help="dataset folder the image paths are relative to")
    ap.add_argument("--out", required=True, help="runner-format results .jsonl")
    ap.add_argument("--patches", choices=["none", "synth_chart"], default="none",
                    help="synth_chart: read the synthetic chart's patches at their known positions "
                         "(only for the synthetic chart subset)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    items = []
    for p in a.annotations:
        items += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    cands = candidates_for(items)
    set_candidates(cands)
    items = [it for it in items if it["task"] == "curve"]
    if a.limit:
        items = items[: a.limit]
    root = Path(a.data_root)
    missing = [it["item_id"] for it in items if not (root / it["input"]["image_webp"]).is_file()]
    if missing:
        print(f"[warn] {len(missing)} frames not found under {root}, e.g. {missing[0]}", file=sys.stderr)
    items = [it for it in items if it["item_id"] not in set(missing)]
    if not items:
        print("no curve items with frames found", file=sys.stderr)
        return 1
    ann = None
    if a.patches == "synth_chart":
        from cameralogbench.synth.chart import build_chart
        ann = build_chart()[1]
    jobs = [(str(root / it["input"]["image_webp"]), it["clip_id"], ann) for it in items]
    print(f"{len(items)} curve items, {len(cands)} candidate curves, patches={a.patches}", file=sys.stderr)
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cands,)) as ex:
            evs = list(ex.map(_work, jobs, chunksize=4))
    else:
        evs = [_work(j) for j in jobs]

    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    recs = []
    for it, ev in zip(items, evs):
        options = list(it.get("options") or [])
        gold = it["gold"]
        accept = [c for c in dict.fromkeys(list(gold.get("accept") or []) + [gold.get("primary")]) if c in options]
        pred = predict(ev, ann is not None)
        ranked = [c for c in ev["classical_prediction"]["ranked"] if c in options]
        top2 = [pred] + [c for c in ranked if c != pred][:1]
        recs.append({
            "run_tag": Path(a.out).stem, "model": "formula-baseline", "tier": "formula",
            "item_id": it["item_id"], "clip_id": it.get("clip_id"), "task": "curve",
            "rotation": 0, "n_rotations": 1, "options": options, "gold": gold, "accept_ids": accept,
            "twin_of": it.get("twin_of"), "raw": None, "parsed": None, "error": None, "parse_status": "formula",
            "choice_letter": None, "choice_id": pred, "top2_ids": top2,
            "correct": pred in accept, "top2_hit": any(t in accept for t in top2),
            "ranked": ev["classical_prediction"]["ranked"][:3], "ruled_out": ev["classical_prediction"]["ruled_out"],
            "ts": ts, "prompt_version": None, "image_variant": "webp", "frames_mode": "single", "frame_pos": None,
            "image_path": it["input"]["image_webp"], "dry_run": False,
            "item_meta": {k: v for k, v in (it.get("meta") or {}).items() if k in ("subset", "source_class")},
        })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out, recs), (out.with_name(out.stem + ".evidence.jsonl"), evs)):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    log = [r for r in recs if r["gold"]["primary"] != "not_log"]
    by: dict[str, list[bool]] = {}
    for r in log:
        by.setdefault(r["gold"]["primary"], []).append(r["correct"])
    if log:
        acc = sum(r["correct"] for r in log) / len(log)
        top2 = sum(r["top2_hit"] for r in log) / len(log)
        avg = sum(sum(v) / len(v) for v in by.values()) / len(by)
        print(f"Log items: n={len(log)} acc {acc:.2f} top-2 {top2:.2f} per-curve avg {avg:.2f} "
              f"({len(by)} encodings)", file=sys.stderr)
    neg = [r for r in recs if r["gold"]["primary"] == "not_log"]
    if neg:
        print(f"not-Log items: n={len(neg)} acc {sum(r['correct'] for r in neg) / len(neg):.2f}", file=sys.stderr)
    print(f"wrote {out} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
