"""Scene-linear <-> Log RGB via native gamut (D65).

Clamp policy (re-encode / out-of-gamut)
--------------------------------------
1. Decode source Log code values to scene-linear in the *source* native gamut.
2. Convert to CIE XYZ (D65) with ``gamut_to_xyz``, then to the *target* gamut
   with ``xyz_to_gamut``.  After the gamut change, channels may be negative
   (outside the target primaries) or arbitrarily large.
3. **Negative linear** after the gamut transform: clamp to 0.0.  Fraction of
   pixels with any negative channel *before* clamp is recorded as
   ``clamp.negative_linear_fraction``.
4. **Encode** with the target transfer function (per channel).  The published
   curves are defined for non-negative linear; values that would encode above
   code 1.0 or below 0.0 are **clamped to [0, 1]** after encode.  Fractions:
   ``clamp.code_below_0_fraction``, ``clamp.code_above_1_fraction``.
5. Documented sensor clip codes (LogC3 / D-Log EI tables): **chart and
   linear_dir** clamp codes at that clip (``apply_sensor_clip=True``), so the
   chart's specular patch forms a plateau at the documented value; curves
   without a documented clip plateau at code 1.0.  **Re-encode does not**
   (``apply_sensor_clip=False``): it is a pure curve/gamut remap and keeps the
   source footage's own clip.
6. Quantise to 16-bit full-range: ``round(code * 65535)`` clipped to uint16.

Identity round-trip (same curve + gamut) should show near-zero clamp fractions
and low MSE aside from 16-bit quantisation and the few published
``round_trip_exclusions`` intervals in ``cameralogbench.curves``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cameralogbench import curves
from cameralogbench.taxonomy import CURVES


def rec709_oetf(linear: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 OETF (display-referred), input scene-linear >= 0."""
    x = np.maximum(np.asarray(linear, dtype=np.float64), 0.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


def fake_log_encode(linear_rgb: np.ndarray) -> np.ndarray:
    """Non-vendor 'pseudo Log' look: mild log + pull toward grey midtones.

    Intentionally *not* equal to any curve in taxonomy; gold label = not_log.
    """
    x = np.maximum(np.asarray(linear_rgb, dtype=np.float64), 0.0)
    # soft log mapping linear [0, 8] onto [0, 1]: black at code 0, 18 % grey at ~257 CV
    y = np.log1p(x * 12.0) / np.log1p(12.0 * 8.0)
    y = np.clip(y, 0.0, 1.0)
    luma = 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
    # desaturate toward grey (a flat 'grey' look)
    return 0.40 * y + 0.60 * luma[..., None]


def apply_exposure(linear: np.ndarray, stops: float) -> np.ndarray:
    if stops == 0.0:
        return linear
    return linear * (2.0 ** stops)


def add_sensor_noise(linear: np.ndarray, rng: np.random.Generator,
                     read_std: float = 0.002, poisson_scale: float = 800.0) -> np.ndarray:
    """Optional Poisson shot noise + Gaussian read noise in scene-linear."""
    x = np.maximum(linear, 0.0)
    # Poisson on photon counts; poisson_scale ~ electrons at linear=1
    lam = x * poisson_scale
    # normal approx for large lam; exact Poisson for small
    shot = rng.normal(lam, np.sqrt(np.maximum(lam, 1e-6))) / poisson_scale
    read = rng.normal(0.0, read_std, size=x.shape)
    return np.maximum(shot + read, 0.0)


def rgb709_to_gamut(linear_709: np.ndarray, gamut_id: str) -> np.ndarray:
    """Rec.709 scene-linear RGB (D65) -> target gamut scene-linear RGB."""
    M_src = curves.gamut_to_xyz("rec709")
    M_dst_inv = curves.xyz_to_gamut(gamut_id)
    xyz = linear_709 @ M_src.T
    return xyz @ M_dst_inv.T


def gamut_to_gamut(linear_src: np.ndarray, src_gamut: str, dst_gamut: str) -> np.ndarray:
    if src_gamut == dst_gamut:
        return linear_src
    M_src = curves.gamut_to_xyz(src_gamut)
    M_dst_inv = curves.xyz_to_gamut(dst_gamut)
    xyz = linear_src @ M_src.T
    return xyz @ M_dst_inv.T


def encode_linear_to_code(linear_native: np.ndarray, curve_id: str, ei=None,
                          apply_sensor_clip: bool = True
                          ) -> tuple[np.ndarray, dict[str, float]]:
    """Encode non-negative native-gamut linear -> code [0,1]; return clamp stats.

    When ``apply_sensor_clip`` is True and the curve publishes a per-EI sensor
    clip (LogC3 / D-Log X9), codes above that clip are clamped so the chart
    specular patch forms a plateau at the documented 10-bit clip value.
    """
    neg = linear_native < 0
    neg_frac = float(np.mean(np.any(neg, axis=-1))) if linear_native.ndim == 3 else float(np.mean(neg))
    lin = np.maximum(linear_native, 0.0)
    code = np.asarray(curves.encode(curve_id, lin, ei), dtype=np.float64)
    below = code < 0
    above = code > 1
    below_frac = float(np.mean(np.any(below, axis=-1))) if code.ndim == 3 else float(np.mean(below))
    above_frac = float(np.mean(np.any(above, axis=-1))) if code.ndim == 3 else float(np.mean(above))
    code = np.clip(code, 0.0, 1.0)
    sensor_frac = 0.0
    ei_used = curves.resolve_ei(curve_id, ei)
    spec = curves.CURVE_SPECS[curve_id]
    if apply_sensor_clip and spec.clip_cv10_by_ei and ei_used in spec.clip_cv10_by_ei:
        clip_code = spec.clip_cv10_by_ei[ei_used] / 1023.0
        over = code > clip_code
        sensor_frac = float(np.mean(np.any(over, axis=-1))) if code.ndim == 3 else float(np.mean(over))
        code = np.minimum(code, clip_code)
    stats = {
        "negative_linear_fraction": round(neg_frac, 6),
        "code_below_0_fraction": round(below_frac, 6),
        "code_above_1_fraction": round(above_frac, 6),
        "sensor_clip_fraction": round(sensor_frac, 6),
    }
    return code, stats


def code_to_u16(code: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(code * 65535.0), 0, 65535).astype(np.uint16)


def u16_to_code(img16: np.ndarray) -> np.ndarray:
    return img16.astype(np.float64) / 65535.0


def encode_chart_linear(linear_709: np.ndarray, curve_id: str, ei=None,
                        exposure_stops: float = 0.0,
                        noise_rng: Optional[np.random.Generator] = None
                        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Chart Rec.709 linear -> 16-bit Log RGB (RGB order) + meta/clamp."""
    lin = apply_exposure(linear_709, exposure_stops)
    if noise_rng is not None:
        lin = add_sensor_noise(lin, noise_rng)
    gamut_id = curves.curve_gamut(curve_id) or "rec709"
    native = rgb709_to_gamut(lin, gamut_id)
    code, clamp = encode_linear_to_code(native, curve_id, ei)
    meta = {
        "curve_id": curve_id,
        "gamut_id": gamut_id,
        "ei": curves.resolve_ei(curve_id, ei),
        "exposure_stops": exposure_stops,
        "noise": noise_rng is not None,
        "clamp": clamp,
        "confidence": curves.CURVE_SPECS[curve_id].confidence,
        "gamut_provisional": curves.CURVE_SPECS[curve_id].gamut_provisional,
    }
    return code_to_u16(code), meta


def encode_not_log(linear_709: np.ndarray, variant: str,
                   exposure_stops: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    lin = apply_exposure(linear_709, exposure_stops)
    if variant == "rec709":
        code = np.clip(rec709_oetf(lin), 0.0, 1.0)
    elif variant == "fake_log":
        code = fake_log_encode(lin)
    else:
        raise ValueError(f"unknown not_log variant {variant!r}")
    return code_to_u16(code), {
        "curve_id": "not_log", "gamut_id": "rec709", "ei": None,
        "exposure_stops": exposure_stops, "not_log_variant": variant,
        "clamp": {"negative_linear_fraction": 0.0, "code_below_0_fraction": 0.0,
                  "code_above_1_fraction": 0.0, "sensor_clip_fraction": 0.0},
    }


def reencode_log_rgb(img16_rgb: np.ndarray, src_curve: str, dst_curve: str,
                     src_ei=None, dst_ei=None, src_gamut: Optional[str] = None,
                     dst_gamut: Optional[str] = None
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode source Log RGB16 -> XYZ -> target gamut+curve -> RGB16.

    See module docstring for clamp policy.  ``dji_dlogm`` as source is rejected
    by the caller (unverified); as target it is allowed and flagged.
    """
    if src_curve not in curves.CURVE_SPECS:
        raise KeyError(src_curve)
    if dst_curve not in curves.CURVE_SPECS:
        raise KeyError(dst_curve)
    src_g = src_gamut or curves.curve_gamut(src_curve) or "rec709"
    dst_g = dst_gamut or curves.curve_gamut(dst_curve) or "rec709"
    code_src = u16_to_code(img16_rgb)
    lin_src = np.asarray(curves.decode(src_curve, code_src, src_ei), dtype=np.float64)
    lin_dst = gamut_to_gamut(lin_src, src_g, dst_g)
    # No destination sensor-clip hard-cap: re-encode is a pure curve/gamut remap
    # (a control), not a camera-clip simulation.
    code, clamp = encode_linear_to_code(lin_dst, dst_curve, dst_ei, apply_sensor_clip=False)
    meta = {
        "src_curve_id": src_curve, "dst_curve_id": dst_curve,
        "src_gamut_id": src_g, "dst_gamut_id": dst_g,
        "src_ei": curves.resolve_ei(src_curve, src_ei),
        "dst_ei": curves.resolve_ei(dst_curve, dst_ei),
        "clamp": clamp,
        "dst_confidence": curves.CURVE_SPECS[dst_curve].confidence,
        "dst_gamut_provisional": curves.CURVE_SPECS[dst_curve].gamut_provisional,
        "identity": src_curve == dst_curve and src_g == dst_g,
    }
    return code_to_u16(code), meta


def identity_error(src16: np.ndarray, dst16: np.ndarray) -> dict[str, float]:
    a = src16.astype(np.float64)
    b = dst16.astype(np.float64)
    err = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    return {
        "max_abs_u16": float(err.max()),
        "mse_u16": mse,
        "rmse_u16": float(np.sqrt(mse)),
        "max_abs_cv10": float(err.max() * 1023.0 / 65535.0),
        "mse_cv10": float(np.mean(((a - b) * 1023.0 / 65535.0) ** 2)),
    }


def brand_for_curve(curve_id: str) -> Optional[str]:
    c = CURVES.get(curve_id)
    return c.brand if c else None
