"""Log transfer functions, native gamuts, expected code values and the
whitepaper digest of CameraLogBench.

Conventions (shared by every function in this module)
------------------------------------------------------
* ``linear`` is relative scene-linear exposure: 0.18 = 18 % grey card,
  0.90 = 90 % white card, values > 1 are specular / bright highlights.
  (Canon's "scene linear" uses 0.20 for grey; we always take *reflectance*
  as input and apply Canon's 1/0.9 factor internally.)
* ``code`` is the normalised full-range code value in [0, 1]:
  10-bit code value = ``code * 1023``.  Every whitepaper below states its
  curve in this form (S-Log3, LogC3, D-Log, V-Log, F-Log2, N-Log, Canon "Full %",
  Apple Log, LogC4 "10-bit Full", Log3G10, BMD Film Gen 5 normalised value).
  Some vendors *record* the same normalised value in video levels
  (``64 + 876 * code``): Blackmagic documents this for camera ProRes (Film Gen 5
  black = 145, grey = 400 "10-bit video levels").  Full-range 16-bit exports
  of RED and DJI material sit at ``code * 1023``, so this module uses
  ``code * 1023`` throughout and reports the alternative in
  ``expected_code_values(...)["levels_note"]``.
* All gamuts use the D65 white point (0.3127, 0.3290); matrices are derived
  from primaries + white and rounded to 6 decimals.

Curves covered: every taxonomy curve with status ``v1_main`` / ``v1_second``
(12 curves; extension curves are added with status ``v1_ext``).  Per-curve source URL, document version and confidence live in
``CURVE_SPECS[curve_id]`` (fields ``source``, ``doc_version``, ``confidence``)
and are summarised in the curve-function docstrings.  Confidence levels:

* ``verified``            - constants and anchor values read in the official PDF.
* ``verified_secondary``  - official anchor values confirmed; constants taken
                             from a reputable transcription (primary PDF not
                             machine-readable or behind a login).
* ``unverified``          - no official document exists / was found.

Gamut matrix cross-check (derived from primaries vs. the vendor's own printed
matrix, max |difference| over the 9 entries; see ``matrix_crosscheck()``):

* arri_awg3   4.5e-7  (ARRI LogC VFX paper)
* arri_awg4   1e-16   (ARRI LogC4 spec, 18-digit matrix)
* red_rwg     8.3e-7  (RED 915-0187 Rev-C)
* panasonic_vgamut 4.7e-7 (V-Log/V-Gamut manual)
* bmd_wg_gen5 2.5e-4  (BMD Gen 5 reference).  The BMD matrix uses BMD's own
  white (0.3127170, 0.3290312); with the plain D65 white mandated here the
  difference stays < 1e-3, so no special handling.
* dji_dgamut  1.8e-4  (DJI prints only 4 decimals; derived matrix agrees
  within that rounding).

All 11 officially documented curves were also cross-checked against
colour-science 0.4.7 (encode identical to <= 1e-15 on 0..30 linear; LogC3 at
EI160/400/800/1600); the spectral-locus table matches its CIE 1931 2-degree
CMFs to 6e-5.

CLI::

    python -m cameralogbench.curves --table          # expected code values
    python -m cameralogbench.curves --check          # matrix cross-check
    python -m cameralogbench.curves --digest         # write work/reference/digest{,_spec}.md + .json
    python -m cameralogbench.curves --digest --ext2  # the 17-option digest (digest_ext2*.md)

The shipped digests are in ``cameralogbench/reference/``; the generator reproduces them byte for byte
(see tests/test_curves.py).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from cameralogbench.taxonomy import CURVES, GAMUTS

# v1 curves, in the canonical taxonomy order
CURVE_IDS: list[str] = [cid for cid, c in CURVES.items() if c.status in ("v1_main", "v1_second")]

_TINY = 1e-300


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _log10(v: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(v, _TINY))


def _log2(v: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(v, _TINY))


def _ln(v: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(v, _TINY))


def _out(x_in, y: np.ndarray):
    """Return a Python float for scalar input, ndarray otherwise."""
    return float(y) if np.ndim(x_in) == 0 else y


# --------------------------------------------------------------------------- #
# Sony S-Log3
# --------------------------------------------------------------------------- #
_SLOG3_CUT_CV = 171.2102946929


def slog3_encode(x, ei=None):
    """Sony S-Log3 (curve independent of EI).

    Source: Sony "Technical Summary for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3"
    (F65/F55/F5 firmware v3.0, 2014), https://pro.sony/s3/cms-static-content/file/12/1237494271406.pdf
    Confidence: verified.
    Anchors: 0 -> 95, 0.18 -> 420, 0.90 -> 598 (10-bit).
    """
    x = _arr(x)
    lo = (x * (_SLOG3_CUT_CV - 95.0) / 0.01125 + 95.0) / 1023.0
    hi = (420.0 + _log10((x + 0.01) / 0.19) * 261.5) / 1023.0
    return np.where(x >= 0.01125, hi, lo)


def slog3_decode(y, ei=None):
    y = _arr(y)
    hi = 10.0 ** ((y * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lo = (y * 1023.0 - 95.0) * 0.01125 / (_SLOG3_CUT_CV - 95.0)
    return np.where(y >= _SLOG3_CUT_CV / 1023.0, hi, lo)


# --------------------------------------------------------------------------- #
# ARRI LogC3 (exposure-value parameters, EI 160-1600)
# --------------------------------------------------------------------------- #
#        EI: (cut, a, b, c, d, e, f)
LOGC3_PARAMS: dict[int, tuple[float, ...]] = {
    160: (0.005561, 5.555556, 0.080216, 0.269036, 0.381991, 5.842037, 0.092778),
    200: (0.006208, 5.555556, 0.076621, 0.266007, 0.382478, 5.776265, 0.092782),
    250: (0.006871, 5.555556, 0.072941, 0.262978, 0.382966, 5.710494, 0.092786),
    320: (0.007622, 5.555556, 0.068768, 0.259627, 0.383508, 5.637732, 0.092791),
    400: (0.008318, 5.555556, 0.064901, 0.256598, 0.383999, 5.571960, 0.092795),
    500: (0.009031, 5.555556, 0.060939, 0.253569, 0.384493, 5.506188, 0.092800),
    640: (0.009840, 5.555556, 0.056443, 0.250219, 0.385040, 5.433426, 0.092805),
    800: (0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809),
    1000: (0.011361, 5.555556, 0.047996, 0.244161, 0.386036, 5.301883, 0.092814),
    1280: (0.012235, 5.555556, 0.043137, 0.240810, 0.386590, 5.229121, 0.092819),
    1600: (0.013047, 5.555556, 0.038625, 0.237781, 0.387093, 5.163350, 0.092824),
}
# Log C value at sensor saturation per EI ("Clipping Level" table, same document)
LOGC3_CLIP_CODE: dict[int, float] = {
    160: 0.8128, 200: 0.8341, 250: 0.8549, 320: 0.8773, 400: 0.8968, 500: 0.9158,
    640: 0.9362, 800: 0.9539, 1000: 0.9711, 1280: 0.9895, 1600: 1.0,
    2000: 1.0, 2560: 1.0, 3200: 1.0,
}


def logc3_resolve_ei(ei) -> int:
    """Map a requested EI to a row of the LogC3 parameter table.

    None -> 800 (ARRI's recommended default).  EI > 1600 has no closed form
    (camera adds a shoulder); ARRI says the EI1600 formula is accurate up to
    Log C ~0.8, so EI1600 parameters are used.  Other EIs snap to the nearest
    table row in log2 distance.
    """
    if ei is None:
        return 800
    ei = float(ei)
    if ei >= 1600:
        return 1600
    return min(LOGC3_PARAMS, key=lambda k: abs(math.log2(ei / k)))


def logc3_encode(x, ei=None):
    """ARRI LogC3 (ALEXA SUP >= 3.0, AMIRA, Mini, LF, 65). Curve depends on EI.

    Source: ARRI "ALEXA Log C Curve - Usage in VFX", H. Brendel, revision
    09-Mar-2017, https://www.arri.com/en/learn-help/learn-help-camera-system/technical-downloads
    Confidence: verified.  Anchors: 0 -> 0.0928 (95), 0.18 -> 0.391 (400) at every EI.
    """
    cut, a, b, c, d, e, f = LOGC3_PARAMS[logc3_resolve_ei(ei)]
    x = _arr(x)
    return np.where(x > cut, c * _log10(a * x + b) + d, e * x + f)


def logc3_decode(y, ei=None):
    cut, a, b, c, d, e, f = LOGC3_PARAMS[logc3_resolve_ei(ei)]
    y = _arr(y)
    return np.where(y > e * cut + f, (10.0 ** ((y - d) / c) - b) / a, (y - f) / e)


# --------------------------------------------------------------------------- #
# ARRI LogC4
# --------------------------------------------------------------------------- #
_C4_A = (2.0 ** 18 - 16.0) / 117.45
_C4_B = (1023.0 - 95.0) / 1023.0
_C4_C = 95.0 / 1023.0
_C4_S = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * _C4_C / _C4_B)) / (_C4_A * _C4_B)
_C4_T = (2.0 ** (14.0 * (-_C4_C / _C4_B) + 6.0) - 64.0) / _C4_A


def logc4_encode(x, ei=None):
    """ARRI LogC4 (ALEXA 35 / 265). Single EI-independent curve.

    Source: ARRI "ARRI LogC4 Specification" (document history up to 2022-11-08),
    https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf
    Confidence: verified (equations 2/3, CTL reference, Appendix B/D tables:
    0 -> 0.0929 (95 full / 145 legal), 0.18 -> 0.2784 (285 full / 308 legal),
    code 1.0 -> 469.80).
    """
    x = _arr(x)
    hi = (_log2(_C4_A * x + 64.0) - 6.0) / 14.0 * _C4_B + _C4_C
    return np.where(x >= _C4_T, hi, (x - _C4_T) / _C4_S)


def logc4_decode(y, ei=None):
    y = _arr(y)
    hi = (2.0 ** (14.0 * (y - _C4_C) / _C4_B + 6.0) - 64.0) / _C4_A
    return np.where(y >= 0.0, hi, y * _C4_S + _C4_T)


# --------------------------------------------------------------------------- #
# DJI D-Log
# --------------------------------------------------------------------------- #
_DLOG_C = 0.256663
_DLOG_D = 0.584555
# Zenmuse X9 (Ronin 4D / Inspire 3) clipping level per EI, 10-bit code, 6K table.
DLOG_X9_CLIP_CV10: dict[int, float] = {
    200: 789, 250: 816, 320: 842, 400: 868, 500: 895, 640: 921, 800: 948,
    1000: 948, 1250: 948, 1600: 948, 2000: 948, 2500: 948, 3200: 948,
    4000: 921, 5000: 948, 6400: 948, 8000: 948, 10000: 948, 12800: 948,
}


def dlog_encode(x, ei=None):
    """DJI D-Log (Zenmuse X5S/X7 2017; Zenmuse X9 = Ronin 4D / Inspire 3, 2022).

    Sources: DJI "White Paper on D-Log and D-Gamut of DJI Cinema Color System"
    Rev 1.0, 2017-09-29, https://dl.djicdn.com/downloads/zenmuse+x7/20171010/D-Log_D-Gamut_Whitepaper.pdf
    and the X9 6K/8K edition Rev 1.0, 2022-02,
    https://dl.djicdn.com/downloads/DJI_Ronin_4D/X9_D_Log_D_Gamut_Whitepaper.pdf
    (same formula; adds the EI clipping table: 948 at native EI, 789 at EI200).
    Confidence: verified.  Anchors: 0 -> 95, 0.18 -> 408, 0.90 -> 586.

    In scene-linear terms the curve does not move with EI (grey stays at 408);
    lower EIs only lower the clip level.  The published forward function is
    discontinuous at in = 0.0078 (linear branch ends at 0.140895, log branch
    starts at 0.139896); decode uses the published cut (code 0.14) with exact
    reciprocal constants, so linear values in [0.00763, 0.00797] do not
    round-trip exactly (see ``round_trip_exclusions``).
    """
    x = _arr(x)
    return np.where(x <= 0.0078, 6.025 * x + 0.0929, _log10(x * 0.9892 + 0.0108) * _DLOG_C + _DLOG_D)


def dlog_decode(y, ei=None):
    y = _arr(y)
    hi = (10.0 ** ((y - _DLOG_D) / _DLOG_C) - 0.0108) / 0.9892
    return np.where(y <= 0.14, (y - 0.0929) / 6.025, hi)


# --------------------------------------------------------------------------- #
# DJI D-Log M (no official formula)
# --------------------------------------------------------------------------- #
_DLM_XS = -0.736169159412384
_DLM_YS = -0.9837290048599243
_DLM_SC = 7.4200921058654785
_DLM_S1 = 0.4385978579521179
_DLM_IC = 0.09801799058914185
_DLM_CUT = 0.1745949685573578
_DLM_MG = 0.0558909009613


def dlogm_decode(y, ei=None):
    """DJI D-Log M - UNVERIFIED community fit, not a DJI specification.

    DJI has published no formula or gamut for D-Log M (only D-Log M -> Rec.709
    LUTs; DJI support calls it proprietary; Resolve has no CST for it).  We use
    Thatcher Freeman's empirical fit for the Mavic 3 Pro wide camera,
    https://github.com/thatcherfreeman/dwg-transforms ("DJI Mavic 3 Pro D-Log M
    to DWG.dctl", 2024).  The fit author *assumed* 40 % signal = 18 % grey, so
    the grey anchor is a convention, not a measurement.  Fits for other bodies
    (Mini 4 Pro, Pocket 3, Action 5) differ noticeably: 90 % white lands at
    697-779 and code 1.0 at 2.3-4.8 linear.  Confidence: unverified.
    Shape: black ~3, grey 409, ~118 CV/stop at grey (much steeper than a true
    log), clip at only ~4.7 stops above grey.
    """
    y = _arr(y)
    t = 2.0 ** (y * _DLM_SC + _DLM_YS) + _DLM_XS
    return np.where(t < _DLM_CUT, t * _DLM_S1 + _DLM_IC, t) * _DLM_MG


def dlogm_encode(x, ei=None):
    x = _arr(x)
    u = x / _DLM_MG
    t = np.where(u < _DLM_CUT, (u - _DLM_IC) / _DLM_S1, u)
    return (_log2(t - _DLM_XS) - _DLM_YS) / _DLM_SC


# --------------------------------------------------------------------------- #
# RED Log3G10
# --------------------------------------------------------------------------- #
_L3G10_A, _L3G10_B, _L3G10_C, _L3G10_G = 0.224282, 155.975327, 0.01, 15.1927


def log3g10_encode(x, ei=None):
    """RED Log3G10 (IPP2; also Nikon ZR). EI-independent.

    Source: RED "White Paper on REDWideGamutRGB and Log3G10", 915-0187 Rev-C,
    2017-11-06, https://docs.red.com/955-0187/PDF/915-0187%20Rev-C%20%20%20RED%20OPS%2C%20White%20Paper%20on%20REDWideGamutRGB%20and%20Log3G10.pdf
    Confidence: verified (a=0.224282, b=155.975327, c=0.01, g=15.1927; mapping
    table -0.01 -> 0, 0 -> 0.091551, 0.18 -> 1/3, 1.0 -> 0.493449, 184.322 -> 1.0).
    """
    x = _arr(x) + _L3G10_C
    return np.where(x < 0.0, x * _L3G10_G, _L3G10_A * _log10(x * _L3G10_B + 1.0))


def log3g10_decode(y, ei=None):
    y = _arr(y)
    return np.where(y < 0.0, y / _L3G10_G, (10.0 ** (y / _L3G10_A) - 1.0) / _L3G10_B) - _L3G10_C


# --------------------------------------------------------------------------- #
# Canon Log 3
# --------------------------------------------------------------------------- #
def clog3_encode(x, ei=None):
    """Canon Log 3 ("Full %" form, recorded full range). EI-independent.

    Sources: Canon USA "Canon Log Gamma Curves - Description of the Canon Log,
    Canon Log 2 and Canon Log 3 Gamma Curves", 2018-11-01,
    https://www.usa.canon.com/content/dam/canon-assets/white-papers/pro/white-paper-canon-log-gamma-curves.pdf
    (FAQ Q6/Q7: Canon Log is recorded full range; appendix formulas are images),
    and Canon USA "Deep Dive into HDR, Part 2" (2017), 18 % grey = code 351 (34.3 %),
    0 % black = code 128.  Constants (0.36726845, 14.98325, 0.12240537,
    1.9754798, 0.12512219, 0.12783901, cut +-0.014 on x/0.9) from the
    colour-science 0.4.7 'v1.2' implementation / Antler Post transcription of
    that appendix; they reproduce both official anchors (128.0, 351.3).
    Confidence: verified_secondary.
    """
    r = _arr(x) / 0.9
    lo = -0.36726845 * _log10(1.0 - 14.98325 * r) + 0.12783901
    mid = 1.9754798 * r + 0.12512219
    hi = 0.36726845 * _log10(14.98325 * r + 1.0) + 0.12240537
    return np.where(r < -0.014, lo, np.where(r <= 0.014, mid, hi))


def clog3_decode(y, ei=None):
    y = _arr(y)
    lo = -(10.0 ** ((0.12783901 - y) / 0.36726845) - 1.0) / 14.98325
    mid = (y - 0.12512219) / 1.9754798
    hi = (10.0 ** ((y - 0.12240537) / 0.36726845) - 1.0) / 14.98325
    r = np.where(y < 0.097465473, lo, np.where(y <= 0.15277891, mid, hi))
    return r * 0.9


# --------------------------------------------------------------------------- #
# Canon Log 2 (extension curve: real charts and real footage only)
# --------------------------------------------------------------------------- #
_CLOG2_A = 0.24136077
_CLOG2_B = 87.09937546
_CLOG2_C = 0.092864125


def clog2_encode(x, ei=None):
    """Canon Log 2 (normalised code value, same convention as clog3_encode). EI-independent.

    Source: Canon "Input Transform Version 202007 for EOS C300 Mark II" (2020), constants as
    transcribed in colour-science (canon.py, 'v1.2': 0.24136077, 87.09937546, 0.092864125; its
    doctest gives 0.18 -> 0.398254692).  The anchors below are computed from these constants, not
    read off Canon's table: 0 % black = code 95.0, 18 % grey = code 407.4 (39.2 % IRE on the 64-940
    scale), 90 % white = 575.2.  Confidence: verified_secondary.
    """
    r = _arr(x) / 0.9
    lo = -_CLOG2_A * _log10(1.0 - _CLOG2_B * r) + _CLOG2_C
    hi = _CLOG2_A * _log10(_CLOG2_B * r + 1.0) + _CLOG2_C
    return np.where(r < 0.0, lo, hi)


def clog2_decode(y, ei=None):
    y = _arr(y)
    lo = -(10.0 ** ((_CLOG2_C - y) / _CLOG2_A) - 1.0) / _CLOG2_B
    hi = (10.0 ** ((y - _CLOG2_C) / _CLOG2_A) - 1.0) / _CLOG2_B
    return np.where(y < _CLOG2_C, lo, hi) * 0.9


# --------------------------------------------------------------------------- #
# Blackmagic Film Generation 5
# --------------------------------------------------------------------------- #
_BMD_A = 0.08692876065491224
_BMD_B = 0.005494072432257808
_BMD_C = 0.5300133392291939
_BMD_D = 8.283605932402494
_BMD_E = 0.09246575342465753
_BMD_LIN_CUT = 0.005
_BMD_LOG_CUT = _BMD_D * _BMD_LIN_CUT + _BMD_E


def bmd_film_gen5_encode(x, ei=None):
    """Blackmagic Film Generation 5 (Gen 5 colour science, 2021+). EI-independent.

    Source: Blackmagic Design "Blackmagic Generation 5 Color Science - Technical
    Reference", May 2021 (shipped with the Blackmagic RAW SDK; text checked via
    https://www.repaire.net/forums/attachments/blackmagic-generation-5-color-science-technical-reference-pdf.51283/).
    Confidence: verified.  Mapping table: 0 -> 0.0924657534 (video levels 145),
    0.18 -> 0.3835616438 (400), 1 -> 0.5304896 (529), 222.86 -> 1.0 (940).
    Note the camera records 64 + 876*y in ProRes (video levels); RAW decoded in
    Resolve gives y directly.
    """
    x = _arr(x)
    return np.where(x < _BMD_LIN_CUT, _BMD_D * x + _BMD_E, _BMD_A * _ln(x + _BMD_B) + _BMD_C)


def bmd_film_gen5_decode(y, ei=None):
    y = _arr(y)
    return np.where(y < _BMD_LOG_CUT, (y - _BMD_E) / _BMD_D, np.exp((y - _BMD_C) / _BMD_A) - _BMD_B)


# --------------------------------------------------------------------------- #
# Apple Log
# --------------------------------------------------------------------------- #
_AL_R0, _AL_RT, _AL_C = -0.05641088, 0.01, 47.28711236
_AL_BETA, _AL_GAMMA, _AL_DELTA = 0.00964052, 0.08550479, 0.69336945
_AL_PT = _AL_C * (_AL_RT - _AL_R0) ** 2


def apple_log_encode(x, ei=None):
    """Apple Log (iPhone 15 Pro and later), Rec.2020 primaries. EI-independent.

    Source: Apple "Apple Log Profile White Paper" (Sept 2023), distributed via
    https://developer.apple.com/download/all/?q=Apple%20Log%20profile (Apple
    developer login required, not read directly).  Constants R0=-0.05641088,
    Rt=0.01, c=47.28711236, beta=0.00964052, gamma=0.08550479, delta=0.69336945
    as transcribed on ACESCentral (https://community.acescentral.com/t/iphone-15-to-have-true-log-aces-support/5176)
    and Antler Post (https://antlerpost.com/colour-spaces/AppleLog.html); Antler
    Post's camera-waveform table implies code10 = P*1023 (black 10.2 % IRE =
    code 154).  Confidence: verified_secondary.  Values below R0 encode to 0.
    """
    x = _arr(x)
    quad = _AL_C * (x - _AL_R0) ** 2
    hi = _AL_GAMMA * _log2(x + _AL_BETA) + _AL_DELTA
    return np.where(x >= _AL_RT, hi, np.where(x >= _AL_R0, quad, 0.0))


def apple_log_decode(y, ei=None):
    y = _arr(y)
    hi = 2.0 ** ((y - _AL_DELTA) / _AL_GAMMA) - _AL_BETA
    mid = np.sqrt(np.maximum(y, 0.0) / _AL_C) + _AL_R0
    return np.where(y >= _AL_PT, hi, np.where(y >= 0.0, mid, _AL_R0))


# --------------------------------------------------------------------------- #
# Panasonic V-Log
# --------------------------------------------------------------------------- #
def vlog_encode(x, ei=None):
    """Panasonic V-Log. EI-independent.

    Source: Panasonic "V-Log/V-Gamut Reference Manual" Rev.1.0, 2014-11-28,
    https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf
    Confidence: verified.  Anchors: 0 -> 128, 0.18 -> 433, 0.90 -> 602;
    VariCam 35 clips at 911, VariCam HS at 896 (all ISO).
    """
    x = _arr(x)
    return np.where(x < 0.01, 5.6 * x + 0.125, 0.241514 * _log10(x + 0.00873) + 0.598206)


def vlog_decode(y, ei=None):
    y = _arr(y)
    return np.where(y < 0.181, (y - 0.125) / 5.6, 10.0 ** ((y - 0.598206) / 0.241514) - 0.00873)


# --------------------------------------------------------------------------- #
# Fujifilm F-Log2
# --------------------------------------------------------------------------- #
_FL2 = dict(a=5.555556, b=0.064829, c=0.245281, d=0.384316, e=8.799461, f=0.092864,
            cut1=0.000889, cut2=0.100686685370811)


def flog2_encode(x, ei=None):
    """Fujifilm F-Log2. EI-independent.

    Source: Fujifilm "F-Log2 Data Sheet Ver.1.0",
    https://dl.fujifilm-x.com/support/lut/F-Log2_DataSheet_E_Ver.1.0.pdf
    Confidence: verified.  Anchors: 0 -> 95, 0.18 -> 400, 0.90 -> 570.
    """
    p = _FL2
    x = _arr(x)
    return np.where(x >= p["cut1"], p["c"] * _log10(p["a"] * x + p["b"]) + p["d"], p["e"] * x + p["f"])


def flog2_decode(y, ei=None):
    p = _FL2
    y = _arr(y)
    hi = 10.0 ** ((y - p["d"]) / p["c"]) / p["a"] - p["b"] / p["a"]
    return np.where(y >= p["cut2"], hi, (y - p["f"]) / p["e"])


# --------------------------------------------------------------------------- #
# Nikon N-Log
# --------------------------------------------------------------------------- #
def nlog_encode(x, ei=None):
    """Nikon N-Log (Rec.2020 primaries). EI-independent.

    Source: Nikon "N-Log Specification Document" Version 1.0.0, 2018-09-01,
    https://download.nikonimglib.com/archive3/hDCmK00m9JDI03RPruD74xpoU905/N-Log_Specification_(En)01.pdf
    Confidence: verified.  Formula is in 10-bit code values:
    code = 650*(y+0.0075)^(1/3) for y < 0.328, else 150*ln(y) + 619.
    The published inverse switches at code 452 while the forward switches at
    reflectance 0.328 (code 451.7 / 451.8), so reflectances in
    [0.328, 0.3285] do not round-trip exactly (error < 1e-4).
    """
    x = _arr(x)
    cube = 650.0 * np.cbrt(x + 0.0075)
    lg = 150.0 * _ln(x) + 619.0
    return np.where(x < 0.328, cube, lg) / 1023.0


def nlog_decode(y, ei=None):
    cv = _arr(y) * 1023.0
    return np.where(cv < 452.0, (cv / 650.0) ** 3 - 0.0075, np.exp((cv - 619.0) / 150.0))


# --------------------------------------------------------------------------- #
# Kinefinity KineLOG3
# --------------------------------------------------------------------------- #
KINELOG3 = dict(a=66.64, b=0.296, c=0.907136, d=0.092864, cut=-0.008239, s=0.017178)


def kinelog3_encode(x, ei=None):
    """Kinefinity KineLOG3 (Kinefinity Wide Gamut). EI-independent.

    Source: Kinefinity "KineLog3 Technical Specifications", Beijing, 2025-09-24,
    https://kinefinity.com/support/guides/kinelog3-technical-specifications
    Confidence: verified (constants and anchors read on the official page).
    y = log10(a*x + 1)*b*c + d for x >= cut, else (x - cut)/s.
    The two branches meet at y = 0 (x = cut); the published toe slope 1/s (58.2) differs from the
    log branch slope at cut (17.2), which only matters for codes below 0.
    """
    p, x = KINELOG3, _arr(x)
    lg = _log10(p["a"] * x + 1.0) * p["b"] * p["c"] + p["d"]
    return np.where(x >= p["cut"], lg, (x - p["cut"]) / p["s"])


def kinelog3_decode(y, ei=None):
    p, y = KINELOG3, _arr(y)
    return np.where(y >= 0.0, (10.0 ** ((y - p["d"]) / p["b"] / p["c"]) - 1.0) / p["a"], y * p["s"] + p["cut"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurveSpec:
    id: str
    encode: Callable
    decode: Callable
    gamut_id: Optional[str]          # gamut used for decoding to XYZ
    gamut_provisional: bool          # True when the native gamut is not officially known
    ei_dependence: str               # "curve" | "clip_only" | "none"
    default_ei: Optional[int]
    formula: str
    source: str
    doc_version: str
    confidence: str                  # verified | verified_secondary | unverified
    levels_note: str = ""
    clip_cv10_by_ei: Optional[dict] = None   # documented sensor clipping level (10-bit)
    clip_note: str = ""
    round_trip_exclusions: tuple = ()        # linear intervals where the published pair is not inverse


CURVE_SPECS: dict[str, CurveSpec] = {s.id: s for s in [
    CurveSpec(
        "arri_logc3", logc3_encode, logc3_decode, "arri_awg3", False, "curve", 800,
        "y = c*log10(a*x + b) + d for x > cut, else e*x + f. EI800: cut 0.010591, a 5.555556, b 0.052272, "
        "c 0.247190, d 0.385537, e 5.367655, f 0.092809 (c falls from 0.269036 @EI160 to 0.237781 @EI1600; "
        "grey is 0.391 at every EI)",
        "https://www.arri.com/en/learn-help/learn-help-camera-system/technical-downloads (ALEXA Log C Curve - Usage in VFX)",
        "ARRI, H. Brendel, rev. 2017-03-09", "verified",
        clip_cv10_by_ei={ei: v * 1023.0 for ei, v in LOGC3_CLIP_CODE.items()},
        clip_note="ARRI clipping-level table (sensor saturation, Log C value per EI: 0.8128 @EI160 ... 1.0 @EI>=1600)"),
    CurveSpec(
        "arri_logc4", logc4_encode, logc4_decode, "arri_awg4", False, "none", None,
        "y = ((log2(a*x + 64) - 6)/14)*b + c for x >= t, else (x - t)/s; a = (2^18 - 16)/117.45, "
        "b = 928/1023, c = 95/1023, s = 7*ln2*2^(7 - 14c/b)/(a*b), t = (2^(6 - 14c/b) - 64)/a",
        "https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf",
        "ARRI LogC4 Specification, history to 2022-11-08", "verified",
        levels_note="ARRI tables give both: 10-bit full (black 95, grey 285) and 10-bit legal (145 / 308).",
        clip_note="Curve reaches 1.0 at 469.8; ALEXA 35 sensor clip moves with EI (encoding limit hit above EI3200)"),
    CurveSpec(
        "sony_slog3", slog3_encode, slog3_decode, "sony_sgamut3cine", False, "none", None,
        "y = (420 + 261.5*log10((x + 0.01)/0.19))/1023 for x >= 0.01125, else (95 + x*(171.2102946929 - 95)/0.01125)/1023",
        "https://pro.sony/s3/cms-static-content/file/12/1237494271406.pdf",
        "Sony Technical Summary S-Gamut3.Cine/S-Log3 & S-Gamut3/S-Log3 (2014)", "verified",
        levels_note="Native gamut can also be S-Gamut3 (same curve).",
        clip_note="Sony gives no per-camera clip code; curve reaches 1.0 at 38.4"),
    CurveSpec(
        "dji_dlog", dlog_encode, dlog_decode, "dji_dgamut", False, "clip_only", 800,
        "y = 0.256663*log10(0.9892*x + 0.0108) + 0.584555 for x > 0.0078, else 6.025*x + 0.0929",
        "https://dl.djicdn.com/downloads/zenmuse+x7/20171010/D-Log_D-Gamut_Whitepaper.pdf ; "
        "https://dl.djicdn.com/downloads/DJI_Ronin_4D/X9_D_Log_D_Gamut_Whitepaper.pdf",
        "DJI D-Log/D-Gamut whitepaper Rev1.0 2017-09-29 (X7) and Rev1.0 2022-02 (X9 6K/8K)", "verified",
        clip_cv10_by_ei=DLOG_X9_CLIP_CV10,
        clip_note="Zenmuse X9 (Ronin 4D / Inspire 3) 6K table: 948 at native EI800/5000 and above, "
                  "921 @640, 895 @500, 868 @400, 842 @320, 816 @250, 789 @200. Not documented for X7 / drones.",
        round_trip_exclusions=((0.00763, 0.00797),)),
    CurveSpec(
        "dji_dlogm", dlogm_encode, dlogm_decode, "rec2020", True, "none", None,
        "No official formula. Community fit (Mavic 3 Pro): L = 0.0558909*T(2^(7.420092*y - 0.983729) - 0.736169), "
        "T(t) = t for t >= 0.174595 else 0.438598*t + 0.098018; grey anchor (40 % -> 0.18) is the fit's assumption",
        "https://github.com/thatcherfreeman/dwg-transforms (DJI Mavic 3 Pro D-Log M to DWG.dctl)",
        "community empirical fit, 2024; no DJI document", "unverified",
        levels_note="Native gamut undocumented by DJI; rec2020 is a provisional assumption for decoding.",
        clip_note="Fit reaches code 1.0 at 4.8 linear (other bodies' fits: 2.3-3.9)"),
    CurveSpec(
        "red_log3g10", log3g10_encode, log3g10_decode, "red_rwg", False, "none", None,
        "y = 0.224282*log10(155.975327*(x + 0.01) + 1) for x >= -0.01, else 15.1927*(x + 0.01)",
        "https://docs.red.com/955-0187/PDF/915-0187%20Rev-C%20%20%20RED%20OPS%2C%20White%20Paper%20on%20REDWideGamutRGB%20and%20Log3G10.pdf",
        "RED 915-0187 Rev-C, 2017-11-06", "verified",
        clip_note="Curve reaches 1.0 at 184.32 (10 stops above grey); sensor clip camera/ISO dependent, undocumented"),
    CurveSpec(
        "canon_clog3", clog3_encode, clog3_decode, "canon_cinemagamut", False, "none", None,
        "r = x/0.9; y = 0.36726845*log10(14.98325*r + 1) + 0.12240537 for r > 0.014; 1.9754798*r + 0.12512219 "
        "for |r| <= 0.014; -0.36726845*log10(1 - 14.98325*r) + 0.12783901 for r < -0.014",
        "https://www.usa.canon.com/content/dam/canon-assets/white-papers/pro/white-paper-canon-log-gamma-curves.pdf ; "
        "http://downloads.canon.com/nw/learn/white-papers/cinema-eos/White_Paper_Deep-Dive-HDR-Part2.pdf",
        "Canon USA white paper 2018-11-01 (+ HDR Deep Dive Part 2, 2017)", "verified_secondary",
        levels_note="Canon records Canon Log full range (FAQ Q6/Q7).",
        clip_note="Sensor saturation = 1600 % (r = 16) -> code 0.997 (~1020) on C300 Mk II at base ISO"),
    CurveSpec(
        "canon_clog2", clog2_encode, clog2_decode, "canon_cinemagamut", False, "none", None,
        "r = x/0.9; y = 0.24136077*log10(87.09937546*r + 1) + 0.092864125 for r >= 0; "
        "-0.24136077*log10(1 - 87.09937546*r) + 0.092864125 for r < 0",
        "Canon Input Transform Version 202007 for EOS C300 Mark II (constants via colour-science canon.py v1.2)",
        "Canon Input Transform 202007 (2020)", "verified_secondary",
        levels_note="Same code-value convention as Canon Log 3.",
        clip_note="Curve reaches 1.0 at r = 65.8 (about 8.4 stops above grey); sensor clip camera/ISO dependent"),
    CurveSpec(
        "bmd_film_gen5", bmd_film_gen5_encode, bmd_film_gen5_decode, "bmd_wg_gen5", False, "none", None,
        "y = 0.08692876*ln(x + 0.00549407) + 0.53001334 for x >= 0.005, else 8.28360593*x + 0.09246575",
        "Blackmagic RAW SDK: 'Blackmagic Generation 5 Color Science Technical Reference.pdf' (mirror: "
        "https://www.repaire.net/forums/attachments/blackmagic-generation-5-color-science-technical-reference-pdf.51283/)",
        "Blackmagic Design technical reference, May 2021", "verified",
        levels_note="BMD camera ProRes stores video levels 64 + 876*y: black 145, grey 400, 1.0 -> 940. "
                    "RAW decoded in Resolve yields y directly (black 95, grey 392).",
        clip_note="Curve reaches 1.0 at 222.86 (10.27 stops above grey)"),
    CurveSpec(
        "apple_log", apple_log_encode, apple_log_decode, "rec2020", False, "none", None,
        "y = 0.08550479*log2(x + 0.00964052) + 0.69336945 for x >= 0.01; 47.28711236*(x + 0.05641088)^2 "
        "for -0.05641088 <= x < 0.01; 0 below",
        "https://developer.apple.com/download/all/?q=Apple%20Log%20profile (login) ; "
        "https://antlerpost.com/colour-spaces/AppleLog.html",
        "Apple Log Profile White Paper, Sept 2023 (via transcription)", "verified_secondary",
        clip_note="Curve reaches 1.0 at ~12.0 linear"),
    CurveSpec(
        "apple_log2", apple_log_encode, apple_log_decode, "apple_wide_gamut", False, "none", None,
        "identical to Apple Log (Apple Log 2 uses the Apple Log transfer function): "
        "y = 0.08550479*log2(x + 0.00964052) + 0.69336945 for x >= 0.01; 47.28711236*(x + 0.05641088)^2 "
        "for -0.05641088 <= x < 0.01; 0 below; gamut Apple Wide Gamut",
        "Apple Log 2 White Paper (developer.apple.com, login; copy attached to colour-science/colour issue #1428)",
        "Apple Log 2 White Paper, Sept 2025, Part 028-00834 v1.1", "verified",
        clip_note="Curve reaches 1.0 at ~12.0 linear"),
    CurveSpec(
        "panasonic_vlog", vlog_encode, vlog_decode, "panasonic_vgamut", False, "none", None,
        "y = 0.241514*log10(x + 0.00873) + 0.598206 for x >= 0.01, else 5.6*x + 0.125",
        "https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf",
        "Panasonic V-Log/V-Gamut Reference Manual Rev.1.0, 2014-11-28", "verified",
        clip_note="VariCam 35 clips at 911, VariCam HS at 896 (documented); Lumix bodies undocumented"),
    CurveSpec(
        "fuji_flog2", flog2_encode, flog2_decode, "fuji_fgamut", False, "none", None,
        "y = 0.245281*log10(5.555556*x + 0.064829) + 0.384316 for x >= 0.000889, else 8.799461*x + 0.092864",
        "https://dl.fujifilm-x.com/support/lut/F-Log2_DataSheet_E_Ver.1.0.pdf",
        "Fujifilm F-Log2 Data Sheet Ver.1.0", "verified",
        clip_note="Curve reaches 1.0 at 58.2; camera clip undocumented"),
    CurveSpec(
        "fuji_flog2c", flog2_encode, flog2_decode, "fuji_fgamut_c", False, "none", None,
        "identical to F-Log2 (F-Log2 C data sheet: 'the gamma curve of F-Log2 C is identical to that of F-Log2'): "
        "y = 0.245281*log10(5.555556*x + 0.064829) + 0.384316 for x >= 0.000889, else 8.799461*x + 0.092864; "
        "gamut F-Gamut C",
        "https://dl.fujifilm-x.com/support/lut/F-Log2C_DataSheet_E_Ver.1.0.pdf",
        "Fujifilm F-Log2 C Data Sheet Ver.1.0", "verified",
        clip_note="Curve reaches 1.0 at 58.2; camera clip undocumented"),
    CurveSpec(
        "nikon_nlog", nlog_encode, nlog_decode, "rec2020", False, "none", None,
        "code10 = 650*(x + 0.0075)^(1/3) for x < 0.328, else 150*ln(x) + 619 (y = code10/1023)",
        "https://download.nikonimglib.com/archive3/hDCmK00m9JDI03RPruD74xpoU905/N-Log_Specification_(En)01.pdf",
        "Nikon N-Log Specification Document v1.0.0, 2018-09-01", "verified",
        clip_note="Curve reaches 1.0 at 14.8 linear; camera clip undocumented",
        round_trip_exclusions=((0.328, 0.3285),)),
    CurveSpec(
        "kine_log3", kinelog3_encode, kinelog3_decode, "kinefinity_kwg", False, "none", None,
        "y = 0.268512*log10(66.64*x + 1) + 0.092864 for x >= -0.008239 (b*c = 0.296*0.907136), "
        "else (x + 0.008239)/0.017178",
        "https://kinefinity.com/support/guides/kinelog3-technical-specifications",
        "Kinefinity KineLog3 Technical Specifications, 2025-09-24", "verified",
        clip_note="Curve reaches 1.0 at about 36 linear; camera clip undocumented"),
]}

# CURVE_IDS = curves in the benchmark; CURVE_SPECS also holds extension and dropped curves
assert set(CURVE_IDS) <= set(CURVE_SPECS), (set(CURVE_IDS) - set(CURVE_SPECS))


def _spec(curve_id: str) -> CurveSpec:
    try:
        return CURVE_SPECS[curve_id]
    except KeyError:
        raise KeyError(f"no transfer function for curve id {curve_id!r}; known: {CURVE_IDS}") from None


def encode(curve_id: str, linear, ei=None):
    """Scene-linear (0.18 = grey) -> normalised full-range code value [0, 1]."""
    y = _spec(curve_id).encode(linear, ei)
    return _out(linear, y)


def decode(curve_id: str, code, ei=None):
    """Normalised full-range code value -> scene-linear (inverse of ``encode``)."""
    x = _spec(curve_id).decode(code, ei)
    return _out(code, x)


def round_trip_exclusions(curve_id: str) -> tuple:
    """Linear intervals where the published encode/decode pair is not an exact inverse."""
    return _spec(curve_id).round_trip_exclusions


def resolve_ei(curve_id: str, ei=None) -> Optional[int]:
    spec = _spec(curve_id)
    if curve_id == "arri_logc3":
        return logc3_resolve_ei(ei)
    if spec.ei_dependence == "clip_only":
        return int(ei) if ei is not None else spec.default_ei
    return None


# --------------------------------------------------------------------------- #
# Gamuts
# --------------------------------------------------------------------------- #
D65_XY = (0.3127, 0.3290)
D65_XYZ = np.array([D65_XY[0] / D65_XY[1], 1.0, (1.0 - D65_XY[0] - D65_XY[1]) / D65_XY[1]])

_REC2020 = {"R": (0.708, 0.292), "G": (0.170, 0.797), "B": (0.131, 0.046)}

GAMUT_PRIMARIES: dict[str, dict] = {
    "arri_awg3": {"R": (0.6840, 0.3130), "G": (0.2210, 0.8480), "B": (0.0861, -0.1020),
                  "source": "ARRI ALEXA Log C Curve - Usage in VFX (2017)", "confidence": "verified"},
    "arri_awg4": {"R": (0.7347, 0.2653), "G": (0.1424, 0.8576), "B": (0.0991, -0.0308),
                  "source": "ARRI LogC4 Specification (2022)", "confidence": "verified"},
    "sony_sgamut3cine": {"R": (0.766, 0.275), "G": (0.225, 0.800), "B": (0.089, -0.087),
                         "source": "Sony S-Log3 technical summary (2014)", "confidence": "verified"},
    "sony_sgamut3": {"R": (0.730, 0.280), "G": (0.140, 0.855), "B": (0.100, -0.050),
                     "source": "Sony S-Log3 technical summary (2014)", "confidence": "verified"},
    "dji_dgamut": {"R": (0.71, 0.31), "G": (0.21, 0.88), "B": (0.09, -0.08),
                   "source": "DJI D-Log/D-Gamut whitepapers (2017, 2022)", "confidence": "verified"},
    "red_rwg": {"R": (0.780308, 0.304253), "G": (0.121595, 1.493994), "B": (0.095612, -0.084589),
                "source": "RED 915-0187 Rev-C (2017)", "confidence": "verified"},
    "canon_cinemagamut": {"R": (0.74, 0.27), "G": (0.17, 1.14), "B": (0.08, -0.10),
                          "source": "Canon Cinema Gamut (EOS C300 Mk II IDT docs) via colour-science 0.4.7",
                          "confidence": "verified_secondary"},
    "bmd_wg_gen5": {"R": (0.7177215, 0.3171181), "G": (0.2280410, 0.8615690), "B": (0.1005841, -0.0820452),
                    "source": "BMD Generation 5 Color Science reference (2021); BMD white (0.3127170, 0.3290312) "
                              "replaced by D65", "confidence": "verified"},
    "bmd_wg_gen4": {"R": (0.7177215, 0.3171181), "G": (0.2280410, 0.8615690), "B": (0.1005841, -0.0820452),
                    "source": "assumed identical to Gen 5 Blackmagic Wide Gamut; no separate Gen 4 primaries found",
                    "confidence": "unverified"},
    "panasonic_vgamut": {"R": (0.730, 0.280), "G": (0.165, 0.840), "B": (0.100, -0.030),
                         "source": "Panasonic V-Log/V-Gamut Reference Manual (2014)", "confidence": "verified"},
    "fuji_fgamut": dict(_REC2020, source="Fujifilm F-Log2 Data Sheet (F-Gamut = BT.2020)", confidence="verified"),
    "rec2020": dict(_REC2020, source="ITU-R BT.2020", confidence="verified"),
    "rec709": {"R": (0.64, 0.33), "G": (0.30, 0.60), "B": (0.15, 0.06),
               "source": "ITU-R BT.709", "confidence": "verified"},
    "kinefinity_kwg": {"R": (0.7571, 0.2282), "G": (0.2139, 1.1480), "B": (0.0536, -0.2236),
                       "source": "Kinefinity KineLog3 Technical Specifications (2025-09-24)", "confidence": "verified"},
    # only in digests that include F-Log2 C / Apple Log 2 (see _digest_data)
    "fuji_fgamut_c": {"R": (0.73470, 0.26530), "G": (0.02630, 0.97370), "B": (0.11730, -0.02240),
                      "source": "Fujifilm F-Log2 C Data Sheet Ver.1.0 (2024)", "confidence": "verified"},
    "apple_wide_gamut": {"R": (0.725, 0.301), "G": (0.221, 0.814), "B": (0.068, -0.076),
                         "source": "Apple Log 2 White Paper, Sept 2025 (Part 028-00834 v1.1)", "confidence": "verified"},
}
EXT2_GAMUTS = {"fuji_fgamut_c": "fuji_flog2c", "apple_wide_gamut": "apple_log2"}   # gamut -> the curve that uses it
assert set(GAMUT_PRIMARIES) == set(GAMUTS), (set(GAMUT_PRIMARIES) ^ set(GAMUTS))

# Vendor-printed RGB->XYZ matrices, for cross-checking the derivation.
OFFICIAL_RGB_TO_XYZ: dict[str, list[list[float]]] = {
    "arri_awg3": [[0.638008, 0.214704, 0.097744], [0.291954, 0.823841, -0.115795], [0.002798, -0.067034, 1.153294]],
    "arri_awg4": [[0.704858320407232064, 0.129760295170463003, 0.115837311473976537],
                  [0.254524176404027025, 0.781477732712002049, -0.036001909116029039],
                  [0.0, 0.0, 1.089057750759878429]],
    "dji_dgamut": [[0.6482, 0.1940, 0.1082], [0.2830, 0.8132, -0.0962], [-0.0183, -0.0832, 1.1903]],
    "red_rwg": [[0.735275, 0.068609, 0.146571], [0.286694, 0.842979, -0.129673], [-0.079681, -0.347343, 1.516081]],
    "panasonic_vgamut": [[0.679644, 0.152211, 0.118600], [0.260686, 0.774894, -0.035580],
                         [-0.009310, -0.004612, 1.102980]],
    "bmd_wg_gen5": [[0.606530, 0.220408, 0.123479], [0.267989, 0.832731, -0.100720],
                    [-0.029442, -0.086611, 1.204861]],
    "kinefinity_kwg": [[0.6879, 0.1979, 0.0646], [0.2073, 1.0622, -0.2696], [0.0134, -0.3349, 1.4106]],
}


def gamut_primaries(gamut_id: str) -> dict:
    """{'R': (x, y), 'G': ..., 'B': ..., 'W': D65, 'source': ..., 'confidence': ...}."""
    p = dict(GAMUT_PRIMARIES[gamut_id])
    p["W"] = D65_XY
    return p


def _npm(gamut_id: str) -> np.ndarray:
    p = GAMUT_PRIMARIES[gamut_id]
    cols = []
    for k in ("R", "G", "B"):
        x, y = p[k]
        cols.append([x / y, 1.0, (1.0 - x - y) / y])
    P = np.array(cols).T
    S = np.linalg.solve(P, D65_XYZ)
    return P * S


def gamut_to_xyz(gamut_id: str) -> np.ndarray:
    """Linear RGB (gamut) -> CIE 1931 XYZ, derived from primaries + D65, 6 decimals."""
    return np.round(_npm(gamut_id), 6)


def xyz_to_gamut(gamut_id: str) -> np.ndarray:
    """CIE 1931 XYZ -> linear RGB (gamut), inverse of the unrounded NPM, 6 decimals."""
    return np.round(np.linalg.inv(_npm(gamut_id)), 6)


def matrix_crosscheck() -> dict[str, float]:
    """Max |derived - vendor matrix| for gamuts with a published matrix."""
    return {g: float(np.max(np.abs(_npm(g) - np.array(m)))) for g, m in OFFICIAL_RGB_TO_XYZ.items()}


def curve_gamut(curve_id: str) -> Optional[str]:
    return _spec(curve_id).gamut_id


# CIE 1931 2-degree spectral locus, 380-700 nm in 5 nm steps (x, y).
SPECTRAL_LOCUS_XY = np.array([
    (0.1741, 0.0050), (0.1740, 0.0050), (0.1738, 0.0049), (0.1736, 0.0049), (0.1733, 0.0048),
    (0.1730, 0.0048), (0.1726, 0.0048), (0.1721, 0.0048), (0.1714, 0.0051), (0.1703, 0.0058),
    (0.1689, 0.0069), (0.1669, 0.0086), (0.1644, 0.0109), (0.1611, 0.0138), (0.1566, 0.0177),
    (0.1510, 0.0227), (0.1440, 0.0297), (0.1355, 0.0399), (0.1241, 0.0578), (0.1096, 0.0868),
    (0.0913, 0.1327), (0.0687, 0.2007), (0.0454, 0.2950), (0.0235, 0.4127), (0.0082, 0.5384),
    (0.0039, 0.6548), (0.0139, 0.7502), (0.0389, 0.8120), (0.0743, 0.8338), (0.1142, 0.8262),
    (0.1547, 0.8059), (0.1929, 0.7816), (0.2296, 0.7543), (0.2658, 0.7243), (0.3016, 0.6923),
    (0.3373, 0.6589), (0.3731, 0.6245), (0.4087, 0.5896), (0.4441, 0.5547), (0.4788, 0.5202),
    (0.5125, 0.4866), (0.5448, 0.4544), (0.5752, 0.4242), (0.6029, 0.3965), (0.6270, 0.3725),
    (0.6482, 0.3514), (0.6658, 0.3340), (0.6801, 0.3197), (0.6915, 0.3083), (0.7006, 0.2993),
    (0.7079, 0.2920), (0.7140, 0.2859), (0.7190, 0.2809), (0.7230, 0.2770), (0.7260, 0.2740),
    (0.7283, 0.2717), (0.7300, 0.2700), (0.7311, 0.2689), (0.7320, 0.2680), (0.7327, 0.2673),
    (0.7334, 0.2666), (0.7340, 0.2660), (0.7344, 0.2656), (0.7346, 0.2654), (0.7347, 0.2653),
])


def relative_purity(xy: np.ndarray, white=D65_XY) -> np.ndarray:
    """Distance white->xy divided by distance white->locus boundary along the same ray.

    The boundary is the spectral locus closed by the purple line.  Values > 1
    are outside the locus (imaginary colours).  ``xy`` has shape (..., 2).
    """
    xy = np.asarray(xy, dtype=np.float64)
    shp = xy.shape[:-1]
    d = xy.reshape(-1, 2) - np.asarray(white)
    poly = SPECTRAL_LOCUS_XY
    a = poly - np.asarray(white)
    b = np.roll(poly, -1, axis=0) - np.asarray(white)   # last edge = purple line
    e = b - a
    # Solve white + t*d = a + s*e  ->  t*d - s*e = a
    den = d[:, None, 0] * (-e[None, :, 1]) - d[:, None, 1] * (-e[None, :, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[None, :, 0] * (-e[None, :, 1]) - a[None, :, 1] * (-e[None, :, 0])) / den
        s = (d[:, None, 0] * a[None, :, 1] - d[:, None, 1] * a[None, :, 0]) / den
    ok = (np.abs(den) > 1e-15) & (s >= 0.0) & (s <= 1.0) & (t > 0.0)
    t = np.where(ok, t, np.inf)
    tmin = t.min(axis=1)
    norm = np.linalg.norm(d, axis=1)
    pur = np.where(norm < 1e-12, 0.0, 1.0 / tmin)
    return pur.reshape(shp)


# --------------------------------------------------------------------------- #
# Expected code values
# --------------------------------------------------------------------------- #


def expected_code_values(curve_id: str, ei=None) -> dict:
    """Whitepaper code values for one curve.

    Keys: ``black`` / ``grey18`` / ``white90`` (normalised codes for linear
    0 / 0.18 / 0.90); ``clip`` = linear value at which the signal clips
    (documented sensor clip for this EI when the vendor publishes one, else
    the linear value at code 1.0); ``clip_code``; ``linear_at_code_1``;
    ``stops_above_grey_at_clip``; ``slope_per_stop_at_grey`` (10-bit CV per
    stop, symmetric +-0.5 stop around 0.18); ``cv10`` (integer 10-bit values,
    ``clip`` = clip code); ``cv10_float``.
    """
    spec = _spec(curve_id)
    ei_used = resolve_ei(curve_id, ei)

    def enc(v: float) -> float:
        return float(spec.encode(v, ei_used))

    black, grey, white = enc(0.0), enc(0.18), enc(0.90)
    lin_at_1 = float(spec.decode(1.0, ei_used))
    clip_code, clip_source = 1.0, "code 1.0 (no documented sensor clip for this curve/EI)"
    if spec.clip_cv10_by_ei and ei_used in spec.clip_cv10_by_ei:
        clip_code = spec.clip_cv10_by_ei[ei_used] / 1023.0
        clip_source = f"documented sensor clip @EI{ei_used}: {spec.clip_note}"
    clip_lin = float(spec.decode(clip_code, ei_used))
    slope = (enc(0.18 * 2 ** 0.5) - enc(0.18 * 2 ** -0.5)) * 1023.0
    return {
        "curve_id": curve_id,
        "ei": ei_used,
        "ei_dependence": spec.ei_dependence,
        "black": black,
        "grey18": grey,
        "white90": white,
        "clip": clip_lin,
        "clip_code": clip_code,
        "clip_source": clip_source,
        "linear_at_code_1": lin_at_1,
        "stops_above_grey_at_clip": math.log2(clip_lin / 0.18),
        "slope_per_stop_at_grey": slope,
        "cv10": {"black": int(round(black * 1023)), "grey18": int(round(grey * 1023)),
                 "white90": int(round(white * 1023)), "clip": int(round(clip_code * 1023))},
        "cv10_float": {"black": round(black * 1023, 2), "grey18": round(grey * 1023, 2),
                       "white90": round(white * 1023, 2), "clip": round(clip_code * 1023, 2)},
        "gamut_id": spec.gamut_id,
        "gamut_provisional": spec.gamut_provisional,
        "confidence": spec.confidence,
        "levels_note": spec.levels_note,
    }


def expected_table_row(curve_id: str, ei=None) -> dict:
    """Flat row used in evidence.jsonl ``expected_table`` (10-bit integers)."""
    e = expected_code_values(curve_id, ei)
    row = dict(e["cv10"])
    row["slope_per_stop"] = round(e["slope_per_stop_at_grey"], 1)
    row["clip_linear"] = round(e["clip"], 3)
    if e["ei"] is not None:
        row["ei"] = e["ei"]
    return row


# --------------------------------------------------------------------------- #
# Digest (prompt tier 2 "whitepaper summary" + human cheat sheet)
# --------------------------------------------------------------------------- #
_SHORT = {
    "arri_logc3": "ARRI LogC3", "arri_logc4": "ARRI LogC4", "sony_slog3": "Sony S-Log3",
    "dji_dlog": "DJI D-Log", "dji_dlogm": "DJI D-Log M", "red_log3g10": "RED Log3G10",
    "canon_clog3": "Canon Log 3", "canon_clog2": "Canon Log 2", "bmd_film_gen5": "BMD Film Gen 5", "apple_log": "Apple Log",
    "panasonic_vlog": "Panasonic V-Log", "fuji_flog2": "Fujifilm F-Log2", "nikon_nlog": "Nikon N-Log",
    "fuji_flog2c": "Fujifilm F-Log2 C", "apple_log2": "Apple Log 2",
    "kine_log3": "Kinefinity KineLOG3",
}
_EI_TEXT = {"curve": "yes - curve changes with EI", "clip_only": "clip level only", "none": "no"}


def _fmt_xy(p) -> str:
    return f"({p[0]:.4f}, {p[1]:.4f})"


def _digest_data(extra: tuple = ()) -> dict:
    """extra: extension curve ids (taxonomy status "v1_ext") to include; default = the 14-option set."""
    curves = {}
    for cid in list(CURVE_IDS) + [c for c in extra if c not in CURVE_IDS]:
        spec = CURVE_SPECS[cid]
        e = expected_code_values(cid)
        tax = CURVES[cid]
        curves[cid] = {
            "name": tax.name,
            "native_gamut": spec.gamut_id if not spec.gamut_provisional else None,
            "decode_gamut": spec.gamut_id,
            "formula": spec.formula,
            "cv10": e["cv10"],
            "cv10_float": e["cv10_float"],
            "ei_default": e["ei"],
            "slope_per_stop_at_grey_cv10": round(e["slope_per_stop_at_grey"], 1),
            "linear_at_code_1": round(e["linear_at_code_1"], 2),
            "stops_above_grey_at_code_1": round(math.log2(e["linear_at_code_1"] / 0.18), 2),
            "ei_dependence": spec.ei_dependence,
            "clip_note": spec.clip_note,
            "levels_note": spec.levels_note,
            "source": spec.source,
            "doc_version": spec.doc_version,
            "confidence": spec.confidence,
        }
    gamuts = {}
    for gid in GAMUT_PRIMARIES:
        if gid in EXT2_GAMUTS and EXT2_GAMUTS[gid] not in curves:   # keeps the 14- / 15-option digests unchanged
            continue
        p = gamut_primaries(gid)
        gamuts[gid] = {"name": GAMUTS[gid], "R": p["R"], "G": p["G"], "B": p["B"], "W": p["W"],
                       "rgb_to_xyz": gamut_to_xyz(gid).tolist(), "confidence": p["confidence"]}
    logc3_ei = {ei: {"clip_cv10": round(LOGC3_CLIP_CODE[ei] * 1023),
                     "white90_cv10": round(float(logc3_encode(0.9, ei)) * 1023),
                     "slope_per_stop": round((float(logc3_encode(0.18 * 2 ** 0.5, ei))
                                              - float(logc3_encode(0.18 * 2 ** -0.5, ei))) * 1023, 1)}
                for ei in LOGC3_PARAMS}
    return {"version": "v0", "conventions": {
        "linear": "relative scene-linear, 0.18 = 18% grey, 0.90 = 90% white",
        "code": "10-bit full-range code value = normalised value * 1023",
        "white_point": "D65 (0.3127, 0.3290) for every gamut"},
        "curves": curves, "gamuts": gamuts, "logc3_by_ei": logc3_ei,
        "dlog_x9_clip_by_ei": DLOG_X9_CLIP_CV10}


def _digest_markdown(data: dict, part: str = "full") -> str:
    """part = "spec": the manufacturers' specification only (sections 1-5);
    part = "full": the same plus the reading guidance written by us (sections 6-8)."""
    C = data["curves"]
    L: list[str] = []
    add = L.append
    # The digest is prompt text: this header line is kept as it was when the benchmark was run, so the
    # regenerated files stay byte-identical to cameralogbench/reference/*.md
    add("# Log encoding reference digest (auto-generated by bench/curves.py - do not edit by hand)")
    add("")
    if part == "full":
        add("Sections 1-5 restate the manufacturers' published specifications. Sections 6-8 are reading guidance "
            "written by the benchmark authors.")
    else:
        add("All sections restate the manufacturers' published specifications.")
    add("")
    add("Conventions: code values are 10-bit full range (normalised value x 1023). Scene-linear input: "
        "0.18 = 18% grey card, 0.90 = 90% white card. All gamuts use a D65 white point. "
        "\"clip @1.0\" = scene-linear value that reaches code 1023; real cameras usually clip lower.")
    add("")
    add("## 1. Key code values (10-bit)")
    add("")
    # no "typical cameras" column: it would hand the model a brand -> curve shortcut
    add("| Encoding | Native gamut | 0% black | 18% grey | 90% white | CV/stop @grey | clip @1.0 (stops > grey) | EI-dependent |")
    add("|---|---|---|---|---|---|---|---|")
    for cid, c in C.items():
        g = GAMUTS[c["native_gamut"]] if c["native_gamut"] else "undocumented"
        flag = " (unverified)" if c["confidence"] == "unverified" else ""
        add(f"| {_SHORT[cid]}{flag} | {g} | {c['cv10']['black']} | {c['cv10']['grey18']} | {c['cv10']['white90']} | "
            f"{c['slope_per_stop_at_grey_cv10']:.0f} | {c['linear_at_code_1']:g} ({c['stops_above_grey_at_code_1']:+.1f}) | "
            f"{_EI_TEXT[c['ei_dependence']]} |")
    add("")
    add("LogC3 values are for EI800; grey (400) and black (95) are identical at every EI.")
    add("")
    add("## 2. Formulas (x = scene-linear, y = normalised code)")
    add("")
    for cid, c in C.items():
        add(f"- **{_SHORT[cid]}**: {c['formula']}.")
    add("")
    add("## 3. Native gamut primaries (CIE 1931 xy, white D65)")
    add("")
    add("| Gamut | Red | Green | Blue | Used by |")
    add("|---|---|---|---|---|")
    users: dict[str, list[str]] = {}
    for cid, c in C.items():
        if c["native_gamut"]:
            users.setdefault(c["native_gamut"], []).append(_SHORT[cid])
    users.setdefault("sony_sgamut3", []).append("Sony S-Log3 (alt.)")
    for gid in ("arri_awg3", "arri_awg4", "sony_sgamut3cine", "sony_sgamut3", "dji_dgamut", "red_rwg",
                "canon_cinemagamut", "bmd_wg_gen5", "kinefinity_kwg", "rec2020", "rec709",
                *[g for g in EXT2_GAMUTS if g in data["gamuts"]]):
        g = data["gamuts"][gid]
        who = ", ".join(users.get(gid, [])) or ("display reference" if gid == "rec709" else "")
        if gid == "rec2020":
            who = "Apple Log, Nikon N-Log, Fujifilm F-Log2 (F-Gamut)"
        add(f"| {g['name']} | {_fmt_xy(g['R'])} | {_fmt_xy(g['G'])} | {_fmt_xy(g['B'])} | {who} |")
    add("")
    official = [cid for cid in C if C[cid]["confidence"] != "unverified"]

    def ranked(key: str, ids) -> str:
        items = sorted(ids, key=lambda c: C[c]["cv10"][key])
        return " < ".join(f"{_SHORT[c]} {C[c]['cv10'][key]}" for c in items)

    fam = {"~95": [], "~128": [], "~154": []}
    for cid in official:
        b = C[cid]["cv10"]["black"]
        fam["~95" if b < 110 else "~128" if b < 140 else "~154"].append(f"{_SHORT[cid]} {b}")
    slopes = sorted(official, key=lambda c: C[c]["slope_per_stop_at_grey_cv10"])
    s3 = data["logc3_by_ei"]
    add("## 4. Documented EI-dependent values")
    add("")
    add("ARRI LogC3 (the only curve whose shape changes with EI; black 95 and grey 400 are fixed):")
    add("")
    add("| EI | sensor clip | 90% white | CV/stop @grey |")
    add("|---|---|---|---|")
    for ei, v in s3.items():
        add(f"| {ei} | {v['clip_cv10']} | {v['white90_cv10']} | {v['slope_per_stop']:.0f} |")
    add("")
    dx9 = data["dlog_x9_clip_by_ei"]
    add("DJI D-Log, Zenmuse X9 whitepaper (2022): one curve at every EI; only the clip level moves: "
        + "; ".join(f"{v} at EI " + ", ".join(str(k) for k in sorted((int(k) for k, x in dx9.items() if x == v)))
                    for v in sorted(set(dx9.values()), reverse=True)) + ".")
    add("")
    add("## 5. Sources and confidence")
    add("")
    for cid, c in C.items():
        add(f"- {_SHORT[cid]}: {c['doc_version']} ({c['confidence']}).")
    add("- DJI D-Log M has no published formula or gamut: its row above is a community fit, so treat it as rough.")
    add("")
    if part == "spec":
        return "\n".join(L) + "\n"
    add("## 6. What actually separates the encodings")
    add("")
    add(f"- **Black level families**: ~95 = {', '.join(fam['~95'])}; ~128 = {', '.join(fam['~128'])}; "
        f"~154 = {', '.join(fam['~154'])}. Only true black (lens cap, deep unlit shadow) shows it; a scene floor "
        "above the black level proves nothing, a floor clearly *below* it rules the curve out.")
    add(f"- **18% grey**: {ranked('grey18', official)}. Needs a grey card or a confidently known mid-tone; "
        "an exposure error of +-1 stop moves grey by 57-85 CV, more than most gaps.")
    add("- **Contrast (CV per stop at grey)**: "
        + ", ".join(f"{_SHORT[c]} {C[c]['slope_per_stop_at_grey_cv10']:.0f}" for c in slopes)
        + f". D-Log M (community fit) is ~{C['dji_dlogm']['slope_per_stop_at_grey_cv10']:.0f}, i.e. not a true log.")
    add("- **Clip behaviour**: LogC3 and D-Log clip levels move with EI (section 4). "
        "Log3G10 reaches 1.0 only at 184 (10 stops over grey), BMD Gen 5 at 223, LogC4 at 470, so their highlights "
        "usually clip well below 1023 at a camera-specific level.")
    add("- **Gamut (blue primary y)**: Kinefinity WG -0.224, AWG3 -0.102, Cinema Gamut -0.100, S-Gamut3.Cine -0.087, RWG -0.085, "
        "BMD WG -0.082, D-Gamut -0.080, S-Gamut3 -0.050, AWG4 -0.031, Rec.2020 +0.046. "
        "RWG (green y 1.494), Kinefinity WG (1.148) and Cinema Gamut (1.14) contain large imaginary regions. "
        "The same saturated object therefore has different RGB code ratios in each encoding.")
    add("- **EI dependence**: only LogC3 changes the curve itself (section 4); D-Log changes only the clip level; "
        "all others are single curves.")
    add("")
    add("## 7. How a DIT uses these numbers")
    add("")
    add("1. Read code values with a 10-bit full-range scope (0-1023). If blacks sit near 64 and nothing exceeds 940, "
        "the file may have been squeezed to video levels: full-range code y then appears as 64 + 0.856 * y "
        "(black 95 -> 145, grey 400 -> 406); BMD camera ProRes is recorded this way.")
    add("2. Find the darkest flat area (lens cap, deep shadow under an object). A floor at 93-96 points to the "
        "95-family, 126-130 to Canon Log 3 / N-Log, ~154 to Apple Log. Noise makes single pixels dip "
        "below black; use averages over flat patches.")
    add("3. Find a neutral reference (grey card, white card, overcast sky is NOT neutral-reliable). Compare its "
        "code with the grey / white columns, remembering the exposure is unknown: use differences between two "
        "known patches (e.g. grey vs. white = 2.32 stops) to estimate CV/stop, which is exposure-independent.")
    add("4. Look for a hard highlight plateau (many pixels at one identical value, often in all channels). Compare "
        "its code with the documented clip levels (section 4). A plateau above a curve's "
        "maximum possible code rules that curve out.")
    add("5. Decode saturated, well-exposed patches with each candidate curve + gamut to XYZ. Physically real "
        "surface colours stay inside the spectral locus; many negative or out-of-locus results mean the assumed "
        "gamut is too wide or the curve too steep. Too-small saturation everywhere hints the assumed gamut is too narrow.")
    add("6. Combine: black family + grey/CV-per-stop + clip level + gamut plausibility. Each cue alone is weak; "
        "when they disagree, report the top-2 candidates instead of a single answer.")
    add("")
    # Deliberately scene-neutral (studio chart, no brand / location / subject cue): an earlier version
    # described a real test clip and the model simply pattern-matched it.
    add("Worked example (studio chart): averaged floor 128 on a lens-cap black patch, a grey card at ~372, and "
        "no plateau anywhere -> the 128-family (Canon Log 3 / N-Log); grey 372 matches N-Log (Canon Log 3 "
        "would put grey at 351; check with the white card: N-Log 603, Canon Log 3 577). Same chart with floor 95 and grey 400 but no clip plateau: the "
        "95-family stays open (LogC3 / F-Log2 first by grey, then S-Log3 420, D-Log 408, BMD 392, Log3G10 341, "
        "LogC4 285 only if the image is clearly under-exposed).")
    add("")
    add("## 8. Caveats")
    add("")
    add("- D-Log M is visibly higher-contrast than D-Log.")
    add("- Container colour tags (e.g. Rec.709 nclc) and re-exports say nothing about the original encoding.")
    return "\n".join(L) + "\n"


def write_digest(md_path, json_path=None, extra: tuple = ()) -> tuple[Path, Path]:
    """Write the whitepaper digest (Markdown for prompts/humans + JSON for tools).

    extra = extension curves to add (e.g. ("canon_clog2",)).  The digest text is part of the request
    hash: never regenerate a shipped digest in place with a different curve set while a run that used it
    may still be resumed - write an extended digest to its own path (digest_ext.md)."""
    md_path = Path(md_path)
    json_path = Path(json_path) if json_path else md_path.with_suffix(".json")
    data = _digest_data(tuple(extra))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_digest_markdown(data, "full"), encoding="utf-8")
    md_path.with_name(md_path.stem + "_spec" + md_path.suffix).write_text(_digest_markdown(data, "spec"), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path, json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_table() -> None:
    print(f"{'curve':16s} {'EI':>5s} {'black':>6s} {'grey':>6s} {'white':>6s} {'clip':>6s} {'clipLin':>9s} "
          f"{'CV/stop':>7s} {'conf':>18s}")
    for cid in CURVE_IDS:
        e = expected_code_values(cid)
        f = e["cv10_float"]
        print(f"{cid:16s} {str(e['ei'] or '-'):>5s} {f['black']:6.1f} {f['grey18']:6.1f} {f['white90']:6.1f} "
              f"{f['clip']:6.1f} {e['clip']:9.3f} {e['slope_per_stop_at_grey']:7.1f} {e['confidence']:>18s}")


DEFAULT_DIGEST = "work/reference/digest.md"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true", help="print expected code values")
    ap.add_argument("--check", action="store_true", help="print derived-vs-vendor matrix differences")
    ap.add_argument("--digest", action="store_true", help="write the digest (.md, _spec.md and .json)")
    ap.add_argument("--digest-path", default=DEFAULT_DIGEST)
    ap.add_argument("--ext2", action="store_true",
                    help="add C-Log2 + Apple Log 2 + F-Log2 C (17-option set); default file name digest_ext2.md")
    ap.add_argument("--ext", action="store_true",
                    help="add Canon Log 2 (15-option set); default file name digest_ext.md")
    a = ap.parse_args(argv)
    if a.table or not (a.check or a.digest):
        _print_table()
    if a.check:
        for g, d in matrix_crosscheck().items():
            print(f"{g:18s} max|derived - vendor| = {d:.2e}")
    if a.digest:
        default = a.digest_path == DEFAULT_DIGEST
        if a.ext2:
            from cameralogbench.taxonomy import EXT2_CURVE_IDS
            path = Path(a.digest_path).with_name("digest_ext2.md") if default else a.digest_path
            md, js = write_digest(path, extra=tuple(EXT2_CURVE_IDS))
        elif a.ext:
            from cameralogbench.taxonomy import EXT_CURVE_IDS
            path = Path(a.digest_path).with_name("digest_ext.md") if default else a.digest_path
            md, js = write_digest(path, extra=tuple(EXT_CURVE_IDS))
        else:
            md, js = write_digest(a.digest_path)
        text = md.read_text(encoding="utf-8")
        print(f"wrote {md} ({len(text)} chars, ~{len(text) // 4} tokens) and {js}")


if __name__ == "__main__":
    main()
