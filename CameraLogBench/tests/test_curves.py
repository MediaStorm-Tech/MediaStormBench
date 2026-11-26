"""Whitepaper-anchored unit tests for cameralogbench/curves.py.

Anchor values come from the vendor documents cited in each curve's docstring;
tolerance is +-1 10-bit code value unless the document prints more digits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cameralogbench import curves as C
from cameralogbench.taxonomy import CURVES, GAMUTS

CV = 1023.0
D65 = np.array([0.9505, 1.0, 1.0891])


def cv10(curve_id, lin, ei=None):
    return C.encode(curve_id, lin, ei) * CV


def _samples(curve_id, ei=None, n=200):
    """200 points in [0, clip]: half linear, half log-spaced, minus documented exclusion bands."""
    clip = C.decode(curve_id, 1.0, ei)
    x = np.unique(np.concatenate([np.linspace(0.0, clip, n // 2), np.geomspace(1e-5, clip, n // 2)]))
    keep = np.ones_like(x, dtype=bool)
    for lo, hi in C.round_trip_exclusions(curve_id):
        keep &= ~((x >= lo) & (x <= hi))
    return x[keep]


def test_registry_covers_all_v1_curves():
    v1 = [cid for cid, c in CURVES.items() if c.status in ("v1_main", "v1_second")]
    assert C.CURVE_IDS == v1
    assert len(v1) == 12
    for cid in v1:
        spec = C.CURVE_SPECS[cid]
        assert spec.confidence in ("verified", "verified_secondary", "unverified")
        assert spec.source and spec.doc_version and spec.formula


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curve_id", C.CURVE_IDS)
def test_round_trip(curve_id):
    x = _samples(curve_id)
    assert len(x) >= 190
    back = C.decode(curve_id, C.encode(curve_id, x))
    assert np.max(np.abs(back - x)) < 1e-6


@pytest.mark.parametrize("ei", sorted(C.LOGC3_PARAMS))
def test_round_trip_logc3_every_ei(ei):
    x = _samples("arri_logc3", ei)
    back = C.decode("arri_logc3", C.encode("arri_logc3", x, ei), ei)
    assert np.max(np.abs(back - x)) < 1e-6


@pytest.mark.parametrize("curve_id", [c for c in C.CURVE_IDS if C.round_trip_exclusions(c)])
def test_exclusion_bands_are_narrow(curve_id):
    for lo, hi in C.round_trip_exclusions(curve_id):
        assert hi - lo < 1e-3


def test_scalar_in_scalar_out():
    assert isinstance(C.encode("sony_slog3", 0.18), float)
    assert isinstance(C.decode("sony_slog3", 0.41), float)
    assert C.encode("sony_slog3", np.array([0.18])).shape == (1,)


# --------------------------------------------------------------------------- #
# Whitepaper anchors (10-bit, +-1 code value)
# --------------------------------------------------------------------------- #
ANCHORS = [
    # curve, linear, expected 10-bit code, source
    ("sony_slog3", 0.0, 95, "Sony S-Log3 summary"),
    ("sony_slog3", 0.18, 420, "Sony S-Log3 summary"),
    ("sony_slog3", 0.90, 598, "Sony S-Log3 summary"),
    ("arri_logc3", 0.0, 0.0928 * CV, "ARRI LogC: black 0.0928"),
    ("arri_logc3", 0.18, 400, "ARRI LogC: grey 0.391 = 400/1023"),
    ("dji_dlog", 0.0, 95, "DJI D-Log table"),
    ("dji_dlog", 0.18, 408, "DJI D-Log table"),
    ("dji_dlog", 0.90, 586, "DJI D-Log table"),
    ("arri_logc4", 0.0, 95, "LogC4 App. B/C: 0.0929, 10-bit full 95"),
    ("arri_logc4", 0.18, 285, "LogC4 App. B/D: 0.2784, 10-bit full 285"),
    ("red_log3g10", 0.0, 0.091551 * CV, "RED mapping table"),
    ("red_log3g10", 0.18, CV / 3.0, "RED: 18% -> 1/3"),
    ("red_log3g10", 1.0, 0.493449 * CV, "RED mapping table"),
    ("canon_clog3", 0.0, 128, "Canon: 0% black = code 128"),
    ("canon_clog3", 0.18, 351, "Canon HDR Part 2: 18% grey = code 351 (34.3%)"),
    ("canon_clog2", 0.0, 95, "Canon Input Transform 202007 constant 0.092864125"),
    ("canon_clog2", 0.18, 0.398254692 * CV, "colour-science v1.2 doctest: 0.18 -> 0.398254692"),
    ("bmd_film_gen5", 0.0, 0.0924657534246575 * CV, "BMD Gen5 mapping table"),
    ("bmd_film_gen5", 0.18, 0.3835616438356165 * CV, "BMD Gen5 mapping table"),
    ("bmd_film_gen5", 1.0, 0.5304896249573048 * CV, "BMD Gen5 mapping table"),
    ("bmd_film_gen5", 10.0, 0.7302219538415439 * CV, "BMD Gen5 mapping table"),
    ("apple_log", 0.0, 64 + 876 * 0.102, "Apple Log: black 10.2% camera IRE"),
    ("apple_log", 0.18, 64 + 876 * 0.497, "Apple Log: grey 49.7% camera IRE"),
    ("panasonic_vlog", 0.0, 128, "Panasonic V-Log table"),
    ("panasonic_vlog", 0.18, 433, "Panasonic V-Log table"),
    ("panasonic_vlog", 0.90, 602, "Panasonic V-Log table"),
    ("fuji_flog2", 0.0, 95, "Fujifilm F-Log2 table"),
    ("fuji_flog2", 0.18, 400, "Fujifilm F-Log2 table"),
    ("fuji_flog2", 0.90, 570, "Fujifilm F-Log2 table"),
    # N-Log's spec is written directly in 10-bit code values: 650*(y+0.0075)^(1/3)
    ("nikon_nlog", 0.0, 650 * 0.0075 ** (1 / 3), "Nikon N-Log formula (code domain)"),
    ("nikon_nlog", 0.18, 650 * 0.1875 ** (1 / 3), "Nikon N-Log formula (code domain)"),
    ("nikon_nlog", 0.90, 150 * math.log(0.9) + 619, "Nikon N-Log formula (code domain)"),
    # KineLog3 spec anchors are printed as IRE: black 0.09 (= d 0.092864), 18% 39.2, 90% 57.2
    ("kine_log3", 0.0, 0.092864 * 1023, "Kinefinity KineLog3 spec (d)"),
    ("kine_log3", 0.18, 0.392 * 1023, "Kinefinity KineLog3 spec (18% -> 39.2)"),
    ("kine_log3", 0.90, 0.572 * 1023, "Kinefinity KineLog3 spec (90% -> 57.2)"),
]


def test_kinelog3_branches_meet_at_zero():
    assert abs(C.encode("kine_log3", -0.008239)) < 1e-4
    assert C.decode("kine_log3", 0.0) == pytest.approx(-0.008239, abs=1e-4)


@pytest.mark.parametrize("curve_id,lin,expected,src", ANCHORS)
def test_anchor(curve_id, lin, expected, src):
    assert abs(cv10(curve_id, lin) - expected) <= 1.0, src


@pytest.mark.parametrize("curve_id,code,lin", [
    ("red_log3g10", 1.0, 184.322),
    ("bmd_film_gen5", 1.0, 222.86),
    ("arri_logc4", 1.0, 469.80),
])
def test_documented_code_one(curve_id, code, lin):
    assert C.decode(curve_id, code) == pytest.approx(lin, rel=2e-4)


def test_exact_printed_values():
    assert C.encode("red_log3g10", 0.18) == pytest.approx(1 / 3, abs=1e-6)
    assert C.encode("red_log3g10", -0.01) == pytest.approx(0.0, abs=1e-12)
    assert C.encode("bmd_film_gen5", 0.18) == pytest.approx(0.3835616438356165, abs=1e-9)
    assert C.encode("arri_logc4", 0.18) == pytest.approx(0.2784, abs=5e-5)
    assert C.encode("arri_logc4", 0.0) == pytest.approx(95 / 1023, abs=1e-12)
    assert C.decode("arri_logc4", 0.0) == pytest.approx(-0.0181, abs=5e-5)


def test_bmd_video_levels_table():
    # BMD prints 10-bit video levels 145 / 400 / 529 / 940 for 0 / 0.18 / 1 / 222.86
    for lin, legal in ((0.0, 145), (0.18, 400), (1.0, 529), (222.86, 940)):
        assert abs(64 + 876 * C.encode("bmd_film_gen5", lin) - legal) <= 1.0


def test_logc4_legal_table():
    # LogC4 Appendix D: 10-bit legal 145 (black), 308 (grey)
    for lin, legal in ((0.0, 145), (0.18, 308)):
        assert abs(64 + 876 * C.encode("arri_logc4", lin) - legal) <= 1.0


def test_dlogm_fit_shape():
    # unverified community fit: only check the documented shape
    assert abs(cv10("dji_dlogm", 0.18) - 409.2) < 1.0
    assert C.expected_code_values("dji_dlogm")["slope_per_stop_at_grey"] > 100
    assert C.CURVE_SPECS["dji_dlogm"].confidence == "unverified"


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curve_id", C.CURVE_IDS)
def test_monotonic(curve_id):
    clip = C.decode(curve_id, 1.0)
    x = np.unique(np.concatenate([np.linspace(0, clip, 2000), np.geomspace(1e-6, clip, 2000)]))
    y = C.encode(curve_id, x)
    assert np.all(np.diff(y) > 0)
    codes = np.linspace(C.encode(curve_id, 0.0), 1.0, 2000)
    assert np.all(np.diff(C.decode(curve_id, codes)) > 0)
    assert 0.0 <= C.encode(curve_id, 0.0) < C.encode(curve_id, 0.18) < C.encode(curve_id, 0.9) < 1.0


# --------------------------------------------------------------------------- #
# EI behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ei", sorted(C.LOGC3_PARAMS))
def test_logc3_grey_and_black_constant_over_ei(ei):
    assert abs(cv10("arri_logc3", 0.18, ei) - 400) <= 1.0
    assert abs(C.encode("arri_logc3", 0.18, ei) - 0.391) < 5e-4
    assert abs(C.encode("arri_logc3", 0.0, ei) - 0.0928) < 1e-4


def test_logc3_parameter_table_consistency():
    for ei, (cut, a, b, c, d, e, f) in C.LOGC3_PARAMS.items():
        # linear and log segments meet at the cut (published e*cut+f column)
        assert c * math.log10(a * cut + b) + d == pytest.approx(e * cut + f, abs=2e-6)


def test_logc3_clip_table_and_ei_resolution():
    assert C.expected_code_values("arri_logc3", 800)["cv10"]["clip"] == 976
    assert C.expected_code_values("arri_logc3", 160)["cv10"]["clip"] == 831
    assert C.expected_code_values("arri_logc3", 3200)["cv10"]["clip"] == 1023
    assert C.logc3_resolve_ei(None) == 800
    assert C.logc3_resolve_ei(3200) == 1600
    assert C.logc3_resolve_ei(850) == 800


def test_dlog_clip_table():
    assert C.expected_code_values("dji_dlog")["cv10"]["clip"] == 948
    assert C.expected_code_values("dji_dlog", 400)["cv10"]["clip"] == 868
    assert C.expected_code_values("dji_dlog", 200)["cv10"]["clip"] == 789
    # grey does not move with EI in scene-linear terms
    assert C.expected_code_values("dji_dlog", 200)["cv10"]["grey18"] == 408


# --------------------------------------------------------------------------- #
# expected_code_values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curve_id", C.CURVE_IDS)
def test_expected_code_values_structure(curve_id):
    e = C.expected_code_values(curve_id)
    for k in ("black", "grey18", "white90", "clip", "clip_code", "slope_per_stop_at_grey", "cv10", "linear_at_code_1"):
        assert k in e
    assert e["cv10"]["grey18"] == round(e["grey18"] * CV)
    assert e["cv10"]["black"] == round(e["black"] * CV)
    assert 45 < e["slope_per_stop_at_grey"] < 130
    assert e["clip"] <= e["linear_at_code_1"] + 1e-9
    assert C.encode(curve_id, e["clip"]) == pytest.approx(e["clip_code"], abs=1e-6)
    row = C.expected_table_row(curve_id)
    assert set(row) >= {"black", "grey18", "white90", "clip", "slope_per_stop"}


# --------------------------------------------------------------------------- #
# Gamuts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamut_id", list(GAMUTS))
def test_rgb_to_xyz_rows_sum_to_d65(gamut_id):
    M = C.gamut_to_xyz(gamut_id)
    assert M.shape == (3, 3)
    assert np.allclose(M.sum(axis=1), D65, atol=1e-4)
    assert np.allclose(M @ np.ones(3), C.D65_XYZ, atol=2e-6)


@pytest.mark.parametrize("gamut_id", list(GAMUTS))
def test_inverse_and_primaries(gamut_id):
    M, Mi = C.gamut_to_xyz(gamut_id), C.xyz_to_gamut(gamut_id)
    assert np.allclose(Mi @ M, np.eye(3), atol=2e-5)
    p = C.gamut_primaries(gamut_id)
    for i, k in enumerate("RGB"):
        X, Y, Z = M[:, i]
        assert (X / (X + Y + Z), Y / (X + Y + Z)) == pytest.approx(p[k], abs=1e-5)
    assert p["W"] == C.D65_XY


@pytest.mark.parametrize("gamut_id,tol", [
    ("arri_awg3", 1e-5), ("arri_awg4", 1e-6), ("red_rwg", 1e-5), ("panasonic_vgamut", 1e-5),
    ("bmd_wg_gen5", 1e-3), ("dji_dgamut", 1e-3),
])
def test_matches_vendor_matrices(gamut_id, tol):
    assert C.matrix_crosscheck()[gamut_id] < tol


def test_awg3_inverse_matches_arri():
    arri = np.array([[1.789066, -0.482534, -0.200076], [-0.639849, 1.396400, 0.194432],
                     [-0.041532, 0.082335, 0.878868]])
    assert np.max(np.abs(C.xyz_to_gamut("arri_awg3") - arri)) < 1e-5


def test_relative_purity():
    pur = C.relative_purity(np.array([C.D65_XY, (0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]))
    assert pur[0] == 0.0
    assert np.all((pur[1:] > 0.5) & (pur[1:] < 1.0))
    assert C.relative_purity(np.array([[0.121595, 1.493994]]))[0] > 1.0   # RWG green is imaginary
    locus = C.SPECTRAL_LOCUS_XY[5:-5]
    assert np.allclose(C.relative_purity(locus), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Evidence on a synthetic S-Log3 frame with known black, clip and colours
# --------------------------------------------------------------------------- #
def _synthetic_slog3_png(path, clip_cv=920.0):
    import cv2

    rng = np.random.default_rng(0)
    h, w = 288, 480
    lin = np.zeros((h, w, 3))
    lin[:, 96:] = np.geomspace(0.005, 60.0, w - 96)[None, :, None]           # grey ramp, clips on the right
    lin[:96, 96:240] = 0.18                                                   # grey card
    M = C.xyz_to_gamut("sony_sgamut3cine") @ C.gamut_to_xyz("rec709")
    for i, rgb in enumerate([(0.5, 0.05, 0.05), (0.05, 0.4, 0.05), (0.05, 0.05, 0.5)]):
        lin[192:288, 96 + 48 * i:144 + 48 * i] = M @ np.array(rgb)          # Rec.709 colours, in-gamut
    code = C.encode("sony_slog3", lin) * CV + rng.normal(0, 1.0, lin.shape)
    code = np.minimum(code, clip_cv)
    v16 = np.clip(np.rint(code / CV * 65535), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), v16[..., ::-1])


def test_evidence_on_synthetic_slog3(tmp_path):
    from cameralogbench import evidence as E

    png = tmp_path / "synthetic.png"
    _synthetic_slog3_png(png)
    r = E.compute_evidence(str(png), "c_test", "unit", "synthetic")
    assert r["black"]["has_true_black"]
    assert abs(r["black"]["estimate_cv"] - 95.0) < 2.0
    assert r["highlight"]["plateau_cv"] == pytest.approx(920.0, abs=2.0)
    ruled = set(r["classical_prediction"]["ruled_out"])
    assert {"canon_clog3", "nikon_nlog", "apple_log"} <= ruled
    assert "sony_slog3" not in ruled
    assert r["curve_consistency"]["sony_slog3"]["imaginary_fraction"] == 0.0
    assert abs(sum(e["score"] for e in r["curve_consistency"].values()) - 1.0) < 1e-3
    assert set(r["expected_table"]) == set(C.CURVE_IDS)


def test_plateau_detector_rejects_smooth_tail():
    from cameralogbench import evidence as E

    hist = np.zeros(1024)
    hist[100:800] = np.linspace(5000, 10, 700)
    assert E.detect_plateau(hist, int(hist.sum())) is None
    hist[760] += 20000
    hist[761:800] = 0
    assert E.detect_plateau(hist, int(hist.sum())) == pytest.approx(760.5, abs=0.6)


# --------------------------------------------------------------------------- #
# Digest
# --------------------------------------------------------------------------- #
def test_write_digest(tmp_path):
    md, js = C.write_digest(tmp_path / "digest.md", tmp_path / "digest.json")
    text = md.read_text(encoding="utf-8")
    for short in ("LogC3", "LogC4", "S-Log3", "D-Log", "D-Log M", "Log3G10", "Canon Log 3", "Film Gen 5",
                  "Apple Log", "F-Log2", "N-Log", "KineLOG3"):
        assert short in text
    assert "V-Log" not in text  # not a benchmark curve
    spec = (tmp_path / "digest_spec.md").read_text(encoding="utf-8")
    assert "How a DIT" not in spec and "Worked example" not in spec and "## 4. Documented EI" in spec
    assert spec.split("## 6.")[0].strip() in text or len(spec) < len(text)
    assert 7000 < len(text) < 16000           # ~2500-3500 tokens
    data = json.loads(js.read_text(encoding="utf-8"))
    assert set(data["curves"]) == set(C.CURVE_IDS)
    assert data["curves"]["sony_slog3"]["cv10"] == {"black": 95, "grey18": 420, "white90": 598, "clip": 1023}


def test_canon_log2_is_an_extension_curve(tmp_path):
    """Canon Log 2 has a formula and a 15-option set, but stays out of the 14-option lists."""
    from cameralogbench.taxonomy import EXT_CURVE_IDS, OPTION_SET_CURVE, OPTION_SET_CURVE_EXT
    assert "canon_clog2" in C.CURVE_SPECS and "canon_clog2" not in C.CURVE_IDS
    assert "canon_clog2" not in OPTION_SET_CURVE and len(OPTION_SET_CURVE) == 14
    assert OPTION_SET_CURVE_EXT == OPTION_SET_CURVE[:6] + ["canon_clog2"] + OPTION_SET_CURVE[6:]  # before C-Log3
    x = np.linspace(-0.02, 8.0, 801)
    assert np.allclose(C.decode("canon_clog2", C.encode("canon_clog2", x)), x, atol=1e-9)
    md, _ = C.write_digest(tmp_path / "digest.md")                       # default digest: unchanged curve set
    assert "Canon Log 2" not in md.read_text(encoding="utf-8")
    md, _ = C.write_digest(tmp_path / "digest_ext.md", extra=tuple(EXT_CURVE_IDS))
    assert "Canon Log 2" in md.read_text(encoding="utf-8")
    assert "Canon Log 2" in (tmp_path / "digest_ext_spec.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra,name", [((), "digest"), (("canon_clog2",), "digest_ext"),
                                        (("canon_clog2", "apple_log2", "fuji_flog2c"), "digest_ext2")])
def test_shipped_digests_are_reproduced_byte_for_byte(tmp_path, extra, name):
    """The digests are prompt text: the generator must reproduce the files the paper's runs used."""
    ref = Path(C.__file__).parent / "reference"
    C.write_digest(tmp_path / f"{name}.md", extra=extra)
    for suffix in (".md", "_spec.md", ".json"):
        assert (tmp_path / f"{name}{suffix}").read_bytes() == (ref / f"{name}{suffix}").read_bytes(), suffix
