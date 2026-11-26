"""Tests for the synthetic frame generator."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from cameralogbench import curves
from cameralogbench.synth.chart import W, H, build_chart, measure_patch_cv10
from cameralogbench.synth.colorops import (
    encode_chart_linear, encode_not_log, identity_error, reencode_log_rgb,
    rec709_oetf,
)
from cameralogbench.synth.generate import (
    chart_self_check, make_sample_id, write_frame_versions,
)
from cameralogbench.taxonomy import NOT_LOG, OPTION_SET_CURVE


@pytest.fixture(scope="module")
def chart():
    return build_chart(seed=0)


def test_chart_shape_and_annotations(chart):
    lin, ann, layout = chart
    assert lin.shape == (H, W, 3)
    assert lin.dtype == np.float64
    names = {a["name"] for a in ann}
    assert "grey18" in names and "true_black" in names
    assert any(a["kind"] == "color" and a.get("xy") for a in ann)
    assert layout["width"] == W


@pytest.mark.parametrize("curve_id", curves.CURVE_IDS)
def test_chart_self_check_tol1(chart, curve_id):
    lin, _, layout = chart
    ei = 800 if curve_id == "arri_logc3" else None
    rgb16, _ = encode_chart_linear(lin, curve_id, ei, 0.0)
    sc = chart_self_check(rgb16, layout, curve_id, ei, 0.0, tol_cv=1.0)
    assert sc["all_pass"], sc


def test_sample_id_stable():
    a = make_sample_id("chart_v0", "chart", "sony_slog3", None, "exp+0")
    b = make_sample_id("chart_v0", "chart", "sony_slog3", None, "exp+0")
    assert a == b and a.startswith("s_") and len(a) == 12


def test_not_log_variants(chart):
    lin, _, _ = chart
    for v in ("rec709", "fake_log"):
        rgb16, meta = encode_not_log(lin, v, 0.0)
        assert rgb16.dtype == np.uint16
        assert meta["curve_id"] == NOT_LOG
        assert meta["not_log_variant"] == v


def test_rec709_oetf_anchors():
    assert abs(float(rec709_oetf(0.0)) - 0.0) < 1e-12
    assert abs(float(rec709_oetf(1.0)) - 1.0) < 1e-6


def test_write_frame_versions(tmp_path, chart):
    lin, _, _ = chart
    rgb16, _ = encode_chart_linear(lin, "sony_slog3", None, 0.0)
    out = tmp_path / "frames" / "s_test"
    info = write_frame_versions(out, rgb16)
    assert (out / "main16.png").is_file()
    assert (out / "eval_1920.jpg").is_file()
    assert (out / "thumb_480.jpg").is_file()
    assert info["width"] == W


def test_linear_dir_roundtrip(tmp_path, chart):
    """linear_dir path: write a tiny linear PNG, encode, decode grey ~ 0.18."""
    from cameralogbench.synth.generate import load_linear_rgb
    lin, _, layout = chart
    # crop a small grey region and save as 16-bit linear PNG
    by = {p["name"]: p for p in layout["patches"]}
    x, y, w, h = by["grey18"]["bbox"]
    patch = lin[y:y + h, x:x + w]
    u16 = np.clip(np.rint(np.clip(patch, 0, 1) * 65535), 0, 65535).astype(np.uint16)
    p = tmp_path / "grey_lin.png"
    cv2.imwrite(str(p), u16[..., ::-1])
    loaded = load_linear_rgb(p)
    assert abs(float(loaded.mean()) - 0.18) < 0.01
    rgb16, _ = encode_chart_linear(loaded, "sony_slog3", None, 0.0)
    # measure should be near S-Log3 grey 420
    cv10 = rgb16.astype(np.float64).mean() * 1023.0 / 65535.0
    assert abs(cv10 - 420.0) < 2.0


def test_reencode_identity_synthetic(chart):
    """Encode chart as S-Log3 then re-encode to itself; error near quantisation."""
    lin, _, _ = chart
    src16, _ = encode_chart_linear(lin, "sony_slog3", None, 0.0)
    dst16, meta = reencode_log_rgb(src16, "sony_slog3", "sony_slog3")
    assert meta["identity"]
    err = identity_error(src16, dst16)
    # 16-bit quantisation + float round-trip: allow a few CV
    assert err["max_abs_cv10"] < 2.0
    assert err["mse_cv10"] < 0.5
    assert meta["clamp"]["negative_linear_fraction"] == 0.0


@pytest.mark.parametrize("src,dst", [
    ("sony_slog3", "arri_logc3"), ("sony_slog3", "red_log3g10"),
    ("red_log3g10", "apple_log"), ("arri_logc4", "panasonic_vlog"),
])
def test_reencode_cross_gamut_matches_direct_encode(chart, src, dst):
    """decode(src) -> XYZ -> dst must equal encoding the same linear scene as dst directly.

    Catches swapped gamut_to_xyz / xyz_to_gamut (colour patches would be far off).
    The specular patch is excluded: the source clips it at code 1.0.
    """
    lin_full, _, layout = chart
    x, y, w, h = {p["name"]: p for p in layout["patches"]}["specular_clip"]["bbox"]
    mask = np.ones(lin_full.shape[:2], bool)
    mask[y:y + h, x:x + w] = False
    lin, mask = lin_full[::4, ::4], mask[::4, ::4]
    src16, _ = encode_chart_linear(lin, src, None, 0.0)
    re16, meta = reencode_log_rgb(src16, src, dst)
    assert not meta["identity"] and meta["clamp"]["negative_linear_fraction"] == 0.0
    from cameralogbench.synth.colorops import encode_linear_to_code, rgb709_to_gamut, code_to_u16
    code, _ = encode_linear_to_code(rgb709_to_gamut(lin, curves.curve_gamut(dst)), dst, None,
                                    apply_sensor_clip=False)
    direct16 = code_to_u16(code)
    diff = np.abs(re16.astype(np.float64) - direct16.astype(np.float64))[mask] * 1023.0 / 65535.0
    assert diff.max() < 0.5, diff.max()


def test_frames_on_disk_match_and_follow_eval_spec(tmp_path, chart):
    """main16.png reads back bit-exact in RGB order; eval_1920 = (main16 >> 8), 1920 wide, q92 4:4:4."""
    from cameralogbench import evidence
    from cameralogbench.synth.generate import load_rgb16
    lin, _, layout = chart
    rgb16, _ = encode_chart_linear(lin, "arri_logc4", None, 0.0)
    out = tmp_path / "s_disk"
    info = write_frame_versions(out, rgb16)
    for loader in (load_rgb16, lambda p: evidence.load_rgb16(str(p))):
        back = loader(out / "main16.png")
        assert back.dtype == np.uint16 and np.array_equal(back, rgb16)
    red = {p["name"]: p for p in layout["patches"]}["cc_red"]["bbox"]
    r, g, b = measure_patch_cv10(load_rgb16(out / "main16.png"), tuple(red))
    assert r > g and r > b
    png8 = cv2.imread(str(out / "eval_1920.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    assert np.array_equal(png8, (rgb16 >> 8).astype(np.uint8))
    jpg = cv2.imread(str(out / "eval_1920.jpg"))
    assert jpg.shape == (H, W, 3)
    assert info["outputs"]["eval_1920.jpg"] == {"w": W, "h": H, "jpeg_quality": 92, "sampling": "4:4:4"}


@pytest.mark.parametrize("curve_id,ei,stops", [
    ("sony_slog3", None, 0.0), ("dji_dlog", None, 2.0),
    ("arri_logc3", 400, 0.0), ("arri_logc3", 1600, -2.0),
])
def test_chart_reference_fit_recovers_truth(tmp_path, chart, curve_id, ei, stops):
    """On a noise-free chart the gold curve must fit the annotated patches almost exactly.

    Guards the annotation reflectances (they must equal the rendered linear values)
    and checks that LogC3 EI variants really use their own parameters.
    """
    from cameralogbench import evidence
    lin, ann, layout = chart
    for a in ann:
        if a["kind"] == "neutral":
            x, y, w, h = a["bbox"]
            patch_y = float(lin[y + h // 2, x + w // 2] @ np.array([0.2126, 0.7152, 0.0722]))
            assert a["reflectance"] == pytest.approx(patch_y, rel=1e-9), a["name"]
    rgb16, _ = encode_chart_linear(lin, curve_id, ei, stops)
    out = tmp_path / "s_fit"
    write_frame_versions(out, rgb16)
    ev = evidence.compute_evidence(str(out / "main16.png"), "s_fit", annotations=ann)
    rf = ev["curve_consistency"][curve_id]["reference_fit"]
    assert rf["neutral_rms_cv"] < 0.5
    assert rf["chroma_dxy"] < 1e-3
    assert rf["exposure_stops"] == pytest.approx(stops, abs=0.02)
    if curve_id == "arri_logc3":
        assert rf["ei"] == ei
    assert ev["classical_prediction"]["ranked"][0] == curve_id


def test_reasoning_does_not_borrow_gold_values():
    """Without a grey card the grey cue stays empty even if a scene neutral sits at the gold grey."""
    from cameralogbench.synth.reasoning import build_reasoning
    gold = "sony_slog3"
    grey = curves.expected_code_values(gold)["cv10_float"]["grey18"]
    ev = {
        "black": {"estimate_cv": 150.0, "fine_floor_cv": 140.0, "violation_tol_cv": 5.0,
                  "has_true_black": False},
        "highlight": {"plateau_cv": 900.0, "robust_max_cv": 901.0},
        "neutral_candidates_cv": [grey - 0.3, 600.0],
        "curve_consistency": {c: {"penalty": float(i)} for i, c in enumerate(curves.CURVE_IDS)},
        "classical_prediction": {"ranked": list(curves.CURVE_IDS), "ruled_out": []},
    }
    r = build_reasoning("s_x", gold, ev, source_class="S-sensor", sensor_clip_applied=False)
    st = {s["cue"]: s for s in r["steps"]}
    assert st["grey_18_percent"]["observed"]["mean_cv"] is None
    assert "compatible" in st["black_level"]["conclusion"]
    assert "not diagnostic" in st["highlight_clip"]["conclusion"]
    assert r["evidence_sufficient"] is False and r["gold_rank"] == curves.CURVE_IDS.index(gold) + 1
    assert st["black_level"]["expected_by_curve"][gold] == curves.expected_code_values(gold)["cv10_float"]["black"]


def test_option_set_includes_not_log():
    assert NOT_LOG in OPTION_SET_CURVE


def test_generate_chart_smoke(tmp_path):
    """End-to-end tiny chart set (2 curves x 1 exposure + not_log)."""
    from cameralogbench.synth.generate import main
    art = tmp_path / "synth"
    rc = main([
        "--set", "smoke_chart", "--source", "chart",
        "--out-root", str(art),
        "--curves", "sony_slog3", "arri_logc4",
        "--exposure-offsets", "0",
        "--force",
    ])
    assert rc == 0
    root = art / "smoke_chart"
    for name in ("manifest.jsonl", "annotations.jsonl", "evidence.jsonl",
                 "reasoning.jsonl", "contact_sheet.jpg"):
        assert (root / name).is_file(), name
    items = [json.loads(l) for l in (root / "annotations.jsonl").read_text().splitlines() if l.strip()]
    # 2 log + 2 not_log
    assert len(items) == 4
    assert all(it["task"] == "curve" for it in items)
    assert all(it["options"] == OPTION_SET_CURVE for it in items)
    assert all((root / it["input"]["image_webp"]).is_file() for it in items)  # relative to the set folder
    reasons = [json.loads(l) for l in (root / "reasoning.jsonl").read_text().splitlines()]
    assert reasons[0]["steps"] and reasons[0]["expert_summary"]


# --------------------------------------------------------------------------- #
# prep_linear (public HDR sources of the synthetic natural scenes)
# --------------------------------------------------------------------------- #
def test_prep_d2x_white_maps_to_709_white():
    from cameralogbench.synth.prep_linear import d2x_to_709
    out = d2x_to_709(np.ones((1, 1, 3), np.float32))
    assert np.allclose(out, 1.0, atol=2e-3)


@pytest.mark.parametrize("yaw,u_frac", [(0.0, 0.5), (90.0, 0.75), (-90.0, 0.25)])
def test_prep_equirect_view_direction(yaw, u_frac):
    """A marker at longitude u_frac*W on the horizon lands at the view centre."""
    from cameralogbench.synth.prep_linear import equirect_view
    eq = np.zeros((256, 512, 3), np.float32)
    u = int(u_frac * 512)
    eq[124:132, u - 4:u + 4] = 1.0
    view = equirect_view(eq, yaw, 0.0, 60.0, 64, 36)
    cy, cx = np.unravel_index(np.argmax(view[..., 0]), view.shape[:2])
    assert abs(cx - 32) <= 2 and abs(cy - 18) <= 2


def test_prep_one_normalises_to_key(tmp_path):
    from cameralogbench.synth.generate import load_linear_rgb
    from cameralogbench.synth.prep_linear import KEY, log_average, prep_one
    rng = np.random.default_rng(0)
    img = np.exp(rng.normal(-4.0, 1.5, size=(90, 160, 3))).astype(np.float32)
    src = tmp_path / "src.exr"
    assert cv2.imwrite(str(src), img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    row = {"source_id": "t_src", "dataset": "test", "path": str(src),
           "colour": "rec709_linear", "projection": "perspective"}
    (tmp_path / "out").mkdir()
    rec = prep_one(row, tmp_path / "out", 3840, True)
    out = load_linear_rgb(tmp_path / "out" / "t_src.exr")
    # half-float storage: allow ~0.1 % on the log-average
    assert abs(log_average(out.astype(np.float32)) / KEY - 1.0) < 2e-3
    assert rec["size"] == [160, 90]
