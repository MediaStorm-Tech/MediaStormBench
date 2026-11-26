"""Synthetic frame generator CLI.

Clamp / gamut policy for re-encode lives in ``cameralogbench.synth.colorops`` (module docstring).

Examples::

    # the synthetic chart, every benchmark curve, three exposures
    python -m cameralogbench.synth.generate --set chart_v0 --source chart \\
        --curves all --ei-variants --exposure-offsets -2 0 2

    # natural scenes from scene-linear EXR / HDR images (see synth.fetch_sources and synth.prep_linear)
    python -m cameralogbench.synth.generate --set scenes --source linear_dir \\
        --linear-dir data/synth_sources/linear_v1 --curves all

    # re-encode 16-bit Log frames of known curve with every curve
    python -m cameralogbench.synth.generate --set reencode --source reencode \\
        --reencode frame_a.png sony_slog3 --reencode frame_b.png red_log3g10 --curves all

Writes ``<out-root>/<set>/frames/<sample_id>/`` (main16.png, eval_1920.png / .webp / .jpg, thumb), plus
manifest.jsonl, evidence.jsonl, reasoning.jsonl and ``annotations.jsonl`` in the dataset schema (paths
relative to ``<out-root>/<set>``, so the runner takes ``--data-root <out-root>/<set>``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# OpenCV only decodes .exr (linear_dir sources) when this is set before import.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from cameralogbench import curves, evidence
from cameralogbench.frames import (ensure_webp, 
    EVAL_JPEG_Q, EVAL_LONG, THUMB_JPEG_Q, THUMB_LONG,
    jpeg_params, resize_long, to8,
)
from cameralogbench._io import repo_rel, write_jsonl
from cameralogbench.synth.chart import build_chart, measure_patch_cv10
from cameralogbench.synth.colorops import (
    brand_for_curve, encode_chart_linear, encode_not_log, identity_error,
    reencode_log_rgb, u16_to_code,
)
from cameralogbench.synth.reasoning import build_reasoning
from cameralogbench.taxonomy import CURVES, EXT2_CURVE_IDS, NOT_LOG, OPTION_SET_CURVE, OPTION_SETS_CURVE

# D-Log M has no published formula: never use it as a re-encode *source*; allowed as target.
UNVERIFIED_SOURCE_CURVES = frozenset({"dji_dlogm"})

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# IDs / IO
# --------------------------------------------------------------------------- #
def make_sample_id(set_id: str, source_id: str, curve_id: str, ei, variant: str) -> str:
    ei_s = "" if ei is None else str(ei)
    key = f"{set_id}{source_id}{curve_id}{ei_s}{variant}"
    return "s_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def write_frame_versions(out_dir: Path, rgb16: np.ndarray) -> dict[str, Any]:
    """Write main16.png + eval_1920.jpg/png + thumb_480.jpg (RGB in, BGR on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr16 = np.ascontiguousarray(rgb16[..., ::-1])
    main = out_dir / "main16.png"
    if not cv2.imwrite(str(main), bgr16):
        raise RuntimeError(f"failed to write {main}")
    ev = to8(resize_long(bgr16, EVAL_LONG))
    th = to8(resize_long(bgr16, THUMB_LONG))
    ok = cv2.imwrite(str(out_dir / "eval_1920.jpg"), ev, jpeg_params(EVAL_JPEG_Q))
    ok &= cv2.imwrite(str(out_dir / "eval_1920.png"), ev, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    ok &= cv2.imwrite(str(out_dir / "eval_1920.webp"), ev, [cv2.IMWRITE_WEBP_QUALITY, 101])  # lossless
    ok &= cv2.imwrite(str(out_dir / "thumb_480.jpg"), th, jpeg_params(THUMB_JPEG_Q))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed in {out_dir}")
    return {
        "width": int(rgb16.shape[1]), "height": int(rgb16.shape[0]),
        "outputs": {
            "eval_1920.jpg": {"w": int(ev.shape[1]), "h": int(ev.shape[0]),
                              "jpeg_quality": EVAL_JPEG_Q, "sampling": "4:4:4"},
            "thumb_480.jpg": {"w": int(th.shape[1]), "h": int(th.shape[0]),
                              "jpeg_quality": THUMB_JPEG_Q, "sampling": "4:4:4"},
        },
    }


def load_rgb16(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    elif img.dtype != np.uint16:
        raise RuntimeError(f"unsupported dtype {img.dtype} in {path}")
    return np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB


def load_linear_rgb(path: Path) -> np.ndarray:
    """Load scene-linear RGB float64 from .exr / .hdr / 16-bit linear PNG.

    16-bit PNG convention: code/65535 = linear reflectance in [0, 1]
    (values > 1 are not representable; use EXR/HDR for highlights).
    OpenCV returns BGR; we convert to RGB.
    """
    suf = path.suffix.lower()
    if suf in (".exr", ".hdr", ".hdri"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cannot read linear image {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(img[..., ::-1].astype(np.float64))
    if suf in (".png", ".tif", ".tiff"):
        rgb16 = load_rgb16(path)
        return rgb16.astype(np.float64) / 65535.0
    raise ValueError(f"unsupported linear image type: {path}")


# --------------------------------------------------------------------------- #
# Contact sheet (synth-specific labels)
# --------------------------------------------------------------------------- #
def build_synth_contact_sheet(manifests: list[dict], frames_root: Path, out_path: Path,
                              cols: int = 6) -> None:
    cw, ch, strip, pad = 320, 180, 48, 4
    n = len(manifests)
    rows = max(1, math.ceil(n / cols))
    head = 32
    sheet = np.full((head + rows * (ch + strip + pad) + pad, cols * (cw + pad) + pad, 3), 28, np.uint8)
    title = f"synth {manifests[0].get('set_id', '') if manifests else ''}  n={n}"
    cv2.putText(sheet, title, (pad, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    for k, rec in enumerate(manifests):
        r, c = divmod(k, cols)
        x0, y0 = pad + c * (cw + pad), head + r * (ch + strip + pad)
        thumb_p = frames_root / rec["sample_id"] / "thumb_480.jpg"
        thumb = cv2.imread(str(thumb_p)) if thumb_p.is_file() else None
        if thumb is not None:
            th, tw = thumb.shape[:2]
            s = min(cw / tw, ch / th)
            thumb = cv2.resize(thumb, (max(1, round(tw * s)), max(1, round(th * s))),
                               interpolation=cv2.INTER_AREA)
            th, tw = thumb.shape[:2]
            sheet[y0 + (ch - th) // 2:y0 + (ch - th) // 2 + th,
                  x0 + (cw - tw) // 2:x0 + (cw - tw) // 2 + tw] = thumb
        lab = rec["label"]
        line1 = f"{lab.get('curve_id', '?')}"
        line2 = rec["sample_id"]
        cv2.putText(sheet, line1[:40], (x0 + 2, y0 + ch + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, line2, (x0 + 2, y0 + ch + 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, jpeg_params(90))


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def _label(curve_id: str, ei, confidence: str = "exact", raw: str = "") -> dict:
    c = CURVES.get(curve_id)
    return {
        "curve_id": curve_id,
        "gamut_id": None if c is None else c.gamut_id,
        "brand": brand_for_curve(curve_id),
        "camera_model": None,
        "camera_candidates": [],
        "ei_iso": ei,
        "wb_kelvin": None,
        "fps": None,
        "raw_label_text": raw or curve_id,
        "label_source": "synthetic",
        "confidence": confidence,
        "needs_confirmation": [],
    }


def make_manifest_rec(set_id: str, sample_id: str, source_id: str, source_class: str,
                      curve_id: str, ei, variant: str, meta: dict,
                      layout_note: str = "") -> dict:
    lab = _label(curve_id, ei, confidence="exact")
    if curve_id == "dji_dlogm" or meta.get("dst_gamut_provisional"):
        lab["needs_confirmation"] = ["formula_unverified"]
    return {
        "sample_id": sample_id,
        "clip_id": sample_id,  # alias for items/eval compatibility
        "set_id": set_id,
        "batch_id": set_id,
        "stem": f"{source_id}__{curve_id}__{variant}",
        "source_id": source_id,
        "source_class": source_class,
        "variant": variant,
        "label": lab,
        "scene": {"tags": ["synthetic", "chart" if source_class == "S-main" else "reencode"]},
        "synth": {
            "source": meta.get("source_kind"),
            "exposure_stops": meta.get("exposure_stops", 0.0),
            "ei": ei,
            "clamp": meta.get("clamp"),
            "layout": layout_note,
            "extra": {k: v for k, v in meta.items()
                      if k not in ("clamp", "exposure_stops", "source_kind")},
        },
        "ingest_warnings": [],
    }


def make_item(rec: dict, art: Path, options: Optional[list[str]] = None) -> dict:
    """Curve item in the dataset schema; the image path is relative to ``art`` (the set folder)."""
    sid = rec["sample_id"]
    curve_id = rec["label"]["curve_id"]
    return {
        "item_id": f"{sid}__curve",
        "clip_id": sid,
        "task": "curve",
        "input": {"image_webp": f"frames/{sid}/eval_1920.webp"},
        "options": list(options or OPTION_SET_CURVE),
        "gold": {
            "primary": curve_id,
            "accept": [curve_id],
            "brand": rec["label"].get("brand"),
            "gamut_id": rec["label"].get("gamut_id"),
            "camera_model": None,
        },
        "twin_of": None,
        "meta": {"subset": rec["set_id"], "source_class": rec["source_class"]},
    }


# --------------------------------------------------------------------------- #
# Chart self-check
# --------------------------------------------------------------------------- #
def chart_self_check(rgb16: np.ndarray, layout: dict, curve_id: str, ei,
                     exposure_stops: float = 0.0, tol_cv: float = 1.0,
                     tol_dxy: float = 1e-3) -> dict:
    """Compare measured grey / black / clip vs expected_code_values (+- tol_cv), and
    the ColorChecker colours decoded with curve + native gamut vs their Rec.709 xy
    (+- tol_dxy), which catches a swapped or wrong gamut matrix."""
    by_name = {p["name"]: p for p in layout["patches"]}
    exp = curves.expected_code_values(curve_id, ei)
    # expected in native gamut after exposure: encode(reflectance * 2^stops)
    gamut_id = curves.curve_gamut(curve_id) or "rec709"

    def expected_neutral(refl: float) -> float:
        """Encode reflectance (Rec.709) with the same path as colorops.encode_chart_linear."""
        from cameralogbench.synth.colorops import encode_linear_to_code, rgb709_to_gamut
        lin = np.array([[[refl, refl, refl]]], dtype=np.float64) * (2.0 ** exposure_stops)
        native = rgb709_to_gamut(lin, gamut_id)
        code, _ = encode_linear_to_code(native, curve_id, ei, apply_sensor_clip=True)
        return float(np.mean(code[0, 0])) * 1023.0

    results = {}
    for name, refl_key in (("grey18", 0.18), ("true_black", 0.0)):
        p = by_name[name]
        meas = measure_patch_cv10(rgb16, tuple(p["bbox"]))
        meas_y = float(np.mean(meas))
        # black stays at curve black for reflectance 0 (exposure does not lift true 0)
        if refl_key == 0.0:
            exp_cv = exp["cv10_float"]["black"]
        else:
            exp_cv = expected_neutral(refl_key)
        results[name] = {
            "measured_cv": round(meas_y, 3),
            "expected_cv": round(exp_cv, 3),
            "delta_cv": round(meas_y - exp_cv, 3),
            "pass": abs(meas_y - exp_cv) <= tol_cv,
        }
    # clip / specular: chart uses 1e4 linear then sensor-clip / code-1 clamp
    p = by_name["specular_clip"]
    meas = measure_patch_cv10(rgb16, tuple(p["bbox"]))
    meas_y = float(np.mean(meas))
    exp_cv_eff = expected_neutral(float(p.get("reflectance") or 1.0e4))
    results["specular_clip"] = {
        "measured_cv": round(meas_y, 3),
        "expected_cv": round(exp_cv_eff, 3),
        "delta_cv": round(meas_y - exp_cv_eff, 3),
        "pass": abs(meas_y - exp_cv_eff) <= tol_cv,
        "whitepaper_clip_cv": exp["cv10_float"]["clip"],
    }
    M_dec = curves.gamut_to_xyz(gamut_id)
    M_709 = curves.gamut_to_xyz("rec709")
    clip_code = exp_cv_eff / 1023.0
    dxy = []
    for p in layout["patches"]:
        if p["kind"] != "color" or not p["name"].startswith("cc_"):
            continue
        code = measure_patch_cv10(rgb16, tuple(p["bbox"])) / 1023.0
        if code.min() <= 2.0 / 1023.0 or code.max() >= clip_code - 2.0 / 1023.0:
            continue
        xyz = M_dec @ np.asarray(curves.decode(curve_id, code, ei), dtype=np.float64)
        ref = M_709 @ np.asarray(p["rgb_linear"], dtype=np.float64)
        dxy.append(float(np.linalg.norm(xyz[:2] / xyz.sum() - ref[:2] / ref.sum())))
    results["colorchecker_xy"] = {
        "n": len(dxy),
        "max_dxy": round(max(dxy), 6) if dxy else None,
        "pass": bool(dxy) and max(dxy) <= tol_dxy,
    }
    results["all_pass"] = all(v["pass"] for v in results.values() if isinstance(v, dict) and "pass" in v)
    return results


# --------------------------------------------------------------------------- #
# Job builders
# --------------------------------------------------------------------------- #
def parse_curves(arg: list[str]) -> list[str]:
    if not arg or arg == ["all"]:
        return list(curves.CURVE_IDS)
    out = []
    for a in arg:
        for part in a.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in curves.CURVE_IDS and part not in EXT2_CURVE_IDS and part != NOT_LOG:
                raise SystemExit(f"unknown curve id: {part}")
            out.append(part)
    return out


def chart_jobs(set_id: str, curve_ids: list[str], exposures: list[float],
               ei_variants: bool, noise: bool, negatives: bool = True) -> list[dict]:
    jobs = []
    for cid in curve_ids:
        eis = [None]
        if cid == "arri_logc3" and ei_variants:
            eis = [400, 800, 1600]
        elif cid == "arri_logc3":
            eis = [800]
        for ei in eis:
            for exp in exposures:
                variant = f"exp{int(exp):+d}"
                if ei is not None:
                    variant += f"_ei{ei}"
                if noise:
                    variant += "_noise"
                jobs.append({
                    "kind": "chart_log", "curve_id": cid, "ei": ei,
                    "exposure_stops": float(exp), "noise": noise, "variant": variant,
                    "source_id": "chart",
                })
    # not_log negatives (exposure 0 only)
    for v in (("rec709", "fake_log") if negatives else ()):
        jobs.append({
            "kind": "chart_not_log", "curve_id": NOT_LOG, "ei": None,
            "exposure_stops": 0.0, "noise": False, "variant": v,
            "source_id": "chart", "not_log_variant": v,
        })
    return jobs


def reencode_sources(pairs: list[list[str]]) -> list[dict]:
    """--reencode IMAGE CURVE pairs -> source records (16-bit Log frames of known curve)."""
    out = []
    for path, curve in pairs:
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"re-encode source not found: {p}")
        if curve not in curves.CURVE_SPECS:
            raise SystemExit(f"unknown curve id for {p}: {curve}")
        if curve in UNVERIFIED_SOURCE_CURVES:
            raise SystemExit(f"{p}: {curve} has no published formula and cannot be a re-encode source")
        out.append({"clip_id": p.stem, "path": str(p), "label": {"curve_id": curve, "ei_iso": None}})
    return out


def reencode_jobs(sources: list[dict], curve_ids: list[str]) -> list[dict]:
    jobs = []
    for src in sources:
        sid = src["clip_id"]
        src_curve = src["label"]["curve_id"]
        src_ei = src["label"].get("ei_iso")
        for cid in curve_ids:
            variant = f"from_{src_curve}"
            jobs.append({
                "kind": "reencode_log", "curve_id": cid, "ei": None,
                "exposure_stops": 0.0, "variant": variant,
                "source_id": sid, "src_rec": src, "src_curve": src_curve, "src_ei": src_ei,
            })
        for v in ("rec709", "fake_log"):
            jobs.append({
                "kind": "reencode_not_log", "curve_id": NOT_LOG, "ei": None,
                "exposure_stops": 0.0, "variant": v,
                "source_id": sid, "src_rec": src, "src_curve": src_curve,
                "src_ei": src_ei, "not_log_variant": v,
            })
    return jobs


def linear_dir_jobs(linear_dir: Path, curve_ids: list[str], exposures: list[float],
                    ei_variants: bool) -> list[dict]:
    files = sorted(
        p for p in linear_dir.iterdir()
        if p.suffix.lower() in (".exr", ".hdr", ".hdri", ".png", ".tif", ".tiff")
    )
    if not files:
        raise SystemExit(f"no linear images in {linear_dir}")
    jobs = []
    for fp in files:
        source_id = fp.stem
        for cid in curve_ids:
            eis = [400, 800, 1600] if (cid == "arri_logc3" and ei_variants) else (
                [800] if cid == "arri_logc3" else [None])
            for ei in eis:
                for exp in exposures:
                    variant = f"exp{int(exp):+d}"
                    if ei is not None:
                        variant += f"_ei{ei}"
                    jobs.append({
                        "kind": "linear_log", "curve_id": cid, "ei": ei,
                        "exposure_stops": float(exp), "variant": variant,
                        "source_id": source_id, "linear_path": fp,
                    })
        for v in ("rec709", "fake_log"):
            jobs.append({
                "kind": "linear_not_log", "curve_id": NOT_LOG, "ei": None,
                "exposure_stops": 0.0, "variant": v,
                "source_id": source_id, "linear_path": fp, "not_log_variant": v,
            })
    return jobs


# --------------------------------------------------------------------------- #
# Process one job
# --------------------------------------------------------------------------- #
def process_job(job: dict, set_id: str, art: Path, chart_cache: dict,
                force: bool, annotations: Optional[list], layout: Optional[dict]) -> dict:
    sid = make_sample_id(set_id, job["source_id"], job["curve_id"], job.get("ei"),
                         job["variant"])
    fdir = art / "frames" / sid
    main_p = fdir / "main16.png"
    source_class = "S-sensor" if job["kind"].startswith("reencode") else "S-main"

    info_p = fdir / "frame_info.json"
    if main_p.is_file() and info_p.is_file() and not force:
        meta = json.loads(info_p.read_text(encoding="utf-8")).get("render_meta") or {
            "source_kind": job["kind"], "exposure_stops": job.get("exposure_stops", 0.0), "clamp": {}}
        meta["skipped_write"] = True
        ensure_webp(fdir / "eval_1920.png")
    else:
        rgb16, meta = _render(job, chart_cache)
        meta["source_kind"] = job["kind"]
        write_frame_versions(fdir, rgb16)
        info_p.write_text(
            json.dumps({"sample_id": sid, "source": "synthetic", "kind": job["kind"],
                        "variant": job["variant"], "preprocess":
                        "cv2.INTER_AREA resize on 16-bit data, then >> 8; no colour management",
                        "render_meta": meta},
                       indent=2) + "\n", encoding="utf-8")
    # self-check and evidence measure the PNG on disk, not the in-memory array
    rgb16 = load_rgb16(main_p)

    # Annotations describe the chart layout, not the label: not_log charts get them too,
    # otherwise the presence of reference_patches in evidence.jsonl reveals the label.
    ann = annotations if job["kind"].startswith("chart") else None
    # stem = sample_id: the variant string ("rec709", "exp+0_ei400") would name the label
    ev = evidence.compute_evidence(str(main_p), sid, batch_id=set_id, stem=sid, annotations=ann)

    self_check = None
    if layout is not None and job["kind"] == "chart_log":
        self_check = chart_self_check(rgb16, layout, job["curve_id"], job.get("ei"),
                                      job.get("exposure_stops", 0.0))

    reasoning = build_reasoning(
        sid, job["curve_id"], ev, ei=job.get("ei"),
        exposure_stops=job.get("exposure_stops", 0.0),
        source_class=source_class,
        sensor_clip_applied=job["kind"] in ("chart_log", "linear_log"),
        extra_meta={"variant": job["variant"], "self_check": self_check,
                    "clamp": meta.get("clamp")},
    )

    man = make_manifest_rec(
        set_id, sid, job["source_id"], source_class, job["curve_id"], job.get("ei"),
        job["variant"], meta,
        layout_note="chart_v0" if layout is not None else "",
    )
    if self_check is not None:
        man["synth"]["self_check"] = self_check
    if job.get("src_curve"):
        man["synth"]["src_curve_id"] = job["src_curve"]
    return {"manifest": man, "evidence": ev, "reasoning": reasoning,
            "item": make_item(man, art, OPTION_SETS_CURVE[job.get("option_set", "v5")]),
            "identity": meta.get("identity_stats")}


def _render(job: dict, chart_cache: dict) -> tuple[np.ndarray, dict]:
    kind = job["kind"]
    if kind == "chart_log":
        lin = chart_cache["linear"]
        # str hash() is salted per process; sha1 keeps noise variants reproducible
        seed = int(hashlib.sha1(job["variant"].encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed) if job.get("noise") else None
        return encode_chart_linear(lin, job["curve_id"], job.get("ei"),
                                   job.get("exposure_stops", 0.0), rng)
    if kind == "chart_not_log":
        return encode_not_log(chart_cache["linear"], job["not_log_variant"],
                              job.get("exposure_stops", 0.0))
    if kind == "linear_log":
        lin = load_linear_rgb(Path(job["linear_path"]))
        return encode_chart_linear(lin, job["curve_id"], job.get("ei"),
                                   job.get("exposure_stops", 0.0), None)
    if kind == "linear_not_log":
        lin = load_linear_rgb(Path(job["linear_path"]))
        return encode_not_log(lin, job["not_log_variant"], job.get("exposure_stops", 0.0))
    if kind == "reencode_log":
        src16 = load_rgb16(Path(job["src_rec"]["path"]))
        dst16, meta = reencode_log_rgb(
            src16, job["src_curve"], job["curve_id"],
            src_ei=job.get("src_ei"), dst_ei=job.get("ei"),
        )
        if meta.get("identity"):
            meta["identity_stats"] = identity_error(src16, dst16)
        meta["exposure_stops"] = 0.0
        return dst16, meta
    if kind == "reencode_not_log":
        src16 = load_rgb16(Path(job["src_rec"]["path"]))
        # decode to linear in source gamut, convert to Rec.709, then not_log encode
        code = u16_to_code(src16)
        lin_src = np.asarray(curves.decode(job["src_curve"], code, job.get("src_ei")),
                             dtype=np.float64)
        src_g = curves.curve_gamut(job["src_curve"]) or "rec709"
        from cameralogbench.synth.colorops import gamut_to_gamut
        lin_709 = np.maximum(gamut_to_gamut(lin_src, src_g, "rec709"), 0.0)
        return encode_not_log(lin_709, job["not_log_variant"], 0.0)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Classical baseline summary
# --------------------------------------------------------------------------- #
def classical_hit_rate(evidences: list[dict], manifests: list[dict]) -> dict:
    gold = {m["sample_id"]: m["label"]["curve_id"] for m in manifests}
    labelled = [e for e in evidences if gold.get(e["clip_id"]) not in (None, NOT_LOG)]
    if not labelled:
        return {"n": 0}
    n = len(labelled)
    top1 = sum(e["classical_prediction"]["ranked"][0] == gold[e["clip_id"]] for e in labelled) / n
    top2 = sum(gold[e["clip_id"]] in e["classical_prediction"]["ranked"][:2] for e in labelled) / n
    return {"n": n, "top1": top1, "top2": top2}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True, dest="set_id", help="output set id (folder under --out-root)")
    ap.add_argument("--source", required=True, choices=("chart", "reencode", "linear_dir"))
    ap.add_argument("--reencode", nargs=2, action="append", default=[], metavar=("IMAGE", "CURVE"),
                    help="source=reencode: a 16-bit Log frame and its curve id (repeatable)")
    ap.add_argument("--linear-dir", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=Path("work/synth"))
    ap.add_argument("--curves", nargs="+", default=["all"])
    ap.add_argument("--ei-variants", action="store_true",
                    help="LogC3: emit EI400/800/1600 (default EI800 only)")
    ap.add_argument("--exposure-offsets", nargs="+", type=float, default=[0.0])
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process only first N jobs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--option-set", choices=sorted(OPTION_SETS_CURVE), default="v5",
                    help="curve options of the items: v5 = 14, ext = 15 (adds Canon Log 2), ext2 = 17")
    ap.add_argument("--no-negatives", action="store_true",
                    help="chart source: skip the two not_log charts")
    args = ap.parse_args(argv)

    set_id = args.set_id
    art = (args.out_root / set_id).resolve()
    art.mkdir(parents=True, exist_ok=True)
    curve_ids = [c for c in parse_curves(args.curves) if c != NOT_LOG]
    exposures = [float(x) for x in args.exposure_offsets]

    chart_cache: dict = {}
    annotations = None
    layout = None

    if args.source == "chart":
        linear, annotations, layout = build_chart(seed=0)
        chart_cache["linear"] = linear
        (art / "chart_layout.json").write_text(
            json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        jobs = chart_jobs(set_id, curve_ids, exposures, args.ei_variants, args.noise,
                          negatives=not args.no_negatives)
    elif args.source == "reencode":
        if not args.reencode:
            raise SystemExit("--reencode IMAGE CURVE required for source=reencode")
        sources = reencode_sources(args.reencode)
        log(f"reencode sources ({len(sources)}): " +
            ", ".join(f"{s['clip_id']}={s['label']['curve_id']}" for s in sources))
        jobs = reencode_jobs(sources, curve_ids)
    else:
        if args.linear_dir is None:
            raise SystemExit("--linear-dir required for source=linear_dir")
        jobs = linear_dir_jobs(args.linear_dir, curve_ids, exposures, args.ei_variants)

    for j in jobs:
        j["option_set"] = args.option_set
    if args.limit is not None:
        jobs = jobs[: args.limit]
    log(f"set={set_id} source={args.source} jobs={len(jobs)} -> {art}")

    results = []
    t0 = time.time()

    def _run(j):
        return process_job(j, set_id, art, chart_cache, args.force, annotations, layout)

    if args.workers <= 1:
        for i, j in enumerate(jobs):
            r = _run(j)
            results.append(r)
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                log(f"  {i + 1}/{len(jobs)}  {r['manifest']['sample_id']}  "
                    f"{r['manifest']['label']['curve_id']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                results.append(r)
                if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                    log(f"  {i + 1}/{len(jobs)}  {r['manifest']['sample_id']}")

    # stable order by sample_id
    results.sort(key=lambda r: r["manifest"]["sample_id"])
    manifests = [r["manifest"] for r in results]
    evidences = [r["evidence"] for r in results]
    reasonings = [r["reasoning"] for r in results]
    items = [r["item"] for r in results]

    write_jsonl(art / "manifest.jsonl", manifests)
    write_jsonl(art / "evidence.jsonl", evidences)
    write_jsonl(art / "reasoning.jsonl", reasonings)
    write_jsonl(art / "annotations.jsonl", items)

    build_synth_contact_sheet(manifests, art / "frames", art / "contact_sheet.jpg")

    # reports
    hit = classical_hit_rate(evidences, manifests)
    identity_rows = [r["identity"] for r in results if r.get("identity")]
    self_checks = [m["synth"]["self_check"] for m in manifests
                   if m.get("synth", {}).get("self_check")]
    self_pass = sum(1 for s in self_checks if s.get("all_pass")) if self_checks else 0

    report = {
        "set_id": set_id,
        "source": args.source,
        "n_samples": len(manifests),
        "n_items": len(items),
        "classical_on_set": hit,
        "chart_self_check_pass": f"{self_pass}/{len(self_checks)}" if self_checks else None,
        "identity_reencodes": len(identity_rows),
        "identity_max_abs_cv10_mean": (
            float(np.mean([x["max_abs_cv10"] for x in identity_rows])) if identity_rows else None
        ),
        "identity_mse_cv10_mean": (
            float(np.mean([x["mse_cv10"] for x in identity_rows])) if identity_rows else None
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "out": repo_rel(art),
    }
    (art / "synth_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if self_checks:
        log("\nchart self-check (tol +-1 CV, exposure-aware):")
        log(f"  {self_pass}/{len(self_checks)} samples all_pass")
        for m in manifests:
            sc = m.get("synth", {}).get("self_check")
            var = m.get("variant", "")
            if not sc or not (var == "exp+0" or var.startswith("exp+0_")):
                continue
            g, b, c = sc["grey18"], sc["true_black"], sc["specular_clip"]
            log(f"  {m['label']['curve_id']:16s} ei={m['label'].get('ei_iso')}  "
                f"black d={b['delta_cv']:+.2f}  grey d={g['delta_cv']:+.2f}  "
                f"clip d={c['delta_cv']:+.2f}  pass={sc['all_pass']}")

    log(f"\nclassical baseline on {set_id}: n={hit.get('n')}  "
        f"top-1 {hit.get('top1', 0):.1%}  top-2 {hit.get('top2', 0):.1%}")
    if identity_rows:
        log(f"identity reencodes: n={len(identity_rows)}  "
            f"mean max|d|={report['identity_max_abs_cv10_mean']:.3f} CV  "
            f"mean MSE={report['identity_mse_cv10_mean']:.4f} CV^2")
    log(f"wrote {art}  ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
