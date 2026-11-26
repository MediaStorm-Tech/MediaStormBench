"""Prepare downloaded HDR sources as scene-linear Rec.709 EXR for ``--source linear_dir``.

Per source (rows of ``data/synth_sources/sources.jsonl``):

1. Colour: Fairchild EXRs are Nikon D2x camera RGB -> CIE XYZ with the D65-normalised
   matrix from Fairchild's "Nikon D2x Characterization" (markfairchild.org/HDRPS) ->
   Rec.709 linear.  Poly Haven EXRs are already Rec.709 linear.
   Negative Rec.709 values (colours outside 709) are kept: the encoder's
   709 -> native-gamut transform is linear, so wide-gamut colours survive.
2. Framing: Fairchild kept as shot, long edge capped at ``--max-long``.  Poly Haven
   equirectangular panoramas are reprojected to a 16:9 pinhole view (HFOV 75 deg,
   pitch 0) at native angular resolution; the yaw is the one of 12 candidates with
   the most log-luminance detail.
3. Exposure: auto-exposure to Reinhard key 0.18 (log-average luminance -> 0.18).
   The scale factor is recorded; no per-source jitter.

Writes ``<out>/<source_id>.exr`` (half float) and ``<out>/prep.jsonl``.

Example::

    python -m cameralogbench.synth.prep_linear --out data/synth_sources/linear_v1 --workers 4
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cameralogbench import curves
from cameralogbench._io import ROOT, read_jsonl, repo_rel, write_jsonl

PREP_VERSION = "linear_v1"
KEY = 0.18
HFOV_DEG = 75.0
YAW_CANDIDATES = 12

# Nikon D2x camera RGB -> CIE 1931 XYZ, D65-normalised (Fairchild, D2x Characterization).
D2X_TO_XYZ = np.array([
    [0.4024, 0.4610, 0.0871],
    [0.1904, 0.7646, 0.0450],
    [-0.0249, 0.1264, 0.9873],
])
LUMA_709 = np.array([0.2126, 0.7152, 0.0722])


def read_exr_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img[..., :3][..., ::-1].astype(np.float32))


def d2x_to_709(rgb: np.ndarray) -> np.ndarray:
    m = curves.xyz_to_gamut("rec709") @ D2X_TO_XYZ
    return (rgb.reshape(-1, 3) @ m.T.astype(np.float32)).reshape(rgb.shape)


def cap_long_edge(rgb: np.ndarray, max_long: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_long / max(h, w)
    if s >= 1.0:
        return rgb
    return cv2.resize(rgb, (round(w * s) // 2 * 2, round(h * s) // 2 * 2),
                      interpolation=cv2.INTER_AREA)


def equirect_view(eq: np.ndarray, yaw_deg: float, pitch_deg: float, hfov_deg: float,
                  out_w: int, out_h: int) -> np.ndarray:
    """Pinhole view of an equirectangular panorama (bilinear, horizontal wrap)."""
    H, W = eq.shape[:2]
    tx = math.tan(math.radians(hfov_deg) / 2)
    ty = tx * out_h / out_w
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) / out_w * 2 - 1
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) / out_h * 2 - 1
    x, y = np.meshgrid(xs * tx, -ys * ty)
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, q = math.radians(pitch_deg), math.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(q), 0, math.sin(q)], [0, 1, 0], [-math.sin(q), 0, math.cos(q)]])
    d = d @ (ry @ rx).T
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    map_x = ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32)
    return cv2.remap(eq, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def detail_score(rgb: np.ndarray) -> float:
    y = np.maximum(rgb @ LUMA_709.astype(np.float32), 0) + 1e-4
    ly = np.log2(y).astype(np.float32)
    gx = cv2.Sobel(ly, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ly, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


def pick_yaw(eq: np.ndarray) -> tuple[float, list[float]]:
    small = cv2.resize(eq, (1024, 512), interpolation=cv2.INTER_AREA)
    yaws = [i * 360.0 / YAW_CANDIDATES for i in range(YAW_CANDIDATES)]
    scores = [detail_score(equirect_view(small, yw, 0.0, HFOV_DEG, 192, 108)) for yw in yaws]
    return yaws[int(np.argmax(scores))], [round(s, 4) for s in scores]


def log_average(rgb: np.ndarray) -> float:
    y = rgb @ LUMA_709.astype(np.float32)
    y = y[y > 0]
    return float(np.exp(np.mean(np.log(y.astype(np.float64)))))


def prep_one(row: dict, out: Path, max_long: int, force: bool) -> dict[str, Any]:
    dst = out / f"{row['source_id']}.exr"
    rec: dict[str, Any] = {"source_id": row["source_id"], "dataset": row["dataset"],
                           "src_path": row["path"], "src_sha1": row.get("sha1"),
                           "prep_version": PREP_VERSION, "key": KEY}
    rgb = read_exr_rgb(ROOT / row["path"])
    if row["colour"] == "nikon_d2x_camera_rgb":
        rgb = d2x_to_709(rgb)
        rec["colour_transform"] = "D2x camera RGB -> XYZ (Fairchild D65 matrix) -> Rec.709"
    elif row["colour"] == "rec709_linear":
        rec["colour_transform"] = "none (Rec.709 linear)"
    else:
        raise ValueError(f"unknown colour {row['colour']!r}")

    if row["projection"] == "equirectangular":
        H, W = rgb.shape[:2]
        out_w = round(W * HFOV_DEG / 360.0) // 2 * 2
        out_h = round(out_w * 9 / 16) // 2 * 2
        yaw, scores = pick_yaw(rgb)
        rgb = equirect_view(rgb, yaw, 0.0, HFOV_DEG, out_w, out_h)
        rec["view"] = {"yaw_deg": yaw, "pitch_deg": 0.0, "hfov_deg": HFOV_DEG,
                       "yaw_scores": scores}
    rgb = cap_long_edge(rgb, max_long)

    la = log_average(rgb)
    scale = KEY / la
    rgb = rgb * np.float32(scale)
    y = rgb @ LUMA_709.astype(np.float32)
    pct = np.percentile(y, [1, 50, 99, 99.9])
    rec.update({
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "log_average_raw": la, "exposure_scale": scale,
        "luma_p1_p50_p99_p999": [round(float(v), 5) for v in pct],
        "luma_max": round(float(y.max()), 3),
        "stops_above_key_p999": round(math.log2(max(float(pct[3]), 1e-9) / KEY), 2),
        "negative_709_fraction": round(float(np.mean(np.any(rgb < 0, axis=-1))), 6),
    })
    if force or not dst.is_file():
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        ok = cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        if not ok:
            raise RuntimeError(f"cannot write {dst}")
    rec["path"] = repo_rel(dst)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", type=Path, default=ROOT / "data/synth_sources/sources.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synth_sources/linear_v1")
    ap.add_argument("--max-long", type=int, default=3840)
    ap.add_argument("--only", nargs="*", default=None, help="source_ids to process")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.sources)
    if args.only:
        rows = [r for r in rows if r["source_id"] in set(args.only)]
    recs, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(prep_one, r, out, args.max_long, args.force): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            sid = futs[fut]["source_id"]
            try:
                rec = fut.result()
                recs.append(rec)
                print(f"  {i}/{len(rows)} {sid}  {rec['size'][0]}x{rec['size'][1]}  "
                      f"scale={rec['exposure_scale']:.4g}  p99.9=+{rec['stops_above_key_p999']} st",
                      flush=True)
            except Exception as e:
                failed.append(sid)
                print(f"  {i}/{len(rows)} FAIL {sid}: {e}", flush=True)
    recs.sort(key=lambda r: r["source_id"])
    if not args.only:
        write_jsonl(out / "prep.jsonl", recs)
    print(f"prepared {len(recs)}  failed {len(failed)} {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
