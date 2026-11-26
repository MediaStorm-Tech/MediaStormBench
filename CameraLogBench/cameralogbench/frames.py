"""Extract the benchmark's model input from videos: 8-bit lossless WebP frames at fixed fractions of a clip.

    python -m cameralogbench.frames --videos path/to/clips --out-root work/mydata --subset mine \
        [--labels labels.csv] [--option-set ext2] [--workers 3]

Per clip, three frames at 25 %, 50 % and 75 % of its duration (frame index int(n * fraction)) are decoded at
native size to 16-bit RGB (ffmpeg rgb48), downscaled to a long side of 1920 px with cv2 INTER_AREA on the
16-bit data, shifted to 8 bits (>> 8) and written as lossless WebP: no LUT, colour transform or ICC profile.
This is the recipe of the released frames.

Writes ``<out-root>/frames/<subset>/<clip_id>/p25.webp`` (p50, p75). With ``--labels`` it also writes
``<out-root>/annotations/<subset>.jsonl`` in the dataset's schema (paths relative to ``<out-root>``), so the
runner can be pointed at your own footage with ``--data-root <out-root>``.

labels.csv columns: ``file`` (video file name), ``curve`` (taxonomy id such as ``sony_slog3``, or a common
name such as ``S-Log3``; ``not_log`` for display-referred footage), optional ``brand`` (taxonomy brand id;
default: the curve's brand), ``camera_model`` and ``clip_id`` (default: the video's file stem).
Items per clip: ``curve``, ``camera`` (when the brand is known) and ``curve_absent`` (Log clips only: the same
frames with the correct option removed; gold = undetermined).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np

from cameralogbench._io import write_jsonl
from cameralogbench.taxonomy import (BRANDS, CURVES, NOT_LOG, OPTION_SET_BRAND, OPTION_SETS_CURVE, UNDETERMINED,
                                     normalize_curve_label)

EVAL_LONG, THUMB_LONG = 1920, 480
EVAL_JPEG_Q, THUMB_JPEG_Q = 92, 90
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SHOWINFO_RE = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+(?:e-?\d+)?)")
SHOWINFO_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)\b")
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".mxf", ".mkv")

WEBP_LOSSLESS = [cv2.IMWRITE_WEBP_QUALITY, 101]  # quality > 100 = lossless


def ensure_webp(png: Path) -> Path:
    """Write <stem>.webp next to an 8-bit PNG (lossless, bit-identical) when missing or older."""
    webp = png.with_suffix(".webp")
    if not webp.is_file() or webp.stat().st_mtime < png.stat().st_mtime:
        im = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if im is None or not cv2.imwrite(str(webp), im, WEBP_LOSSLESS):
            raise RuntimeError(f"failed to write {webp}")
    return webp


def jpeg_params(q: int) -> list[int]:
    return [cv2.IMWRITE_JPEG_QUALITY, q,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def load16(path: Path) -> np.ndarray:
    """16-bit BGR (H, W, 3). 8-bit inputs are expanded (v * 257)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    elif img.dtype != np.uint16:
        raise RuntimeError(f"unsupported dtype {img.dtype} in {path}")
    return np.ascontiguousarray(img)


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def to8(img16: np.ndarray) -> np.ndarray:
    return (img16 >> 8).astype(np.uint8)


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
def _ffmpeg_base(video: Path) -> list[str]:
    return [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "info", "-i", str(video),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough"]


def extract_frame16(video: Path, n: int) -> tuple[np.ndarray, Optional[float]]:
    """Decode frame n at native size as rgb48; returns (BGR uint16, pts_s)."""
    cmd = _ffmpeg_base(video) + ["-vf", f"select=eq(n\\,{n}),showinfo", "-frames:v", "1",
                                 "-pix_fmt", "rgb48be", "-f", "rawvideo", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True)
    err = res.stderr.decode("utf-8", "replace")
    info_lines = [l for l in err.splitlines() if "showinfo" in l.lower() and "pts_time" in l]
    if res.returncode != 0 or not info_lines:
        raise RuntimeError(f"ffmpeg extract frame {n} failed (rc={res.returncode}): {err.splitlines()[-3:]}")
    m, s = SHOWINFO_RE.search(info_lines[0]), SHOWINFO_SIZE_RE.search(info_lines[0])
    w, h = int(s.group(1)), int(s.group(2))
    if len(res.stdout) != w * h * 6:
        raise RuntimeError(f"ffmpeg returned {len(res.stdout)} bytes, expected {w * h * 6} for {w}x{h} rgb48")
    rgb = np.frombuffer(res.stdout, dtype=">u2").reshape(h, w, 3).astype(np.uint16)
    return np.ascontiguousarray(rgb[..., ::-1]), (float(m.group(3)) if m else None)


def count_frames(video: Path) -> int:
    """Number of decodable video frames (decodes the stream once)."""
    n, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
    return int(n)


# --------------------------------------------------------------------------- #
# Frames at fixed fractions of the clip, chosen by time position only
# --------------------------------------------------------------------------- #
MULTI_FRACTIONS = (0.25, 0.5, 0.75)


def multi_index(n_frames: int, frac: float) -> int:
    """Frame index for a fraction of the clip; 0.5 -> n // 2, the middle frame."""
    return min(max(0, int(n_frames * frac)), max(0, n_frames - 1))


def eval_frame(img16_bgr: np.ndarray) -> np.ndarray:
    """16-bit BGR frame -> the 8-bit model input (long side 1920, INTER_AREA on 16-bit data, >> 8)."""
    return to8(resize_long(img16_bgr, EVAL_LONG))


def extract_clip(video: Path, out_dir: Path, force: bool = False) -> list[str]:
    """Write p25 / p50 / p75 .webp for one clip; returns the positions written or found."""
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = [f"p{round(f * 100):02d}" for f in MULTI_FRACTIONS]
    if not force and all((out_dir / f"{p}.webp").is_file() for p in positions):
        return positions
    n = count_frames(video)
    for frac, pos in zip(MULTI_FRACTIONS, positions):
        img16, _ = extract_frame16(video, multi_index(n, frac))
        if not cv2.imwrite(str(out_dir / f"{pos}.webp"), eval_frame(img16), WEBP_LOSSLESS):
            raise RuntimeError(f"failed to write {out_dir / pos}.webp")
    return positions


# --------------------------------------------------------------------------- #
# Annotations from labels.csv
# --------------------------------------------------------------------------- #
def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "clip"


def read_labels(path: Path) -> dict[str, dict]:
    """file name -> {curve, brand, camera_model, clip_id}."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for ln, row in enumerate(csv.DictReader(f), 2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            name = row.get("file")
            curve = normalize_curve_label(row.get("curve", ""))
            if not name or curve is None:
                raise SystemExit(f"{path}:{ln}: need a 'file' and a known 'curve' (got {row.get('curve')!r})")
            brand = row.get("brand") or CURVES[curve].brand
            if brand and brand not in BRANDS:
                raise SystemExit(f"{path}:{ln}: unknown brand {brand!r}; expected one of {sorted(BRANDS)}")
            out[name] = {"curve": curve, "brand": brand or None, "camera_model": row.get("camera_model") or None,
                         "clip_id": row.get("clip_id") or None}
    return out


def build_items(clip_id: str, label: dict, frames: list[str], positions: list[str], subset: str,
                curve_options: list[str]) -> list[dict]:
    """Items of one clip in the dataset schema (paths relative to the data root)."""
    curve = label["curve"]
    if curve not in curve_options:
        raise SystemExit(f"{clip_id}: curve {curve!r} is not in the chosen option set; use --option-set ext2")
    mid = frames[positions.index("p50")] if "p50" in positions else frames[len(frames) // 2]
    inp = {"image_webp": mid, "frames_webp": list(frames), "frame_positions": list(positions)}
    c = CURVES[curve]
    gold = {"primary": curve, "accept": [curve], "brand": label["brand"], "gamut_id": c.gamut_id,
            "camera_model": label["camera_model"]}
    meta = {"subset": subset, "source_class": "user"}
    items = [{"item_id": f"{clip_id}__curve", "clip_id": clip_id, "task": "curve", "input": dict(inp),
              "options": list(curve_options), "gold": gold, "twin_of": None, "meta": dict(meta)}]
    if label["brand"]:
        b = label["brand"]
        items.append({"item_id": f"{clip_id}__camera", "clip_id": clip_id, "task": "camera", "input": dict(inp),
                      "options": list(OPTION_SET_BRAND),
                      "gold": {"primary": b, "accept": [b], "brand": b, "gamut_id": None,
                               "camera_model": label["camera_model"]},
                      "twin_of": None, "meta": dict(meta)})
    if curve != NOT_LOG:
        items.append({"item_id": f"{clip_id}__curve__absent", "clip_id": clip_id, "task": "curve_absent",
                      "input": dict(inp), "options": [o for o in curve_options if o != curve],
                      "gold": {"primary": UNDETERMINED, "accept": [UNDETERMINED], "brand": None, "gamut_id": None,
                               "camera_model": None},
                      "twin_of": f"{clip_id}__curve", "meta": dict(meta)})
    return items


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def find_videos(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS and not q.name.startswith("."))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True, help="video files and/or folders (searched recursively)")
    ap.add_argument("--out-root", required=True, help="dataset folder to write frames/ and annotations/ into")
    ap.add_argument("--subset", default="mine", help="subset name (folder under frames/, annotation file name)")
    ap.add_argument("--labels", default=None, help="labels.csv; writes annotations/<subset>.jsonl")
    ap.add_argument("--option-set", choices=sorted(OPTION_SETS_CURVE), default="ext2",
                    help="curve options of the items: v5 = 14, ext = 15, ext2 = 17 (as the real-footage subset)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    videos = find_videos(a.videos)
    labels = read_labels(Path(a.labels)) if a.labels else {}
    if labels:
        unlabelled = [v.name for v in videos if v.name not in labels]
        if unlabelled:
            print(f"[warn] {len(unlabelled)} videos have no row in {a.labels} and get no items, "
                  f"e.g. {unlabelled[0]}", file=sys.stderr)
    root = Path(a.out_root)
    ids = {}
    for v in videos:
        cid = (labels.get(v.name) or {}).get("clip_id") or safe_id(v.stem)
        if cid in ids.values():
            raise SystemExit(f"duplicate clip_id {cid!r} ({v}); set clip_id in labels.csv")
        ids[v] = cid
    print(f"{len(videos)} videos -> {root / 'frames' / a.subset}", flush=True)
    done: dict[Path, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(extract_clip, v, root / "frames" / a.subset / ids[v], a.force): v for v in videos}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                done[v] = fut.result()
                print(f"  ok    {ids[v]}", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue with the other clips
                print(f"  ERROR {v}: {e}", flush=True)
    if labels:
        items = []
        for v in videos:
            if v in done and v.name in labels:
                cid = ids[v]
                frames = [f"frames/{a.subset}/{cid}/{p}.webp" for p in done[v]]
                items += build_items(cid, labels[v.name], frames, done[v], a.subset, OPTION_SETS_CURVE[a.option_set])
        out = root / "annotations" / f"{a.subset}.jsonl"
        write_jsonl(out, items)
        print(f"{len(items)} items -> {out}")
    return 0 if len(done) == len(videos) else 1


if __name__ == "__main__":
    sys.exit(main())
