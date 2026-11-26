"""Graded ("not Log") negatives made from real Log frames.

A model that always names some Log curve is never penalised on Log footage, so part of the real Log clips is
turned into display-referred footage the way a colourist would: decode the labelled curve with its official
formula, convert its native gamut to Rec.709, then apply one of three looks. Gold = not_log. Every graded
item has a Log twin with identical content, so content alone cannot answer it.

Looks (all looks first get balance(): partial grey-world white balance + exposure to a normal mid-grey):
  rec709  soft highlight knee + BT.709 OETF (a plain "Log to Rec.709" conversion)
  look    rec709 + S-curve contrast, +15 % saturation, cool shadows / warm highlights (a grading filter)
  flat    the synthetic 'fake log' grey look (cameralogbench.synth.colorops.fake_log_encode): low contrast and
          lifted blacks that *resemble* Log but follow no vendor curve - the hard negative
Per curve, the k-th selected clip gets LOOK_PLAN[k].

The released graded frames were made from the 16-bit frames. The CLI grades any frame you have:

    python -m cameralogbench.graded --input p50.webp --curve sony_slog3 --look rec709 --out graded.webp

An 8-bit input works but carries 8-bit quantisation into the graded result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cameralogbench import curves
from cameralogbench.frames import WEBP_LOSSLESS, eval_frame, load16
from cameralogbench.synth.colorops import fake_log_encode, gamut_to_gamut, rec709_oetf

LOOKS = ("rec709", "look", "flat")
# per curve, the k-th selected clip gets LOOK_PLAN[k]: one fake-Log grey ("flat") per curve, the rest split
# between the plain conversion and the graded look
LOOK_PLAN = ("flat", "rec709", "look", "rec709", "look", "rec709", "look", "rec709", "look", "rec709")


def soft_knee(x: np.ndarray, knee: float = 0.7, width: float = 0.6) -> np.ndarray:
    """Identity below `knee`, smooth roll-off above (asymptote knee + width) - a camera-style highlight shoulder."""
    over = np.maximum(x - knee, 0.0)
    return np.where(x <= knee, x, knee + over / (1.0 + over / width))


def balance(lin: np.ndarray, wb_strength: float = 0.8, target_median: float = 0.16) -> np.ndarray:
    """What a colourist does before any look: neutralise the cast (partial grey-world on mid-tones) and set
    exposure so the median luminance lands near a normal mid-grey. Gains are clamped to stay plausible."""
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    lo, hi = np.percentile(luma, [5, 95])
    mid = (luma >= lo) & (luma <= hi)
    means = lin[mid].mean(axis=0) + 1e-6
    gains = np.clip((means[1] / means) ** wb_strength, 0.5, 2.0)
    lin = lin * gains
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    scale = float(np.clip(target_median / max(float(np.median(luma)), 1e-6), 0.25, 4.0))
    return lin * scale


def grade(code_rgb: np.ndarray, curve_id: str, look: str) -> np.ndarray:
    """Normalised Log code (H, W, 3 RGB) -> display code [0, 1] for one look."""
    lin = np.asarray(curves.decode(curve_id, code_rgb), dtype=np.float64)
    lin = np.maximum(gamut_to_gamut(lin, curves.curve_gamut(curve_id) or "rec709", "rec709"), 0.0)
    lin = balance(lin)
    if look == "flat":
        return np.clip(fake_log_encode(lin), 0.0, 1.0)
    y = np.clip(rec709_oetf(soft_knee(lin)), 0.0, 1.0)
    if look == "rec709":
        return y
    # "look": S-curve around 0.5, saturation, split toning
    y = 0.5 + (y - 0.5) * 1.15 - 0.15 * (y - 0.5) ** 3 * 4
    luma = 0.2126 * y[..., :1] + 0.7152 * y[..., 1:2] + 0.0722 * y[..., 2:]
    y = luma + (y - luma) * 1.15
    shadows, highlights = np.clip(1 - 2 * luma, 0, 1), np.clip(2 * luma - 1, 0, 1)
    y = y + shadows * np.array([-0.02, 0.005, 0.025]) + highlights * np.array([0.025, 0.008, -0.02])
    return np.clip(y, 0.0, 1.0)


def grade_file(src: Path, curve_id: str, look: str, out: Path) -> None:
    """Grade one Log frame (any bit depth) and write it like the released frames (1920 wide, 8-bit, WebP)."""
    code = load16(src)[..., ::-1].astype(np.float64) / 65535.0  # BGR16 -> RGB code
    disp = grade(code, curve_id, look)
    img16 = np.ascontiguousarray(np.round(disp * 65535.0).astype(np.uint16)[..., ::-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), eval_frame(img16), WEBP_LOSSLESS if out.suffix.lower() == ".webp" else []):
        raise RuntimeError(f"failed to write {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Log frame (PNG / WebP / TIFF, 8 or 16 bit)")
    ap.add_argument("--curve", required=True, choices=sorted(curves.CURVE_SPECS), help="the frame's Log curve")
    ap.add_argument("--look", required=True, choices=LOOKS)
    ap.add_argument("--out", required=True, type=Path, help="output image (.webp = lossless)")
    a = ap.parse_args(argv)
    grade_file(a.input, a.curve, a.look, a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
