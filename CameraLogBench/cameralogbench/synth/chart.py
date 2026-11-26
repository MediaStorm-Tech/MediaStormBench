"""Formula-driven calibration chart in scene-linear Rec.709 / D65.

Layout (1920 x 1080), designed so every patch used by evidence.reference_fit
and the chart self-check has a known bbox:

  * large 18 % grey card
  * ColorChecker Classic 24 (X-Rite sRGB target values, linearized)
  * grey ladder 0 -> 90 % white -> +6 stops above 18 % grey
  * true-black void
  * high-saturation red / cyan / magenta strips (gamut-edge colours)
  * white-noise texture region
  * specular clip patch (very bright) for highlight plateau

ColorChecker source
-------------------
X-Rite ColorChecker Classic printed target sRGB reference values
(widely republished; see e.g. BabelColor / X-Rite ColorData sheet).
Decoded to linear with the IEC 61966-2-1 sRGB transfer function and treated
as Rec.709 scene-linear reflectance under D65 (perfect diffuser = 1.0).

The neutral-row annotations handed to ``evidence.reference_fit`` carry the
Rec.709 luminance of the *rendered* linear value (e.g. black_2 = 0.0343), not
the nominal Munsell reflectance (0.031): the fit compares measured codes with
``encode(reflectance)``, so any mismatch shows up as a spurious neutral RMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

W, H = 1920, 1080

# X-Rite ColorChecker Classic sRGB 8-bit reference (row-major, 6 cols x 4 rows).
# Source: X-Rite ColorChecker Classic target values / BabelColor Average.
COLORCHECKER_SRGB_8BIT: list[tuple[int, int, int]] = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108),
    (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121),
    (85, 85, 85), (52, 52, 52),
]
COLORCHECKER_NAMES: list[str] = [
    "dark_skin", "light_skin", "blue_sky", "foliage", "blue_flower", "bluish_green",
    "orange", "purplish_blue", "moderate_red", "purple", "yellow_green", "orange_yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white_9.5", "neutral_8", "neutral_6.5", "neutral_5", "neutral_3.5", "black_2",
]
# Nominal (Munsell) reflectance of the neutral row, for documentation only.
# Annotations use the luminance of the rendered sRGB-decoded value instead
# (0.896 / 0.578 / 0.352 / 0.194 / 0.091 / 0.034).
CC_NEUTRAL_REFLECTANCE: dict[str, float] = {
    "white_9.5": 0.90, "neutral_8": 0.59, "neutral_6.5": 0.36,
    "neutral_5": 0.198, "neutral_3.5": 0.09, "black_2": 0.031,
}
REC709_LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB decoding (8-bit 0..255 -> [0, 1] linear)."""
    x = np.asarray(u8, dtype=np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True)
class Patch:
    name: str
    kind: str                       # neutral | color | black | specular | texture
    bbox: tuple[int, int, int, int]  # x, y, w, h
    reflectance: float | None = None
    rgb_linear: tuple[float, float, float] | None = None  # Rec.709 scene-linear


def _fill(img: np.ndarray, bbox: tuple[int, int, int, int], rgb) -> None:
    x, y, w, h = bbox
    img[y:y + h, x:x + w] = np.asarray(rgb, dtype=np.float64)


def build_chart(seed: int = 0) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Return (linear_rgb HxWx3 float64, annotations for evidence, layout meta)."""
    img = np.full((H, W, 3), 0.12, dtype=np.float64)  # mid-grey surround
    patches: list[Patch] = []
    rng = np.random.default_rng(seed)

    # --- 18 % grey card (large) ---
    grey_bbox = (60, 60, 420, 360)
    _fill(img, grey_bbox, (0.18, 0.18, 0.18))
    patches.append(Patch("grey18", "neutral", grey_bbox, reflectance=0.18,
                         rgb_linear=(0.18, 0.18, 0.18)))

    # --- true black void ---
    black_bbox = (520, 60, 200, 360)
    _fill(img, black_bbox, (0.0, 0.0, 0.0))
    patches.append(Patch("true_black", "black", black_bbox, reflectance=0.0,
                         rgb_linear=(0.0, 0.0, 0.0)))

    # --- ColorChecker Classic 6x4 ---
    cc_origin = (760, 60)
    pw, ph, gap = 140, 90, 8
    cc_linear = srgb_to_linear(np.array(COLORCHECKER_SRGB_8BIT, dtype=np.float64))
    for i, (name, rgb) in enumerate(zip(COLORCHECKER_NAMES, cc_linear)):
        r, c = divmod(i, 6)
        bbox = (cc_origin[0] + c * (pw + gap), cc_origin[1] + r * (ph + gap), pw, ph)
        rgb_t = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        _fill(img, bbox, rgb_t)
        if name in CC_NEUTRAL_REFLECTANCE:
            patches.append(Patch(f"cc_{name}", "neutral", bbox,
                                 reflectance=float(rgb @ REC709_LUMA), rgb_linear=rgb_t))
        else:
            patches.append(Patch(f"cc_{name}", "color", bbox, rgb_linear=rgb_t))

    # --- specular clip patch (far above every curve's linear_at_code_1) ---
    # 1e4 linear is enough to saturate Log3G10 / BMD / LogC4 headroom; sensor-clip
    # curves (LogC3, D-Log) are hard-clamped to the documented clip in colorops.
    clip_bbox = (1680, 60, 180, 360)
    clip_lin = 1.0e4
    _fill(img, clip_bbox, (clip_lin, clip_lin, clip_lin))
    patches.append(Patch("specular_clip", "specular", clip_bbox, reflectance=clip_lin,
                         rgb_linear=(clip_lin, clip_lin, clip_lin)))

    # --- grey ladder: 0, then 0.18*2^k for k=-6..+6, then 0.90 ---
    ladder_y, ladder_h = 460, 140
    stops = list(range(-6, 7))  # -6 .. +6 around 0.18
    values = [0.0] + [0.18 * (2.0 ** s) for s in stops] + [0.90]
    n = len(values)
    cell_w = (W - 120) // n
    for i, v in enumerate(values):
        bbox = (60 + i * cell_w, ladder_y, cell_w - 4, ladder_h)
        _fill(img, bbox, (v, v, v))
        kind = "black" if v == 0.0 else ("specular" if v > 0.90 else "neutral")
        patches.append(Patch(f"ladder_{i}_r{v:.4g}", kind, bbox, reflectance=v,
                             rgb_linear=(v, v, v)))

    # --- saturated primaries / secondaries (Rec.709 gamut edge at ~0.4 linear) ---
    sat_y, sat_h = 640, 160
    sats = [
        ("sat_red", (0.55, 0.02, 0.02)),
        ("sat_cyan", (0.02, 0.45, 0.50)),
        ("sat_magenta", (0.45, 0.02, 0.50)),
    ]
    sw = 400
    for i, (name, rgb) in enumerate(sats):
        bbox = (60 + i * (sw + 40), sat_y, sw, sat_h)
        _fill(img, bbox, rgb)
        patches.append(Patch(name, "color", bbox, rgb_linear=rgb))

    # --- white-noise texture ---
    tex_bbox = (1380, 640, 480, 160)
    x, y, w, h = tex_bbox
    noise = rng.normal(0.18, 0.04, size=(h, w, 3))
    noise = np.clip(noise, 0.0, 1.0)
    img[y:y + h, x:x + w] = noise
    patches.append(Patch("noise_texture", "texture", tex_bbox, reflectance=0.18))

    # --- bottom margin label strip stays surround grey ---

    annotations = _annotations_for_evidence(patches, img)
    layout = {
        "width": W, "height": H, "working_space": "rec709_linear_d65",
        "colorchecker_source": (
            "X-Rite ColorChecker Classic printed-target sRGB reference values "
            "(BabelColor / X-Rite ColorData); linearized with IEC 61966-2-1 sRGB "
            "EOTF inverse; treated as Rec.709 scene-linear reflectance under D65"
        ),
        "patches": [
            {"name": p.name, "kind": p.kind, "bbox": list(p.bbox),
             "reflectance": p.reflectance,
             "rgb_linear": list(p.rgb_linear) if p.rgb_linear else None}
            for p in patches
        ],
    }
    return img, annotations, layout


def _xy_from_rgb709(rgb: tuple[float, float, float]) -> list[float] | None:
    from cameralogbench.curves import gamut_to_xyz
    M = gamut_to_xyz("rec709")
    xyz = M @ np.asarray(rgb, dtype=np.float64)
    s = float(xyz.sum())
    if s < 1e-12:
        return None
    return [float(xyz[0] / s), float(xyz[1] / s)]


def _annotations_for_evidence(patches: list[Patch], img: np.ndarray) -> list[dict[str, Any]]:
    """Subset of patches evidence.reference_fit cares about."""
    out: list[dict[str, Any]] = []
    keep_neutral = {"grey18", "cc_white_9.5", "cc_neutral_8", "cc_neutral_6.5",
                    "cc_neutral_5", "cc_neutral_3.5", "cc_black_2", "true_black"}
    keep_color = {"cc_red", "cc_green", "cc_blue", "cc_yellow", "cc_magenta", "cc_cyan",
                  "cc_orange", "cc_purplish_blue"}
    for p in patches:
        if p.kind == "neutral" and p.name in keep_neutral and p.reflectance is not None:
            out.append({"name": p.name, "kind": "neutral", "bbox": list(p.bbox),
                        "reflectance": p.reflectance})
        elif p.kind == "black" and p.name == "true_black":
            out.append({"name": p.name, "kind": "black", "bbox": list(p.bbox),
                        "reflectance": 0.0})
        elif p.kind == "color" and p.name in keep_color and p.rgb_linear is not None:
            xy = _xy_from_rgb709(p.rgb_linear)
            if xy is not None:
                out.append({"name": p.name, "kind": "color", "bbox": list(p.bbox),
                            "xy": xy})
        elif p.kind == "specular" and p.name == "specular_clip":
            out.append({"name": p.name, "kind": "specular", "bbox": list(p.bbox),
                        "reflectance": p.reflectance})
    return out


def measure_patch_cv10(img16_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Mean 10-bit code value over the central 60 % of bbox (matches evidence)."""
    x, y, w, h = bbox
    ix, iy = int(round(w * 0.2)), int(round(h * 0.2))
    reg = img16_rgb[y + iy:y + h - iy, x + ix:x + w - ix].astype(np.float64)
    return reg.mean(axis=(0, 1)) * (1023.0 / 65535.0)
