"""Canonical taxonomy of CameraLogBench: curve, gamut and brand ids and the answer-option lists.

Every label stored on disk (annotations, results) uses the ids defined here, never free text. Display
strings are rendered from the ids at prompt time. Every curve option list ends with the two abstention-type
answers (not_log, undetermined).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# Gamuts
# --------------------------------------------------------------------------- #
GAMUTS: dict[str, str] = {
    "arri_awg3": "ALEXA Wide Gamut (AWG3)",
    "arri_awg4": "ALEXA Wide Gamut 4 (AWG4)",
    "sony_sgamut3cine": "S-Gamut3.Cine",
    "sony_sgamut3": "S-Gamut3",
    "dji_dgamut": "D-Gamut",
    "red_rwg": "REDWideGamutRGB",
    "canon_cinemagamut": "Cinema Gamut",
    "bmd_wg_gen5": "Blackmagic Wide Gamut Gen 5",
    "bmd_wg_gen4": "Blackmagic Wide Gamut Gen 4",
    "panasonic_vgamut": "V-Gamut",
    "fuji_fgamut": "F-Gamut (Rec.2020 primaries)",
    "kinefinity_kwg": "Kinefinity Wide Gamut",
    "rec2020": "Rec.2020",
    # gamut variants of an existing curve: F-Log2 C and Apple Log 2 keep the F-Log2 / Apple Log curve
    "fuji_fgamut_c": "F-Gamut C",
    "apple_wide_gamut": "Apple Wide Gamut",
    "rec709": "Rec.709 / sRGB",
}

# --------------------------------------------------------------------------- #
# Curves (transfer functions). `status` = v1_main | v1_second (benchmark curves), v1_ext (extension
# curves, only in the 15- and 17-option sets), dropped, negative | abstain
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Curve:
    id: str
    name: str
    brand: Optional[str]
    gamut_id: Optional[str]
    status: str
    typical_cameras: tuple[str, ...] = ()
    notes: str = ""

    @property
    def gamut_name(self) -> Optional[str]:
        return GAMUTS.get(self.gamut_id) if self.gamut_id else None

    def option_label(self) -> str:
        """Human-readable option text used in prompts (encoding / native gamut)."""
        if self.status in ("negative", "abstain"):
            return self.name
        if self.gamut_id:
            return f"{self.name} / {self.gamut_name}"
        return self.name


CURVES: dict[str, Curve] = {c.id: c for c in [
    Curve("arri_logc3", "ARRI LogC3", "arri", "arri_awg3", "v1_main",
          ("ARRI ALEXA Classic", "ALEXA XT", "ALEXA SXT", "ALEXA Mini", "ALEXA Mini LF", "ALEXA LF", "ALEXA 65", "AMIRA"),
          "Curve parameters depend on EI (EI160-1600 table); 18% grey fixed at 0.391."),
    Curve("arri_logc4", "ARRI LogC4", "arri", "arri_awg4", "v1_main",
          ("ARRI ALEXA 35", "ALEXA 265"),
          "Single curve independent of EI."),
    Curve("sony_slog3", "Sony S-Log3", "sony", "sony_sgamut3cine", "v1_main",
          ("Sony VENICE", "VENICE 2", "BURANO", "FX9", "FX6", "FX3", "FX30", "a7S III", "a7 IV", "a1"),
          "Native gamut may be S-Gamut3.Cine or S-Gamut3; curve independent of EI."),
    Curve("dji_dlog", "DJI D-Log", "dji", "dji_dgamut", "v1_main",
          ("DJI Inspire 3", "Ronin 4D", "Zenmuse X5S/X7", "Mavic 3 series", "Inspire 2"),
          "Per DJI X9 whitepaper (2022): one scene-linear curve for all EI; only the sensor clip "
          "code value changes with EI (EI200 789, EI320 842, EI400 868, native 948)."),
    Curve("dji_dlogm", "DJI D-Log M", "dji", None, "v1_main",
          ("DJI Mavic 3 series", "Air 3", "Mini 4 Pro", "Osmo Pocket 3", "Osmo Action 4/5"),
          "Consumer-oriented flatter-than-709 log; no published gamut."),
    Curve("red_log3g10", "RED Log3G10", "red", "red_rwg", "v1_main",
          ("RED KOMODO", "V-RAPTOR", "DSMC2", "Nikon ZR (RED colour science)"),
          "Also used by Nikon ZR."),
    Curve("canon_clog3", "Canon C-Log3", "canon", "canon_cinemagamut", "v1_main",
          ("Canon C70", "C300 Mk III", "C500 Mk II", "R5 C", "R5", "R6 Mk II")),
    # Extension curves (status "v1_ext") are not in CURVE_IDS, so the 14-option set, its digest, the formula
    # baseline candidates and the synthetic natural-scene set stay unchanged; the 15- and 17-option sets add
    # them.
    Curve("canon_clog2", "Canon C-Log2", "canon", "canon_cinemagamut", "v1_ext",
          ("Canon C300 Mk II", "C300 Mk III", "C500 Mk II", "C70", "C700"),
          "Not in the synthetic natural-scene set."),
    # Same transfer function as F-Log2 / Apple Log with a different gamut; each has an official white paper.
    Curve("fuji_flog2c", "Fujifilm F-Log2 C", "fujifilm", "fuji_fgamut_c", "v1_ext",
          ("Fujifilm GFX ETERNA 55", "GFX100 II", "X-H2S"),
          "F-Log2 C Data Sheet Ver.1.0: gamma identical to F-Log2, gamut F-Gamut C."),
    Curve("apple_log2", "Apple Log 2", "apple", "apple_wide_gamut", "v1_ext",
          ("iPhone 17 Pro", "iPhone 18 Pro"),
          "Apple Log 2 White Paper (Sept 2025): Apple Log transfer function, Apple Wide Gamut."),
    Curve("bmd_film_gen5", "Blackmagic Film Gen 5", "bmd", "bmd_wg_gen5", "v1_main",
          ("Blackmagic PYXIS 6K", "Pocket 6K Pro", "URSA Mini Pro 12K", "URSA Cine 12K"),
          "Gen 5 colour science (2021+). Older bodies/firmware may record Gen 4."),
    Curve("apple_log", "Apple Log", "apple", "rec2020", "v1_main",
          ("iPhone 15 Pro", "iPhone 16 Pro", "iPhone 17 Pro")),
    # not part of the benchmark (no chart or real footage); the formula is kept in curves.py
    Curve("panasonic_vlog", "Panasonic V-Log", "panasonic", "panasonic_vgamut", "dropped",
          ("Panasonic S1H", "S5 II", "GH6", "GH7", "VariCam")),
    Curve("fuji_flog2", "Fujifilm F-Log2", "fujifilm", "fuji_fgamut", "v1_second",
          ("Fujifilm X-H2S", "X-H2", "X-T5", "GFX100 II")),
    Curve("nikon_nlog", "Nikon N-Log", "nikon", "rec2020", "v1_second",
          ("Nikon Z6 III", "Z8", "Z9", "Nikon ZR")),
    Curve("kine_log3", "Kinefinity KineLOG3", "kinefinity", "kinefinity_kwg", "v1_second",
          ("Kinefinity MAVO Edge 6K", "MAVO Edge 8K", "MAVO mark2 LF", "VISTA"),
          "Kinefinity KineLog3 Technical Specifications (2025-09-24)."),
    # abstention-type answers that every curve option set includes
    Curve("not_log", "Not a Log encoding (already display-referred, e.g. Rec.709 / sRGB)", None, "rec709", "negative"),
    Curve("undetermined", "Cannot be determined from this image", None, None, "abstain"),
]}

NOT_LOG = "not_log"
UNDETERMINED = "undetermined"
ABSTAIN_IDS = (NOT_LOG, UNDETERMINED)

# 14-option set for the curve task (12 encodings + not_log + undetermined). The order here is the canonical
# "rotation 0" order; the runner rotates this list (CircularEval).
OPTION_SET_CURVE: list[str] = [
    "arri_logc3", "arri_logc4", "sony_slog3", "dji_dlog", "dji_dlogm", "red_log3g10",
    "canon_clog3", "bmd_film_gen5", "apple_log", "fuji_flog2", "nikon_nlog",
    "kine_log3", NOT_LOG, UNDETERMINED,
]

# The 15-option set adds Canon Log 2.
EXT_CURVE_IDS: list[str] = ["canon_clog2"]
# C-Log2 sits right before C-Log3.
OPTION_SET_CURVE_EXT: list[str] = [o for c in OPTION_SET_CURVE
                                   for o in (["canon_clog2", c] if c == "canon_clog3" else [c])]
# 17 options = the 15-option set + Apple Log 2 right after Apple Log and F-Log2 C right after F-Log2.
# This is the option list of every released curve question ("ext2").
EXT2_CURVE_IDS: list[str] = ["canon_clog2", "apple_log2", "fuji_flog2c"]
OPTION_SET_CURVE_EXT2: list[str] = [o for c in OPTION_SET_CURVE_EXT
                                    for o in ([c, "apple_log2"] if c == "apple_log" else
                                              [c, "fuji_flog2c"] if c == "fuji_flog2" else [c])]
OPTION_SETS_CURVE: dict[str, list[str]] = {"v5": OPTION_SET_CURVE, "ext": OPTION_SET_CURVE_EXT,
                                           "ext2": OPTION_SET_CURVE_EXT2}

# --------------------------------------------------------------------------- #
# Brands (camera task)
# --------------------------------------------------------------------------- #
BRANDS: dict[str, str] = {
    "arri": "ARRI", "sony": "Sony", "dji": "DJI", "red": "RED", "canon": "Canon",
    "bmd": "Blackmagic Design", "apple": "Apple",
    "fujifilm": "Fujifilm", "nikon": "Nikon", "kinefinity": "Kinefinity",
}
OPTION_SET_BRAND: list[str] = list(BRANDS) + [UNDETERMINED]


def option_label(opt_id: str) -> str:
    if opt_id in CURVES:
        return CURVES[opt_id].option_label()
    if opt_id in BRANDS:
        return BRANDS[opt_id]
    raise KeyError(opt_id)


# --------------------------------------------------------------------------- #
# Normalisation of free-text labels (e.g. a labels.csv column)
# --------------------------------------------------------------------------- #
_LABEL_ALIASES: dict[str, str] = {
    # key = lowercase, stripped of spaces / hyphens / underscores / dots
    "log3g10": "red_log3g10", "redlog3g10": "red_log3g10", "redlog3g10rwg": "red_log3g10",
    "blackmagicdesignfilm": "bmd_film_gen5", "blackmagicfilm": "bmd_film_gen5",
    "blackmagicfilmgen5": "bmd_film_gen5", "bmdfilmgen5": "bmd_film_gen5", "bmdfilm": "bmd_film_gen5",
    "dlog": "dji_dlog", "djidlog": "dji_dlog",
    "dlogm": "dji_dlogm", "djidlogm": "dji_dlogm",
    "slog3": "sony_slog3", "sonyslog3": "sony_slog3", "slog3sgamut3cine": "sony_slog3",
    "logc3": "arri_logc3", "logc": "arri_logc3", "arrilogc3": "arri_logc3", "alexalogc": "arri_logc3",
    "logc4": "arri_logc4", "arrilogc4": "arri_logc4",
    "clog3": "canon_clog3", "canonlog3": "canon_clog3", "canonclog3": "canon_clog3",
    "clog2": "canon_clog2", "canonlog2": "canon_clog2", "canonclog2": "canon_clog2",
    "applelog": "apple_log",
    "vlog": "panasonic_vlog", "panasonicvlog": "panasonic_vlog",
    "flog2": "fuji_flog2", "fujifilmflog2": "fuji_flog2",
    "nlog": "nikon_nlog", "nikonnlog": "nikon_nlog",
    "kinelog3": "kine_log3", "kinefinitykinelog3": "kine_log3",
    "rec709": NOT_LOG, "709": NOT_LOG, "srgb": NOT_LOG, "notlog": NOT_LOG,
}


def normalize_curve_label(text: str) -> Optional[str]:
    """Map a free-text label (e.g. CSV column 'Log') to a curve id, or None."""
    if text.strip() in CURVES:  # taxonomy ids pass through
        return text.strip()
    key = re.sub(r"[\s\-_./()]+", "", text.strip().lower())
    return _LABEL_ALIASES.get(key)


if __name__ == "__main__":  # quick self-check
    for cid in OPTION_SET_CURVE_EXT2:
        print(f"{cid:16s} -> {option_label(cid)}")
    for s in ("Log3G10", "Blackmagic Design Film", "D-Log", "D-Log M", "S-Log3", "LogC3"):
        print(f"{s!r:28s} -> {normalize_curve_label(s)}")
