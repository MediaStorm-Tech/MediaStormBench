"""Prompt templates for the eval runner.

Tiers differ only in what is attached (nothing / image / image + reference digest);
the question body is identical across tiers so scores are comparable. `curve_null` and
`curve_absent` use exactly the `curve` prompt: a twin must be indistinguishable from a
normal item (for `curve_absent` the only difference is the shorter option list).
Nothing here may depend on file names, labels or anything else that would leak the answer.

PROMPT_VERSION is part of request_hash; any change to the text below makes results incomparable with the
paper's. The reference digests are versioned separately through their sha1 in the hash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from cameralogbench.taxonomy import option_label

PROMPT_VERSION = "v5"  # the version used for every result in the paper

TASKS = ("curve", "camera", "curve_null", "curve_absent")
# curve_null (a content-free crop of the frame) is an opt-in control and not part of the released data;
# the whole-frame curve_absent is the abstention test.
DEFAULT_TASKS = ("curve", "camera", "curve_absent")
TWIN_TASKS = ("curve_null", "curve_absent")  # gold = undetermined, twin_of = the clip's curve item
TIERS = ("text_only", "zero_shot", "with_spec", "with_reference", "tools")
IMAGE_TIERS = ("zero_shot", "with_spec", "with_reference")
REFERENCE_TIERS = ("with_spec", "with_reference")  # with_spec: digest*_spec.md; with_reference: digest*.md
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM_PROMPT = (
    "You are a senior DIT and colorist with deep, hands-on knowledge of camera log encodings, "
    "their transfer functions and native colour gamuts. When images are attached, each one is an 8-bit "
    "screenshot of a video frame exactly as the file stores it: we applied no LUT, colour-space "
    "transform, grade or display transform of our own, so pixel values are the file's code values "
    "scaled down to 8 bits (multiply by about 4 to get 10-bit code values on a 0-1023 scale). "
    "The file itself may be a camera-original log recording, or it may already be display-referred "
    "(e.g. Rec.709 / a baked-in look); deciding which is part of the task. "
    "Answer strictly with a single JSON object and nothing else."
)

REFERENCE_PLACEHOLDER = "(The reference digest is not available in this run; rely on your own knowledge.)"

_CLAUSE = {
    "text_only": "No image is provided in this variant; answer from the question text alone.",
    "zero_shot": "Look at the attached frame.",
    "with_spec": "Look at the attached frame.",
    "with_reference": "Look at the attached frame.",
}

_REFERENCE_INTRO = (
    "Reference material: below is a digest of the camera manufacturers' white papers "
    "(encoding formulas and key code values). Use its numeric cues to check your reading of the frame."
)

_EVIDENCE_GUIDE = (
    'List the cues you relied on in "evidence". Be quantitative wherever you can: estimated '
    "black level / darkest code value, where a neutral mid-grey or skin tone sits, highlight clip or "
    "plateau code value, hue bias of strongly saturated colours. Give code values on the 10-bit scale "
    "(0-1023) or normalised 0-1."
)

_PICK_GUIDE = (
    'Pick your two most likely options ("top2", most likely first) and your single final answer '
    '("choice", normally the first entry of "top2"). Use option letters only.'
)

_QUESTION = {
    "curve": (
        "Question: Which camera log encoding (transfer function) and native colour gamut was this "
        "footage most likely recorded in?"
    ),
    "camera": (
        "Question: Which camera brand most likely recorded this footage? Also name up to three specific "
        'camera models, most likely first, in "camera_candidates".'
    ),
}

_OPEN_GUIDE = {
    "curve": 'Before reading the options, write your own free-text best guess of the encoding and gamut in "open_answer".',
    "camera": 'Before reading the options, write your own free-text best guess of the camera (brand and model) in "open_answer".',
}

_SCHEMA = {
    "curve": (
        '{"open_answer": "<free-text best guess, made before reading the options>",\n'
        ' "evidence": ["<observable cue 1>", "<cue 2>", "..."],\n'
        ' "top2": ["<letter>", "<letter>"],\n'
        ' "choice": "<single letter>"}'
    ),
    "camera": (
        '{"open_answer": "<free-text best guess, made before reading the options>",\n'
        ' "evidence": ["<observable cue 1>", "<cue 2>", "..."],\n'
        ' "camera_candidates": ["<camera model>", "..."],\n'
        ' "top2": ["<letter>", "<letter>"],\n'
        ' "choice": "<single letter>"}'
    ),
}

_TOOLS_MSG = (
    "tier 'tools' is reserved but not implemented: it needs a function-calling loop that exposes measurement "
    "tools over the frame the model sees. Run text_only / zero_shot / with_spec / with_reference instead."
)


def prompt_task(task: str) -> str:
    """Prompt family for a task; the twin tasks (curve_null / curve_absent) deliberately map to curve."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return "camera" if task == "camera" else "curve"


def load_reference(path: Path) -> tuple[str, bool]:
    """Return (reference text, reference_used). Missing/empty file -> placeholder, False."""
    p = Path(path)
    if p.is_file():
        text = p.read_text(encoding="utf-8").strip()
        if text:
            return text, True
    return REFERENCE_PLACEHOLDER, False


def render_options(option_ids: list[str]) -> str:
    if len(option_ids) > len(LETTERS):
        raise ValueError(f"too many options ({len(option_ids)})")
    return "\n".join(f"{LETTERS[i]}. {option_label(o)}" for i, o in enumerate(option_ids))


_POS_WORDS = {"p25": "25%", "p50": "50%", "p75": "75%"}


def image_clause(tier: str, frame_positions: Optional[list[str]] = None) -> str:
    """Clause naming what is attached. One frame keeps the single-frame wording; several frames say how many and
    where in the clip they come from (positions only - never anything about content or labels)."""
    if tier not in IMAGE_TIERS or not frame_positions or len(frame_positions) == 1:
        return _CLAUSE[tier]
    where = ", ".join(_POS_WORDS.get(p, p) for p in frame_positions)
    return (f"{len(frame_positions)} frames from the same clip are attached, in time order, taken at {where} "
            "of its duration. They share one encoding; use all of them.")


def build_user_text(task: str, tier: str, option_ids: list[str], reference_text: Optional[str] = None,
                    frame_positions: Optional[list[str]] = None) -> str:
    if tier == "tools":
        raise NotImplementedError(_TOOLS_MSG)
    if tier not in _CLAUSE:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    fam = prompt_task(task)
    parts = [image_clause(tier, frame_positions)]
    if tier in REFERENCE_TIERS:
        parts.append(f"{_REFERENCE_INTRO}\n<reference>\n{reference_text or REFERENCE_PLACEHOLDER}\n</reference>")
    parts += [
        _QUESTION[fam],
        "Work in this order:\n"
        f"1. {_OPEN_GUIDE[fam]}\n"
        f"2. {_EVIDENCE_GUIDE}\n"
        f"3. {_PICK_GUIDE}",
        f"Options:\n{render_options(option_ids)}",
        f"Respond with a single JSON object and nothing else:\n{_SCHEMA[fam]}",
    ]
    return "\n\n".join(parts)


def build_messages(task: str, tier: str, option_ids: list[str], image_b64: Optional[str] | list[str] = None,
                   image_mime: str = "image/jpeg", reference_text: Optional[str] = None,
                   frame_positions: Optional[list[str]] = None) -> list[dict]:
    """OpenAI-compatible chat messages for one (task, tier, rotated options) request. `image_b64` may be a
    list (multi-frame input): images come first, in the given order, then the text."""
    images = [image_b64] if isinstance(image_b64, str) else list(image_b64 or [])
    text = build_user_text(task, tier, option_ids, reference_text,
                           frame_positions if len(images) > 1 else None)
    if tier in IMAGE_TIERS:
        if not images:
            raise ValueError(f"tier {tier!r} needs an image")
        content = [{"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b}", "detail": "high"}}
                   for b in images]
        content.append({"type": "text", "text": text})
    else:
        if images:
            raise ValueError("text_only must not carry an image")
        content = [{"type": "text", "text": text}]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
