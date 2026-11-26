"""Eval runner: annotations/<subset>.jsonl -> one results .jsonl (one line per model call).

One output line per model call = (model, tier, item, rotation[, frame]). CircularEval: rotation r shifts
the item's rotation-0 option list right by r * len(options) // k. The model answers letters; the
letter -> id mapping is the rotated list stored in the record.

Resume: a job is skipped when the output file already holds a record with the same request_hash and
error == null, including parse failures (scored as wrong, "Z"); --retry-parse-failures re-asks records
without a valid choice, --force reruns everything. The file is append-only; the scorer keeps the last
record per (model, tier, frames mode, item, option layout).

Usage (the settings of the paper):
  python -m cameralogbench.eval.runner \\
      --annotations data/CameraLogBench/annotations/real.jsonl --data-root data/CameraLogBench \\
      --models gpt-6.1-sol --tiers zero_shot with_reference --frames k3 \\
      --rotations 2 --effort high --max-tokens 32000 --out runs/real__gpt-6.1-sol.jsonl [--dry-run]

Credentials come from environment variables:
  OPENAI_BASE_URL, OPENAI_API_KEY        any OpenAI-compatible /chat/completions endpoint (base URL
                                         including /v1, default https://api.openai.com/v1)
  ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY  claude-* models when --effort is set: native Messages API with
                                         adaptive thinking (default base https://api.anthropic.com)

Image paths in the annotations are relative to --data-root. Default tasks are curve, camera and
curve_absent. The with_spec / with_reference digest is picked per clip from its curve option list: the
released questions have 17 options and use digest_ext2; smaller option lists built with
cameralogbench.frames --option-set use digest / digest_ext. --digest or --reference / --spec-reference
override it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from cameralogbench import taxonomy
from cameralogbench.eval import prompts

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
# option set -> (with_reference digest, with_spec digest)
DIGESTS = {
    "v5": ("digest.md", "digest_spec.md"),
    "ext": ("digest_ext.md", "digest_ext_spec.md"),
    "ext2": ("digest_ext2.md", "digest_ext2_spec.md"),
}
RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
# The model input is the 1920-wide 8-bit frame, stored as lossless WebP
IMAGE_KEY, FRAMES_KEY, IMAGE_VARIANT = "image_webp", "frames_webp", "webp"
FRAME_MODES = ("single", "k3", "each")
# single: the middle frame (input.image_webp)
# k3:     all frames of the clip in one request (input.frames_webp)
# each:   one request per frame (K requests per item): answer stability across frames of one clip
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com"


# --------------------------------------------------------------------------- #
# Hashing / rotation
# --------------------------------------------------------------------------- #
def rotate_options(options: list[str], rotation: int, k: int) -> list[str]:
    shift = (rotation * len(options) // k) % len(options) if options else 0
    return options[-shift:] + options[:-shift] if shift else list(options)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def file_sha1(path: str) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def request_hash(model: str, tier: str, task: str, item_id: str, rotation: int, image_sha1: str,
                 reference_sha1: str, options: list[str], dry_run: bool,
                 params: Optional[dict] = None, frames_mode: str = "single") -> str:
    """Identity of one request. Includes the effective generation parameters (token budget) so that
    records produced under different budgets never satisfy resume for each other."""
    parts: list[Any] = [model, tier, task, item_id, rotation, image_sha1, prompts.PROMPT_VERSION,
                        reference_sha1, options]
    if params:
        parts.append(sorted(params.items()))  # keys are unique, so nested values are never compared
    if frames_mode != "single":  # single-frame requests keep the original hash layout
        parts.append(["frames", frames_mode])
    if dry_run:  # keep fake answers from ever satisfying resume for a real run
        parts.append("dry_run")
    return sha1_text(json.dumps(parts, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def credentials(route: str) -> tuple[str, str]:
    """(base URL, API key) for "openai" or "anthropic" from the environment."""
    if route == ANTHROPIC_ROUTE:
        base, key = os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE, os.environ.get("ANTHROPIC_API_KEY")
        names = "ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL)"
    else:
        base, key = os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE, os.environ.get("OPENAI_API_KEY")
        names = "OPENAI_API_KEY (and optionally OPENAI_BASE_URL)"
    if not key:
        raise SystemExit(f"error: set {names} in the environment")
    return base.strip().rstrip("/"), key.strip()


def _content_text(content: Any) -> Optional[str]:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):  # some servers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict)) or None
    return str(content)


GEMINI_PRO = re.compile(r"^gemini-.*\bpro\b")
OPENAI_REASONING = re.compile(r"^o\d")
GEMINI_PRO_MIN_TOKENS = 8192
CLAUDE = re.compile(r"^claude-")
ANTHROPIC_ROUTE = "anthropic_messages"


def request_params(model: str, max_tokens: int, effort: Optional[str] = None) -> dict[str, Any]:
    """Model-specific generation parameters actually sent (and recorded with every result). Keys starting
    with "_" select the route and are not sent.

    effort: one reasoning level for every model. Claude goes through the native Messages API with adaptive
    thinking + output_config.effort; all other models get `reasoning_effort` on the OpenAI-compatible
    route. Claude takes no sampling parameters, so no temperature there."""
    if effort and CLAUDE.match(model):
        return {"_route": ANTHROPIC_ROUTE, "max_tokens": max_tokens, "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort}}
    # gemini-*-pro thinking tokens often exhaust a 4k budget before the JSON answer
    if GEMINI_PRO.match(model) and max_tokens < GEMINI_PRO_MIN_TOKENS:
        max_tokens = GEMINI_PRO_MIN_TOKENS
    openai_style = model.startswith("gpt-") or bool(OPENAI_REASONING.match(model))
    # gpt-6.1 / gpt-5.6 / o-series reject max_tokens; gpt-6 still accepts it
    if OPENAI_REASONING.match(model) or (model.startswith("gpt-") and any(t in model for t in ("6.1", "5.6"))):
        params: dict[str, Any] = {"max_completion_tokens": max_tokens}
    else:
        params = {"max_tokens": max_tokens}
    if model.startswith("kimi-"):  # Kimi K3 / K3.1 accept only temperature 1
        params["temperature"] = 1
    elif not openai_style:  # gpt-* / o-series reject any non-default temperature
        params["temperature"] = 0
    if effort:
        params["reasoning_effort"] = effort
    return params


def route_of(params: dict) -> str:
    return params.get("_route") or "openai"


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-style [system, user] messages -> (system, Messages-API messages); images become base64 blocks."""
    system = messages[0]["content"]
    content: list[dict] = []
    for part in messages[1]["content"]:
        if part["type"] == "image_url":
            mime, b64 = part["image_url"]["url"][len("data:"):].split(";base64,", 1)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        else:
            content.append({"type": "text", "text": part["text"]})
    return system, [{"role": "user", "content": content}]


def _read_sse_chat(r) -> dict:
    """Collect an OpenAI-style streamed chat completion into the non-streamed response shape.
    Reasoning deltas are dropped (only `content` is the answer); usage comes with the last chunk."""
    text, finish, usage, seen = [], None, {}, False
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ch = json.loads(data)
        except json.JSONDecodeError:
            continue
        if ch.get("error"):
            return {"error": ch["error"]}
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices") or []:
            seen = True
            piece = _content_text((c.get("delta") or {}).get("content"))
            if piece:
                text.append(piece)
            finish = c.get("finish_reason") or finish
    if not seen:
        return {"choices": [], "usage": usage}
    return {"choices": [{"message": {"content": "".join(text) or None}, "finish_reason": finish}], "usage": usage}


def call_api(model: str, messages: list[dict], base: str, key: str, params: dict[str, Any],
             timeout: float, retries: int, stream: bool = False) -> dict:
    """One model call with retries on 429 / 5xx / network errors. stream=True (OpenAI-compatible route
    only) receives the answer as server-sent events: same request, same model output, but bytes keep
    flowing, so gateways that cut long idle connections do not drop slow reasoning models. Streaming is
    not part of request_hash."""
    sent = {k: v for k, v in params.items() if not k.startswith("_")}
    native = params.get("_route") == ANTHROPIC_ROUTE
    stream = stream and not native
    if stream:
        sent = dict(sent, stream=True, stream_options={"include_usage": True})
    if native:
        system, msgs = to_anthropic(messages)
        body = json.dumps(dict(model=model, system=system, messages=msgs, **sent)).encode()
        url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    else:
        body = json.dumps(dict(model=model, messages=messages, **sent)).encode()
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    err = None
    attempt = 0
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers)
        retriable = True
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = _read_sse_chat(r) if stream else json.load(r)
            if stream and d.get("error"):
                err = f"stream error: {json.dumps(d['error'])[:300]}"
                time.sleep(1)
                continue
            if native:
                if d.get("type") == "message":
                    blocks = d.get("content") or []
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None
                    u = dict(d.get("usage") or {})
                    u["prompt_tokens"] = u.get("input_tokens")
                    u["completion_tokens"] = u.get("output_tokens")
                    u["thinking_blocks"] = sum(b.get("type") == "thinking" for b in blocks)
                    # a refusal is model behaviour (scored as wrong), not a transport error
                    return dict(raw=text, usage=u, error=None, finish_reason=d.get("stop_reason"),
                                attempts=attempt)
                err = f"unexpected /v1/messages response: {json.dumps(d)[:300]}"
                time.sleep(1)
                continue
            choices = d.get("choices") or []
            if choices:
                return dict(raw=_content_text(choices[0].get("message", {}).get("content")),
                            usage=d.get("usage") or {}, error=None,
                            finish_reason=choices[0].get("finish_reason"), attempts=attempt,
                            transport="stream" if stream else None)
            err = f"no choices in response: {json.dumps(d)[:300]}"
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300]
            except Exception as read_err:  # noqa: BLE001 - body cut off by a gateway timeout
                detail = repr(read_err).encode()
            err = f"HTTP {e.code}: {detail!r}"
            retriable = e.code in RETRY_HTTP
        except Exception as e:  # noqa: BLE001 - network / timeout / bad JSON
            err = repr(e)
        if not retriable or attempt == retries:
            break
        time.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)) + random.uniform(0, 1))
    return dict(raw=None, usage={}, error=err, finish_reason=None, attempts=attempt)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _loads_lenient(s: str) -> Optional[dict]:
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s.replace("\u201c", '"').replace("\u201d", '"'))):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            pass
    return None


def parse_response(raw: Optional[str]) -> tuple[Optional[dict], str]:
    """-> (parsed, status). status: json | salvaged | no_json. Never invents an answer."""
    if not raw or not raw.strip():
        return None, "no_json"
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        v = _loads_lenient(text[s:e + 1])
        if v is not None:
            return v, "json"
    # several objects / chatter around them: take the last decodable object that has "choice"
    dec = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                v, _ = dec.raw_decode(text[i:])
                if isinstance(v, dict) and "choice" in v:
                    found = v
            except Exception:  # noqa: BLE001
                pass
    if found is not None:
        return found, "json"
    # truncated JSON: salvage only explicitly stated fields; no "choice" -> parse failure (Z)
    mc = re.search(r'"choice"\s*:\s*"([^"]{1,40})"', raw)
    if not mc:
        return None, "no_json"
    out: dict[str, Any] = {"choice": mc.group(1), "_salvaged": True}
    mt = re.search(r'"top2"\s*:\s*\[([^\]]*)\]', raw)
    if mt:
        out["top2"] = re.findall(r'"([^"]{1,40})"', mt.group(1))
    mf = re.search(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)', raw)
    if mf:
        out["confidence"] = float(mf.group(1))
    mo = re.search(r'"open_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if mo:
        out["open_answer"] = mo.group(1)
    return out, "salvaged"


_LETTER = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\]\.:]?(?:\s|$)")


def letter_to_id(value: Any, options: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Map an answer to (letter, option id). Accepts 'C', 'c', '(C)', 'C. Sony ...', or the exact
    option text / id. Anything else -> (None, None)."""
    if value is None:
        return None, None
    s = str(value).strip()
    m = _LETTER.match(s)
    if m:
        idx = prompts.LETTERS.find(m.group(1).upper())
        if 0 <= idx < len(options):
            return prompts.LETTERS[idx], options[idx]
        return None, None
    low = s.lower()
    for i, o in enumerate(options):
        if low in (o.lower(), taxonomy.option_label(o).lower()):
            return prompts.LETTERS[i], o
    return None, None


def parse_confidence(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        m = re.search(r"\d+(?:\.\d+)?", str(v or ""))
        if not m:
            return None
        x = float(m.group(0))
    return max(0.0, min(100.0, x))


def gold_accept(task: str, gold: dict, options: list[str]) -> list[str]:
    """Option ids that count as correct for this item."""
    if task == "camera":
        cands = [gold.get("brand")] + list(gold.get("accept") or []) + [gold.get("primary")]
    else:
        cands = list(gold.get("accept") or []) + [gold.get("primary")]
    out: list[str] = []
    for c in cands:
        if c and c in options and c not in out:
            out.append(c)
    return out


def interpret(parsed: Optional[dict], task: str, options: list[str], accept: list[str]) -> dict:
    res: dict[str, Any] = dict(choice_letter=None, choice_id=None, top2_ids=None, confidence=None,
                               camera_candidates_pred=None, correct=None, top2_hit=None)
    if task == "camera":
        res["camera_candidates_pred"] = []
    if parsed is None:
        return res
    letter, cid = letter_to_id(parsed.get("choice"), options)
    res["choice_letter"], res["choice_id"] = letter, cid
    top2: list[str] = []
    raw_top2 = parsed.get("top2")
    for x in (raw_top2 if isinstance(raw_top2, list) else [raw_top2] if raw_top2 else []):
        _, tid = letter_to_id(x, options)
        if tid and tid not in top2:
            top2.append(tid)
    top2 = top2[:2]
    if cid and not top2:
        top2 = [cid]
    res["top2_ids"] = top2
    res["confidence"] = parse_confidence(parsed.get("confidence"))
    if task == "camera":
        cc = parsed.get("camera_candidates") or []
        if isinstance(cc, str):
            cc = [cc]
        res["camera_candidates_pred"] = [str(c).strip() for c in cc if str(c).strip()][:3]
    if cid:
        res["correct"] = cid in accept
        res["top2_hit"] = any(t in accept for t in top2)
    return res


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
_FAKE_EVIDENCE = [
    "Darkest pixels sit around 95/1023, so the black level is lifted well above zero",
    "A neutral grey wall reads about 0.41 normalised",
    "Highlights plateau near 940 (10-bit), suggesting a clip point below 1023",
    "Overall low contrast with lifted shadows",
    "Saturated reds lean slightly towards orange",
    "Soft highlight roll-off",
    "Skin tones sit around 480-520 on the 10-bit scale",
]
_FAKE_FAILS = [
    'I think this is probably a log image but I cannot format JSON right now.',
    '{"open_answer": "Looks like a flat log image", "evidence": ["Lifted blacks", "Soft highlight',
    "",
]


def fake_response(item: dict, options: list[str], accept: list[str], user_text: str, task: str,
                  seed: str, fail_rate: float, has_image: bool) -> tuple[str, dict]:
    rng = random.Random(seed)
    oracle = bool((item.get("meta") or {}).get("dry_run_oracle"))
    if not oracle and rng.random() < fail_rate:
        raw = rng.choice(_FAKE_FAILS)
    else:
        n = len(options)
        if oracle:
            # read the letter off the rendered prompt, so rendering and letter->id mapping are
            # checked against each other rather than against the runner's own option list
            m = re.search(rf"^([A-Z])\. {re.escape(taxonomy.option_label(accept[0]))}$", user_text, re.M)
            choice = m.group(1) if m else "?"
            second = rng.choice([c for c in prompts.LETTERS[:n] if c != choice])
        else:
            choice = rng.choice(prompts.LETTERS[:n])
            second = rng.choice([c for c in prompts.LETTERS[:n] if c != choice])
        ans: dict[str, Any] = {
            "open_answer": "dry-run answer",
            "evidence": rng.sample(_FAKE_EVIDENCE, k=rng.randint(1, 3)),
            "top2": [choice, second], "choice": choice,
        }
        if task == "camera":
            g = item.get("gold") or {}
            gold_cams = list(g.get("camera_candidates") or ([g["camera_model"]] if g.get("camera_model") else []))
            pool = gold_cams if oracle and gold_cams else [c for cv in taxonomy.CURVES.values()
                                                          for c in cv.typical_cameras]
            ans["camera_candidates"] = rng.sample(pool, k=min(len(pool), rng.randint(1, 3)))
        raw = json.dumps(ans, ensure_ascii=False)
        if rng.random() < 0.3:
            raw = f"```json\n{raw}\n```"
    pt = len(user_text) // 4 + len(prompts.SYSTEM_PROMPT) // 4 + (1600 if has_image else 0)
    ct = max(1, len(raw) // 4)
    return raw, {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    model: str
    tier: str
    item: dict
    rotation: int
    k: int
    options: list[str]
    accept: list[str]
    image_paths: Optional[list[Path]]   # absolute paths of the attached frames, in order
    image_sha1: str                     # sha1 of the image, or of the joined per-frame sha1s
    reference_text: Optional[str]
    reference_sha1: str
    reference_used: Optional[bool]
    request_hash: str
    frames_mode: str = "single"
    frame_pos: Optional[str] = None     # "each": which frame this request carries
    frame_positions: Optional[list[str]] = None  # positions of the attached frames, in order
    effort: Optional[str] = None


def load_items(paths: list[Path]) -> list[dict]:
    items = []
    for path in paths:
        for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{ln} bad JSON skipped: {e}", file=sys.stderr)
    return items


def select_items(items: list[dict], tasks: list[str], limit: int, item_ids: Optional[list[str]]) -> list[dict]:
    sel = [it for it in items if it.get("task") in tasks]
    if item_ids:
        wanted = set(item_ids)
        sel = [it for it in sel if it["item_id"] in wanted]
    if limit:  # first N clips, all selected tasks of each, so twins stay paired with their source items
        clips: list[str] = []
        for it in sel:
            c = it.get("clip_id") or it["item_id"]
            if c not in clips:
                clips.append(c)
        keep = set(clips[:limit])
        sel = [it for it in sel if (it.get("clip_id") or it["item_id"]) in keep]
    return sel


def default_options(task: str) -> list[str]:
    return list(taxonomy.OPTION_SET_BRAND if task == "camera" else taxonomy.OPTION_SET_CURVE)


def option_set_of(curve_options: set[str]) -> str:
    """Name of the curve option set ("v5" = 14 options, "ext" = 15, "ext2" = 17) a set of ids belongs to."""
    if curve_options & (set(taxonomy.EXT2_CURVE_IDS) - set(taxonomy.EXT_CURVE_IDS)):
        return "ext2"
    if curve_options & set(taxonomy.EXT_CURVE_IDS):
        return "ext"
    return "v5"


def clip_option_sets(items: list[dict]) -> dict[str, str]:
    """clip_id -> option-set name, from the curve options of all items of that clip. Camera items (brand
    options) take the set of their clip's curve items; a clip without curve items takes the file's set."""
    by_clip: dict[str, set[str]] = defaultdict(set)
    every: set[str] = set()
    for it in items:
        if it.get("task") != "camera":
            opts = set(it.get("options") or default_options(it.get("task", "curve")))
            by_clip[it.get("clip_id") or it["item_id"]] |= opts
            every |= opts
    fallback = option_set_of(every)
    return defaultdict(lambda: fallback, {c: option_set_of(o) for c, o in by_clip.items()})


def _image_sets(it: dict, tier: str, frames_mode: str, data_root: Path,
                warnings: list[str], tag: str) -> list[tuple[list[Path], list[str], Optional[str]]]:
    """-> [(image paths, their frame positions, frame_pos for 'each')] for one item; [] = skip."""
    if tier not in prompts.IMAGE_TIERS:
        return [([], [], None)]
    inp = it.get("input") or {}
    if frames_mode == "single":
        rel = inp.get(IMAGE_KEY)
        p = data_root / rel if rel else None
        if p is None or not p.is_file():
            warnings.append(f"{it['item_id']}: {IMAGE_KEY} not found ({p}); skipped for {tag}")
            return []
        return [([p], ["p50"], None)]
    paths = [data_root / x for x in inp.get(FRAMES_KEY) or []]
    pos = list(inp.get("frame_positions") or [])
    if not paths or len(paths) != len(pos) or not all(p.is_file() for p in paths):
        warnings.append(f"{it['item_id']}: {FRAMES_KEY} missing or incomplete under {data_root}; skipped for {tag}")
        return []
    if frames_mode == "k3":
        return [(paths, pos, None)]
    return [([p], [q], q) for p, q in zip(paths, pos)]


def build_jobs(items: list[dict], models: list[str], tiers: list[str], k: int, data_root: Path,
               references: dict[str, tuple[str, bool]], spec_references: dict[str, tuple[str, bool]],
               clip_sets: dict[str, str], dry_run: bool, max_tokens: int = 4000,
               frames_mode: str = "single", effort: Optional[str] = None) -> tuple[list[Job], list[str]]:
    """references / spec_references: option-set name -> (digest text, found). clip_sets: clip -> set name."""
    jobs: list[Job] = []
    warnings: list[str] = []
    for model in models:
        for tier in tiers:
            for it in items:
                task = it["task"]
                base_opts = list(it.get("options") or default_options(task))
                for o in base_opts:
                    taxonomy.option_label(o)  # KeyError on ids unknown to the taxonomy
                use_ref = tier in prompts.REFERENCE_TIERS
                oset = clip_sets[it.get("clip_id") or it["item_id"]]
                t_text, t_used = (spec_references if tier == "with_spec" else references)[oset]
                t_sha1 = sha1_text(t_text)
                # single-image subsets (charts, synthetic frames) have no frame list: they are always "single"
                has_frames = bool((it.get("input") or {}).get(FRAMES_KEY))
                mode = frames_mode if tier in prompts.IMAGE_TIERS and has_frames else "single"
                for paths, positions, fpos in _image_sets(it, tier, mode, data_root, warnings, f"{model}/{tier}"):
                    if len(paths) == 1:
                        img_sha1 = file_sha1(str(paths[0]))
                    else:
                        img_sha1 = sha1_text("|".join(file_sha1(str(p)) for p in paths)) if paths else ""
                    for r in range(k):
                        opts = rotate_options(base_opts, r, k)
                        accept = gold_accept(task, it.get("gold") or {}, opts)
                        if not accept and r == 0:
                            warnings.append(f"{it['item_id']}: gold not among options; every answer will score wrong")
                        rsha = t_sha1 if use_ref else ""
                        h = request_hash(model, tier, task, it["item_id"], r, img_sha1, rsha, opts, dry_run,
                                         request_params(model, max_tokens, effort), mode)
                        jobs.append(Job(model, tier, it, r, k, opts, accept, paths or None, img_sha1,
                                        t_text if use_ref else None, rsha, t_used if use_ref else None, h,
                                        frames_mode=mode, frame_pos=fpos, frame_positions=positions or None,
                                        effort=effort))
    return jobs, warnings


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def make_record(job: Job, run_tag: str, resp: dict, params: dict, latency: float, data_root: Path,
                dry_run: bool = False, ts: Optional[str] = None) -> dict:
    """Parse + score one model response into a results record."""
    it, task = job.item, job.item["task"]
    if resp["error"]:
        parsed, status = None, "api_error"
    else:
        parsed, status = parse_response(resp["raw"])
    ans = interpret(parsed, task, job.options, job.accept)
    if parsed is not None and ans["choice_id"] is None:
        status = "invalid_choice"
    meta = it.get("meta") or {}
    image_paths = [_rel(p, data_root) for p in job.image_paths] if job.image_paths else None
    return {
        "run_tag": run_tag, "model": job.model, "tier": job.tier,
        "item_id": it["item_id"], "clip_id": it.get("clip_id"), "task": task,
        "rotation": job.rotation, "n_rotations": job.k, "options": job.options,
        "gold": it.get("gold"), "accept_ids": job.accept, "twin_of": it.get("twin_of"),
        "request_hash": job.request_hash,
        "raw": resp["raw"], "parsed": parsed, "error": resp["error"], "parse_status": status,
        "choice_letter": ans["choice_letter"], "choice_id": ans["choice_id"],
        "top2_ids": ans["top2_ids"], "confidence": ans["confidence"],
        "camera_candidates_pred": ans["camera_candidates_pred"],
        "correct": ans["correct"], "top2_hit": ans["top2_hit"],
        "usage": resp.get("usage") or {}, "latency_s": latency, "finish_reason": resp.get("finish_reason"),
        "attempts": resp.get("attempts"), "transport": resp.get("transport"), "request_params": params,
        "ts": ts or datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt_version": prompts.PROMPT_VERSION,
        "image_variant": IMAGE_VARIANT if job.image_paths else None,
        "image_path": image_paths[0] if image_paths else None,
        "image_sha1": job.image_sha1 or None,
        "frames_mode": job.frames_mode, "frame_pos": job.frame_pos, "frame_positions": job.frame_positions,
        "image_paths": image_paths,
        "effort": job.effort,
        "reference_used": job.reference_used, "reference_sha1": job.reference_sha1 or None,
        "dry_run": bool(dry_run),
        "item_meta": {k: meta[k] for k in ("subset", "source_class") if k in meta},
    }


def run_job(job: Job, run_tag: str, cfg: argparse.Namespace, creds: dict[str, tuple[str, str]]) -> dict:
    it, task = job.item, job.item["task"]
    params = request_params(job.model, cfg.max_tokens, job.effort)
    t0 = time.time()
    try:
        images, mime = [], "image/jpeg"
        for p in job.image_paths or []:
            images.append(base64.b64encode(p.read_bytes()).decode())
            mime = MIME.get(p.suffix.lower(), "image/jpeg")
        image_b64 = images if len(images) > 1 else (images[0] if images else None)
        messages = prompts.build_messages(task, job.tier, job.options, image_b64, mime, job.reference_text,
                                          job.frame_positions)
        user_text = messages[1]["content"][-1]["text"]
        if cfg.dry_run:
            raw, usage = fake_response(it, job.options, job.accept, user_text, task, job.request_hash,
                                       cfg.dry_run_fail_rate, image_b64 is not None)
            resp = dict(raw=raw, usage=usage, error=None, finish_reason="dry_run", attempts=1)
        else:
            base, key = creds[route_of(params)]
            resp = call_api(job.model, messages, base, key, params, cfg.timeout, cfg.retries, stream=cfg.stream)
    except Exception as e:  # noqa: BLE001 - recorded as an error so the run continues and resume retries it
        resp = dict(raw=None, usage={}, error=f"local error: {e!r}", finish_reason=None, attempts=0)
    return make_record(job, run_tag, resp, params, round(time.time() - t0, 2), cfg.data_root, cfg.dry_run)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _usage_tokens(u: dict) -> tuple[int, int, int]:
    pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    ct = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    rt = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return pt, ct, rt


def print_usage_summary(recs: list[dict]) -> None:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in recs:
        a = agg[r["model"]]
        pt, ct, rt = _usage_tokens(r.get("usage") or {})
        a["calls"] += 1
        a["err"] += bool(r.get("error"))
        a["Z"] += (not r.get("error")) and r.get("choice_id") is None
        a["prompt"] += pt
        a["completion"] += ct
        a["reasoning"] += rt
        a["latency"] += r.get("latency_s") or 0
    print(f"\n{'model':28} {'calls':>6} {'err':>4} {'Z':>4} {'prompt_tok':>11} {'compl_tok':>10} "
          f"{'(reason)':>9} {'total_tok':>10} {'avg_lat_s':>9}", file=sys.stderr)
    for m, a in sorted(agg.items()):
        print(f"{m:28} {int(a['calls']):6d} {int(a['err']):4d} {int(a['Z']):4d} {int(a['prompt']):11d} "
              f"{int(a['completion']):10d} {int(a['reasoning']):9d} {int(a['prompt'] + a['completion']):10d} "
              f"{a['latency'] / max(a['calls'], 1):9.1f}", file=sys.stderr)


def _load_digests(choice: str, reference: Optional[str], spec_reference: Optional[str]
                  ) -> tuple[dict[str, tuple[str, bool]], dict[str, tuple[str, bool]]]:
    """option-set name -> (text, found) for with_reference and with_spec. A forced digest (--digest or an
    explicit path) is used for every option set."""
    refs, specs = {}, {}
    for name in DIGESTS:
        pick = name if choice == "auto" else choice
        refs[name] = prompts.load_reference(Path(reference) if reference else REFERENCE_DIR / DIGESTS[pick][0])
        specs[name] = prompts.load_reference(Path(spec_reference) if spec_reference
                                             else REFERENCE_DIR / DIGESTS[pick][1])
    return refs, specs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True, help="annotation .jsonl file(s)")
    ap.add_argument("--data-root", required=True, help="dataset folder the image paths are relative to")
    ap.add_argument("--out", required=True, help="results .jsonl (appended to; resumable)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tiers", nargs="+", default=["text_only", "zero_shot"], choices=prompts.TIERS)
    ap.add_argument("--tasks", nargs="+", default=list(prompts.DEFAULT_TASKS), choices=prompts.TASKS)
    ap.add_argument("--rotations", type=int, default=2, help="CircularEval k (>=1)")
    ap.add_argument("--frames", default="single", choices=FRAME_MODES,
                    help="single = middle frame; k3 = all frames in one request; each = one request per frame. "
                         "Items without a frame list always use their single image")
    ap.add_argument("--effort", default=None, choices=EFFORTS, help="reasoning level sent to every model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (all selected tasks)")
    ap.add_argument("--item-ids", nargs="+", default=None)
    ap.add_argument("--run-tag", default=None, help="stored in every record; default: the --out file name")
    ap.add_argument("--digest", default="auto", choices=["auto", *DIGESTS],
                    help="digest for with_spec / with_reference; auto = by each clip's curve option set")
    ap.add_argument("--reference", default=None, help="explicit digest file for with_reference")
    ap.add_argument("--spec-reference", default=None, help="explicit digest file for with_spec")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help=f"output budget; gemini-*-pro is raised to >= {GEMINI_PRO_MIN_TOKENS}")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=5, help="attempts per call (429/5xx/network)")
    ap.add_argument("--stream", action="store_true",
                    help="receive answers as server-sent events (OpenAI-compatible route), for gateways that "
                         "cut long-running requests")
    ap.add_argument("--dry-run", action="store_true", help="no network; fake answers")
    ap.add_argument("--dry-run-fail-rate", type=float, default=0.1)
    ap.add_argument("--force", action="store_true", help="rerun even if a successful record exists")
    # A parse failure is a model behaviour we measure (scored as wrong), not a transient error, so by default
    # parse-failed records count as done.
    ap.add_argument("--retry-parse-failures", action="store_true",
                    help="re-ask items whose previous answer could not be parsed (default: keep them, scored as Z)")
    a = ap.parse_args(argv)

    if a.rotations < 1:
        ap.error("--rotations must be >= 1")
    if "tools" in a.tiers:
        try:
            prompts.build_user_text("curve", "tools", taxonomy.OPTION_SET_CURVE)
        except NotImplementedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    a.data_root = Path(a.data_root)
    for p in a.annotations:
        if not Path(p).is_file():
            print(f"error: annotations file not found: {p}", file=sys.stderr)
            return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_tag = a.run_tag or out.stem

    all_items = load_items([Path(p) for p in a.annotations])
    items = select_items(all_items, a.tasks, a.limit, a.item_ids)
    n_opts = min((len(it.get("options") or default_options(it["task"])) for it in items), default=0)
    if n_opts and a.rotations > n_opts:
        # shifts r * n // k repeat once k > n, and the scorer would merge the identical layouts
        print(f"error: --rotations {a.rotations} exceeds the smallest option list ({n_opts})", file=sys.stderr)
        return 2
    references, spec_references = _load_digests(a.digest, a.reference, a.spec_reference)
    for tier, refs in (("with_reference", references), ("with_spec", spec_references)):
        if tier in a.tiers and not all(found for _, found in refs.values()):
            print(f"[warn] a {tier} digest is missing; that tier uses a placeholder and records "
                  f"reference_used=false", file=sys.stderr)
    jobs, warnings = build_jobs(items, a.models, a.tiers, a.rotations, a.data_root, references, spec_references,
                                clip_option_sets(all_items), a.dry_run, a.max_tokens, a.frames, a.effort)
    for w in dict.fromkeys(warnings):
        print(f"[warn] {w}", file=sys.stderr)

    done: set[str] = set()
    if out.is_file() and not a.force:
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Z (no_json, invalid_choice, salvaged without a valid letter) counts as done unless retried
            if r.get("error") is None and (r.get("choice_id") is not None or not a.retry_parse_failures):
                done.add(r.get("request_hash"))
    todo = [j for j in jobs if j.request_hash not in done]
    print(f"{len(items)} items, {len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run{' (dry-run)' if a.dry_run else ''} -> {out}", file=sys.stderr)

    creds: dict[str, tuple[str, str]] = {}
    if not a.dry_run:
        for route in sorted({route_of(request_params(j.model, a.max_tokens, j.effort)) for j in todo}):
            creds[route] = credentials(route)
    recs: list[dict] = []
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        futs = [ex.submit(run_job, j, run_tag, a, creds) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            recs.append(rec)
            # one write() per record: several runners may append to the same file
            os.write(f.fileno(), (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            if n % 10 == 0 or n == len(todo) or rec.get("error"):
                tag = rec.get("error") or rec.get("choice_id") or rec["parse_status"]
                print(f"[{n}/{len(todo)}] {rec['model']} {rec['tier']} {rec['item_id']} r{rec['rotation']} "
                      f"-> {str(tag)[:120]}", file=sys.stderr)
    if recs:
        print_usage_summary(recs)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
