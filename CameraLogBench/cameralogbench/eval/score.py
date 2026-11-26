"""Scorer: results .jsonl -> report .md + .csv.

Grouping is model x tier x task (the tier label carries the frames mode, e.g. "zero_shot{k3}").
Denominators always include parse failures (Z) and API errors (E): nothing is randomly back-filled.
Several lines for the same (model, tier, frames mode, item, option layout) -- resumed retries, --force, or
reruns -- keep only the latest one (by ts).

Usage:
  python -m cameralogbench.eval.score --annotations data/CameraLogBench/annotations/real.jsonl \
      --results runs/real__*.jsonl [--report runs/real_report]

--annotations supplies gold labels when records lack them and the source curve of curve_absent and camera
items (per-class tables and the default D-Log M exclusion use it).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

from cameralogbench import taxonomy
from cameralogbench.eval import prompts

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def call_key(r: dict) -> tuple:
    """One logical question: a newer record supersedes an older one even when its request_hash
    differs (prompt / digest / image changed between runs)."""
    return (r["model"], r["tier"], r.get("image_variant"), r.get("frames_mode") or "single", r.get("frame_pos"),
            r["item_id"], tuple(r["options"]))


def _ts_key(r: dict) -> float:
    try:
        return datetime.fromisoformat(r["ts"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def load_records(paths: list[Path]) -> tuple[list[dict], int]:
    """-> (latest record per call_key, number of superseded lines). "Latest" is by `ts`; the sort is
    stable, so ties and records without ts keep file/write order (files are append-only)."""
    recs: list[dict] = []
    for p in paths:
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] {p}:{ln} bad JSON skipped", file=sys.stderr)
    recs.sort(key=_ts_key)  # with --results globs, path order is not chronological
    last: dict[tuple, dict] = {}
    for r in recs:
        k = call_key(r)
        last.pop(k, None)  # re-insert so dict order follows the latest write
        last[k] = r
    return list(last.values()), len(recs) - len(last)


def tier_label(r: dict) -> str:
    """tier + frame mode. 'each' groups are per frame position ({each:p50} is the middle-frame condition)."""
    v = r.get("image_variant")
    t = r["tier"] if not v or v in ("jpg", "webp") else f"{r['tier']}[{v}]"
    fm = r.get("frames_mode") or "single"
    if fm == "k3":
        t += "{k3}"
    elif fm == "each":
        t += f"{{each:{r.get('frame_pos')}}}"
    return t


# D-Log M has no published formula: items whose clip is D-Log M are excluded from scoring by default (it stays
# an answer option, so a model that answers it is simply wrong)
DEFAULT_EXCLUDE_CURVES = ("dji_dlogm",)


def excluded_curve(r: dict, exclude: set[str]) -> bool:
    """True when the record's clip is labelled with an excluded curve (curve items by gold, twins by source).
    The excluded curve stays an answer option: a model that answers it is simply wrong."""
    if r["task"] in prompts.TWIN_TASKS or r["task"] == "camera":
        return source_curve(r) in exclude
    return (r.get("gold") or {}).get("primary") in exclude


def accept_of(r: dict) -> list[str]:
    if r.get("accept_ids") is not None:
        return list(r["accept_ids"])
    g = r.get("gold") or {}
    cands = ([g.get("brand")] if r["task"] == "camera" else []) + list(g.get("accept") or []) + [g.get("primary")]
    return [c for c in dict.fromkeys(cands) if c and c in r["options"]]


def source_curve(r: dict) -> Optional[str]:
    """For twin and camera tasks: the curve of the clip (set by attach_annotations)."""
    return r.get("source_curve") or ((r.get("item_meta") or {}).get("source_gold") or {}).get("curve_id")


def load_annotations(paths: list[Path]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                items[it["item_id"]] = it
    return items


def attach_annotations(recs: list[dict], items: dict[str, dict]) -> int:
    """Fill gold / twin_of from the annotations where a record lacks them and set `source_curve` (the clip's
    curve) on every record. -> number of records whose item is not in the annotations."""
    clip_curve = {it.get("clip_id"): it["gold"]["primary"] for it in items.values() if it["task"] == "curve"}
    missing = 0
    for r in recs:
        it = items.get(r["item_id"])
        if it is None:
            missing += 1
            continue
        if r.get("gold") is None:
            r["gold"] = it["gold"]
        if "twin_of" not in r:
            r["twin_of"] = it.get("twin_of")
        if r["task"] in prompts.TWIN_TASKS and it.get("twin_of") in items:
            r["source_curve"] = items[it["twin_of"]]["gold"]["primary"]
        elif r["task"] == "camera":
            r["source_curve"] = clip_curve.get(it.get("clip_id"))
    return missing


def gold_class(r: dict) -> str:
    """Class used for per-class accuracy and the confusion matrix. Twin tasks are all gold =
    undetermined, so they are broken down by the *source* curve instead (correct still means abstained)."""
    if r["task"] in prompts.TWIN_TASKS:
        src = source_curve(r)
        if src:
            return src
    acc = accept_of(r)
    if acc:
        return acc[0]
    g = r.get("gold") or {}
    return str(g.get("brand") if r["task"] == "camera" else g.get("primary"))


def short(opt_id: Optional[str]) -> str:
    if opt_id is None:
        return "?"
    if opt_id in taxonomy.CURVES and opt_id not in taxonomy.ABSTAIN_IDS:
        return taxonomy.CURVES[opt_id].name
    if opt_id in taxonomy.BRANDS:
        return taxonomy.BRANDS[opt_id]
    return opt_id


def pred_label(r: dict) -> str:
    if r.get("error"):
        return "ERR"
    return r.get("choice_id") or "Z"


# --------------------------------------------------------------------------- #
# Evidence score v0 (rough, rule-based)
# --------------------------------------------------------------------------- #
_EV_STRIP = re.compile(
    r"(?:\b(?:rec|bt|itu)\.?\s*-?\s*\d+|\d+\s*-?\s*bits?\b|\blog\s*c?\s*\d|\bs-?log\d|\bv-?log\b|\bgen\s*\d|"
    r"\b\d+\s*:\s*\d+(?:\s*:\s*\d+)?|\b\d+(?:\.\d+)?\s*k\b|\b(?:iso|ei|asa)\s*\d+|\b18\s*%|"
    r"\b(?:fx|a7s?|a1|c|r|s|gh|x-h|x-t|z|zr)\s*-?\d+\w*|\b\d+\s*(?:mm|fps|p)\b|\bf\s*/?\s*\d+(?:\.\d+)?)",
    re.I)
_EV_NUM = re.compile(r"(?<![\w.])(?:0?\.\d+|1\.0+|\d{1,4})(?:\s*%)?(?![\w.])")
_EV_KW = re.compile(
    r"black|shadow|toe|gr[ae]y|mid-?tone|middle|white|highlight|clip|plateau|knee|shoulder|code value|"
    r"\bcv\b|\bire\b|primar|gamut|saturat|chroma|hue", re.I)


def evidence_texts(r: dict) -> list[str]:
    ev = (r.get("parsed") or {}).get("evidence")
    if isinstance(ev, str):
        ev = [ev]
    return [str(e) for e in (ev or []) if str(e).strip()]


def _has_number(s: str) -> bool:
    for m in _EV_NUM.finditer(_EV_STRIP.sub(" ", s)):
        tok = m.group(0).rstrip("% ").strip()
        try:
            v = float(tok)
        except ValueError:
            continue
        if ("." in tok and 0 <= v <= 1) or ("." not in tok and 0 <= v <= 1023):
            return True
    return False


def evidence_scores(r: dict) -> tuple[int, int]:
    """(strict, loose). strict: one evidence item has both a 0-1023 / 0-1 number and a quantitative
    keyword; loose: any item has a number or a quantitative keyword (the literal v0 rule)."""
    items = evidence_texts(r)
    strict = any(_has_number(e) and _EV_KW.search(e) for e in items)
    loose = any(_has_number(e) or _EV_KW.search(e) for e in items)
    return int(strict), int(loose)


# --------------------------------------------------------------------------- #
# Camera model matching
# --------------------------------------------------------------------------- #
# Leading brand / product-line words dropped before comparing model names ("Canon EOS C70" == "C70").
_BRAND_WORDS = {w for b in list(taxonomy.BRANDS.values()) + list(taxonomy.BRANDS)
                for w in re.findall(r"[a-z0-9]+", b.lower())} | {"fuji", "lumix", "eos", "panasonic"}
_FILLER_WORDS = {"series", "cinema", "camera"}


def cam_tokens(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", str(s).lower().replace("\u03b1", "a"))
    toks = [t for t in re.findall(r"[a-z0-9]+", s) if t not in _FILLER_WORDS]
    while toks and toks[0] in _BRAND_WORDS:
        toks.pop(0)
    return toks


def _gold_variants(g: str) -> list[str]:
    """'Zenmuse X5S/X7' -> ['Zenmuse X5S', 'Zenmuse X7'] (an alternative replaces the last word)."""
    head, *alts = str(g).split("/")
    base = head.rsplit(" ", 1)[0] if " " in head else ""
    return [head] + [f"{base} {a}".strip() for a in alts]


def camera_match(pred: list[str], gold: list[str]) -> bool:
    """A prediction hits a gold candidate when, after dropping brand words, the gold model name is a
    whole-token prefix of the prediction: 'DJI Ronin 4D 8K' hits 'DJI Ronin 4D', 'Mavic 3 Pro' hits
    'DJI Mavic 3 series'; a "series" gold also takes letter suffixes ('Air 3S' hits 'Air 3 series').
    Under-specified or different bodies do not: 'Nikon Z' vs 'Nikon ZR', 'DJI Mavic' vs
    'Mavic 3 series', 'FX30' vs 'FX3', a bare brand name."""
    for g in gold:
        series = bool(re.search(r"\bseries\b", str(g), re.I))
        for gv in _gold_variants(g):
            gj = "".join(cam_tokens(gv))
            if len(gj) < 2:
                continue
            for p in pred:
                acc = ""
                for t in cam_tokens(p):
                    acc += t
                    if acc == gj or (series and acc.startswith(gj) and acc[len(gj):].isalpha()):
                        return True
                    if len(acc) >= len(gj):
                        break
    return False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ece_brier(pairs: list[tuple[float, int]], bins: int = 10) -> tuple[float, float]:
    if not pairs:
        return NAN, NAN
    b: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for c, y in pairs:
        b[min(int(c * bins), bins - 1)].append((c, y))
    n = len(pairs)
    ece = sum(len(v) / n * abs(sum(c for c, _ in v) / len(v) - sum(y for _, y in v) / len(v)) for v in b.values())
    brier = sum((c - y) ** 2 for c, y in pairs) / n
    return ece, brier


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else NAN


def _budget(r: dict) -> str:
    p = r.get("request_params")
    if not p:
        return "unrecorded"
    budget = str(p.get("max_completion_tokens") or p.get("max_tokens"))
    eff = p.get("reasoning_effort") or (p.get("output_config") or {}).get("effort")
    return f"{budget}/{eff}" if eff else budget


def group_metrics(rs: list[dict], index: dict[tuple, dict],
                  index_rot: Optional[dict[tuple, dict]] = None) -> dict[str, Any]:
    n = len(rs)
    task = rs[0]["task"]
    valid = [r for r in rs if r.get("choice_id")]
    m: dict[str, Any] = {"n": n, "n_items": len({r["item_id"] for r in rs})}
    m["E"] = sum(1 for r in rs if r.get("error"))
    m["Z"] = sum(1 for r in rs if not r.get("error") and not r.get("choice_id"))
    m["salvaged"] = sum(1 for r in rs if r.get("parse_status") == "salvaged")
    m["acc_top1"] = sum(r.get("correct") is True for r in rs) / n
    m["hit_top2"] = sum(r.get("top2_hit") is True for r in rs) / n

    per_item: dict[str, list[dict]] = defaultdict(list)
    for r in rs:
        per_item[r["item_id"]].append(r)
    m["circ"] = _mean([float(all(x.get("correct") is True for x in v)) for v in per_item.values()])
    multi = [v for v in per_item.values() if len(v) > 1]
    m["agree"] = _mean([float(len({x.get("choice_id") for x in v}) == 1 and v[0].get("choice_id") is not None)
                        for v in multi]) if multi else NAN
    m["incomplete_items"] = sum(1 for v in per_item.values() if len(v) < (v[0].get("n_rotations") or len(v)))

    letters = Counter(r.get("choice_letter") for r in valid)
    if letters:
        let, cnt = letters.most_common(1)[0]
        m["pos_max"], m["pos_max_letter"] = cnt / len(valid), let
    else:
        m["pos_max"], m["pos_max_letter"] = NAN, ""
    m["pos_uniform"] = 1 / len(rs[0]["options"])
    # letter concentration a perfect answerer would show: with an imbalanced gold distribution and few
    # rotations pos_max is far above uniform even without any position bias
    gold_letters = Counter(r["options"].index(accept_of(r)[0]) for r in rs if accept_of(r))
    m["pos_gold"] = max(gold_letters.values()) / n if gold_letters else NAN
    m["k_rot"] = sorted({r.get("n_rotations") for r in rs}, key=str)
    m["budgets"] = Counter(_budget(r) for r in rs)
    m["abstain"] = sum(r.get("choice_id") == taxonomy.UNDETERMINED for r in rs) / n
    m["not_log"] = (sum(r.get("choice_id") == taxonomy.NOT_LOG for r in rs) / n
                    if taxonomy.NOT_LOG in rs[0]["options"] else NAN)

    confs = [(r["confidence"] / 100, int(r.get("correct") is True)) for r in valid
             if isinstance(r.get("confidence"), (int, float))]
    m["n_conf"] = len(confs)
    m["mean_conf"] = _mean([c * 100 for c, _ in confs])
    m["ece10"], m["brier"] = ece_brier(confs)

    accs = [accept_of(r) for r in rs]
    opts = rs[0]["options"]
    m["maj_top1"] = max(_mean([float(o in a) for a in accs]) for o in opts)
    m["maj_top2"] = max(_mean([float(o1 in a or o2 in a) for a in accs]) for o1, o2 in combinations(opts, 2))
    k = len(opts)
    m["rand_top1"] = _mean([len(a) / k for a in accs])
    m["rand_top2"] = _mean([1 - math.comb(k - len(a), 2) / math.comb(k, 2) for a in accs])

    ev = [evidence_scores(r) for r in rs if r.get("parsed") is not None]
    m["ev_v0"] = _mean([s for s, _ in ev])
    m["ev_loose"] = _mean([l for _, l in ev])

    cls: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rs:
        c = cls[gold_class(r)]
        c[0] += r.get("correct") is True
        c[1] += 1
    m["per_class"] = {g: f"{a}/{b}" for g, (a, b) in sorted(cls.items())}
    m["confusion"] = Counter((gold_class(r), pred_label(r)) for r in rs)

    m["dual"], m["dual_n"] = NAN, 0
    m["src_correct"], m["abstain_given_src_correct"] = NAN, NAN
    if task in prompts.TWIN_TASKS:
        # curve_null shares the source's option list -> pair on the exact layout; curve_absent has one
        # option fewer -> pair on the rotation index (same k, same source item)
        hits, src_ok, abst_ok = [], [], []
        for r in rs:
            if task == "curve_null":
                src = index.get((r["model"], tier_label(r), r.get("twin_of"), tuple(r["options"])))
            else:
                src = (index_rot or {}).get((r["model"], tier_label(r), r.get("twin_of"), r.get("rotation")))
            if src is not None:
                abstained = r.get("choice_id") == taxonomy.UNDETERMINED
                hits.append(float(src.get("correct") is True and abstained))
                src_ok.append(float(src.get("correct") is True))
                if src.get("correct") is True:
                    abst_ok.append(float(abstained))
        m["dual"], m["dual_n"] = _mean(hits), len(hits)
        m["src_correct"], m["abstain_given_src_correct"] = _mean(src_ok), _mean(abst_ok)

    m["brand_acc"], m["model_hit"], m["model_hit_n"] = NAN, NAN, 0
    if task == "camera":
        m["brand_acc"] = m["acc_top1"]
        hits = []
        for r in rs:
            g = r.get("gold") or {}
            gc = list(g.get("camera_candidates") or ([g["camera_model"]] if g.get("camera_model") else []))
            if gc:
                hits.append(float(camera_match(r.get("camera_candidates_pred") or [], gc)))
        m["model_hit"], m["model_hit_n"] = _mean(hits), len(hits)
    return m


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def f2(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def clip(s: Any, n: int = 200) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


CSV_COLS = ["model", "tier", "task", "n", "n_items", "Z", "E", "salvaged", "acc_top1", "hit_top2", "circ", "agree",
            "incomplete_items", "dual", "dual_n", "src_correct", "abstain_given_src_correct", "maj_top1", "rand_top1", "maj_top2", "rand_top2", "pos_max",
            "pos_max_letter", "pos_gold", "pos_uniform", "abstain", "not_log", "n_conf", "mean_conf", "ece10", "brier",
            "ev_v0", "ev_loose", "brand_acc", "model_hit", "model_hit_n", "per_class"]


def render(recs: list[dict], groups: dict[tuple, dict], sources: list[str], title: str,
           superseded: int, excluded_note: str = "") -> str:
    L: list[str] = []
    dry = sum(1 for r in recs if r.get("dry_run"))
    L.append(f"# Evaluation report `{title}`\n")
    L.append(f"- generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    L.append(f"- results: {', '.join(f'`{s}`' for s in sources)}; {len(recs)} calls (one record kept per question "
             f"and option layout; {superseded} older records superseded by retries or reruns)")
    L.append(f"- prompt_version: {', '.join(sorted({str(r.get('prompt_version')) for r in recs}))}")
    if excluded_note:
        L.append(f"- {excluded_note}")
    for tier in ("with_spec", "with_reference"):
        ref = Counter(r.get("reference_used") for r in recs if r["tier"] == tier)
        if ref.get(False):
            L.append(f"- WARNING: {ref[False]} {tier} records used the placeholder instead of a digest")
    if dry:
        L.append(f"- WARNING: **{dry} records are --dry-run fake answers; the numbers only test the pipeline**")
    n_err = sum(1 for r in recs if r.get("error"))
    if n_err:
        L.append(f"- WARNING: {n_err} API errors (E). They count as wrong; rerunning the runner without --force "
                 "retries them")
    budgets: dict[str, Counter] = defaultdict(Counter)
    for (mdl, _, _), g in groups.items():
        budgets[mdl].update(g["budgets"])
    L.append("- output token budget / effort (record.request_params): "
             + "; ".join(f"{m} {dict(c)}" for m, c in sorted(budgets.items())))
    mixed = [" / ".join(k) for k, g in sorted(groups.items()) if len(g["budgets"]) > 1]
    if mixed:
        L.append(f"- WARNING: groups mixing different token budgets: {'; '.join(mixed)}")
    single = [" / ".join(k) for k, g in sorted(groups.items()) if g["k_rot"] == [1]]
    if single:
        L.append(f"- note: only one rotation (no CircularEval), so position bias and class prior cannot be "
                 f"separated: {'; '.join(single)}")
    has_conf = any(g["n_conf"] for g in groups.values())
    L.append("")
    L.append("Denominator = every call of the group (parse failures Z and API errors E count as wrong). "
             "acc = final single choice correct; top-2 = gold among the two candidates; "
             "circ = share of items answered correctly in every rotation; agree = share of items with the same "
             "answer in every rotation; dual = source curve item correct and its curve_absent twin (gold option "
             "removed) answered 'cannot be determined' (paired by rotation); majority / random baselines use "
             "this group's gold distribution; pos_max = share of the most chosen letter among valid answers, "
             "to be compared with pos_gold (the same share for a perfect answerer).\n")

    keys = sorted(groups)
    L.append("## 1. Accuracy and baselines\n")
    rows = []
    for k in keys:
        g = groups[k]
        rows.append([k[0], k[1], k[2], str(g["n"]), str(g["Z"]), str(g["E"]), f2(g["acc_top1"]), f2(g["hit_top2"]),
                     f2(g["circ"]), f2(g["dual"]) + (f" (n={g['dual_n']})" if g["dual_n"] else ""),
                     f"{f2(g['maj_top1'])} / {f2(g['rand_top1'])}", f"{f2(g['maj_top2'])} / {f2(g['rand_top2'])}",
                     f2(g["model_hit"]) + (f" (n={g['model_hit_n']})" if g["model_hit_n"] else "")])
    L.append(md_table(["model", "tier", "task", "n", "Z", "E", "acc", "top-2", "circ", "dual",
                       "acc majority / random", "top-2 majority / random", "camera model hit"], rows))
    L.append("\nFor the camera task acc is brand accuracy; camera model hit = any predicted camera_candidates entry "
             "has a gold model name as a whole-word prefix (brand words, brackets and 'series' dropped).\n")

    L.append("## 2. Behaviour and evidence\n")
    rows = []
    for k in keys:
        g = groups[k]
        pos = "-" if math.isnan(g["pos_max"]) else f"{f2(g['pos_max'])} ({g['pos_max_letter']})"
        row = [k[0], k[1], k[2], f"{pos} / {f2(g['pos_gold'])} / {f2(g['pos_uniform'])}",
               f2(g["agree"]), f2(g["abstain"]), f2(g["not_log"])]
        if has_conf:
            row += ["-" if math.isnan(g["mean_conf"]) else f"{g['mean_conf']:.0f}", f2(g["ece10"]), f2(g["brier"])]
        rows.append(row + [f2(g["ev_v0"]), f2(g["ev_loose"])])
    hdr = ["model", "tier", "task", "pos_max / pos_gold / uniform", "agree", "abstain rate", "not_log rate"]
    if has_conf:
        hdr += ["mean conf", "ECE(10)", "Brier"]
    L.append(md_table(hdr + ["evidence v0", "evidence loose"], rows))
    L.append("\nThe evidence scores are rough rule-based indicators, not a check that the evidence is right. v0: some "
             "evidence item contains both a 0-1023 code value or 0-1 decimal and a quantitative keyword (black, "
             "grey, clip, highlight, primaries, ...); loose: a number or a keyword. Non-measurement numbers "
             "(Rec.709, 10-bit, S-Log3, ISO 800, 18%, ...) are stripped first. Denominator: parsable answers.\n")

    L.append("## 3. Per-class accuracy (correct / total by gold; twin tasks by **source curve**, correct = abstained)\n")
    rows = [[k[0], k[1], k[2], "; ".join(f"{short(c)} {v}" for c, v in groups[k]["per_class"].items())] for k in keys]
    L.append(md_table(["model", "tier", "task", "per-class"], rows))
    twin_rows = [[k[0], k[1], k[2], f2(groups[k]["src_correct"]), f2(groups[k]["abstain_given_src_correct"]),
                  f2(groups[k]["abstain"]), f2(groups[k]["dual"])]
                 for k in keys if k[2] in prompts.TWIN_TASKS and groups[k]["dual_n"]]
    if twin_rows:
        L.append("\nTwin pairs: the ideal curve_absent behaviour is 'source item correct, and once the correct option "
                 "is removed, abstain'; picking another curve after answering the source correctly is a guess.\n")
        L.append(md_table(["model", "tier", "task", "source correct", "abstain | source correct", "abstain rate",
                           "dual"], twin_rows))

    L.append("\n## 4. Confusion matrices (gold -> pred counts; Z = parse failure, ERR = API error; "
             "twin rows = source curve)\n")
    for k in keys:
        conf: Counter = groups[k]["confusion"]
        golds = sorted({gk for gk, _ in conf})
        preds = sorted({p for _, p in conf}, key=lambda p: (p in ("Z", "ERR"), p))
        L.append(f"**{k[0]} / {k[1]} / {k[2]}**\n")
        L.append(md_table(["gold \\ pred"] + [short(p) for p in preds],
                          [[short(gk)] + [str(conf.get((gk, p), "")) for p in preds] for gk in golds]))
        L.append("")

    L.append("## 5. Sample answers (one correct and one wrong per model x tier, curve task first)\n")
    by_mt: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        by_mt[(r["model"], tier_label(r))].append(r)
    order = {"curve": 0, "camera": 1, "curve_null": 2, "curve_absent": 3}
    for (mdl, tl), rs in sorted(by_mt.items()):
        rs = sorted(rs, key=lambda r: (order.get(r["task"], 9), r["item_id"], r["rotation"]))
        L.append(f"**{mdl} / {tl}**\n")
        for want, tag in ((True, "correct"), (False, "wrong")):
            r = next((x for x in rs if x.get("choice_id") and x.get("correct") is want), None)
            if r is None:
                L.append(f"- [{tag}] (none)")
                continue
            p = r.get("parsed") or {}
            L.append(f"- [{tag}] `{r['item_id']}` r{r['rotation']} / gold={short(gold_class(r))} / "
                     f"pred={short(r.get('choice_id'))}")
            L.append(f"  - open_answer: {clip(p.get('open_answer'), 220)}")
            for e in evidence_texts(r)[:3]:
                L.append(f"  - evidence: {clip(e)}")
        L.append("")
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="results .jsonl file(s) or glob(s)")
    ap.add_argument("--annotations", nargs="*", default=[], help="annotation .jsonl file(s) of the scored items")
    ap.add_argument("--report", default=None,
                    help="output path prefix for <prefix>.md / <prefix>.csv (default: next to the first results "
                         "file, <name>_report)")
    ap.add_argument("--exclude-curves", nargs="*", default=list(DEFAULT_EXCLUDE_CURVES),
                    help="drop items whose clip has one of these curves (default: dji_dlogm); pass nothing to "
                         "keep all")
    a = ap.parse_args(argv)

    paths = sorted({Path(p) for g in a.results for p in (glob.glob(g) or [g])})
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("error: no results files found", file=sys.stderr)
        return 2
    recs, superseded = load_records(paths)
    if a.annotations:
        missing = attach_annotations(recs, load_annotations([Path(p) for p in a.annotations]))
        if missing:
            print(f"[warn] {missing} records have no item in the annotations", file=sys.stderr)
    recs = [r for r in recs if r.get("gold") is not None or r.get("accept_ids") is not None]
    excl = set(a.exclude_curves or [])
    n_before = len(recs)
    recs = [r for r in recs if not excluded_curve(r, excl)]
    n_excluded = n_before - len(recs)
    note = f"excluded {n_excluded} records whose clip curve is in {sorted(excl)}" if excl else ""
    if note:
        print(f"[score] {note}", file=sys.stderr)
    if not recs:
        print("error: no scorable records (results empty, or gold labels missing: pass --annotations)",
              file=sys.stderr)
        return 2

    index = {(r["model"], tier_label(r), r["item_id"], tuple(r["options"])): r for r in recs}
    index_rot = {(r["model"], tier_label(r), r["item_id"], r.get("rotation")): r for r in recs}
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        buckets[(r["model"], tier_label(r), r["task"])].append(r)
    groups = {k: group_metrics(v, index, index_rot) for k, v in buckets.items()}

    prefix = Path(a.report) if a.report else paths[0].with_name(paths[0].stem + "_report")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md = render(recs, groups, [str(p) for p in paths], prefix.name, superseded, note)
    md_path, csv_path = prefix.with_name(prefix.name + ".md"), prefix.with_name(prefix.name + ".csv")
    md_path.write_text(md, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for (mdl, tl, task), g in sorted(groups.items()):
            row = {c: g.get(c) for c in CSV_COLS}
            row.update(model=mdl, tier=tl, task=task, per_class=json.dumps(g["per_class"], ensure_ascii=False))
            for c, v in row.items():
                if isinstance(v, float):
                    row[c] = "" if math.isnan(v) else round(v, 4)
            w.writerow(row)
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
