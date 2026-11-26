"""Formula-driven expert reasoning chain for one synthetic sample.

Each step compares a measured cue with the gold curve's whitepaper value.  Cues
the image does not contain (no grey card, no true black, no clip plateau, a clip
inherited from re-encoded source footage) are reported as unavailable; they are
never filled in from the gold label.  ``evidence_sufficient`` is True only when
the formula check alone ranks the gold curve first and does not rule it out;
chains with False describe an ambiguous frame and must not be used as
"the evidence proves X" exemplars.
"""
from __future__ import annotations

from typing import Any, Optional

from cameralogbench import curves
from cameralogbench.taxonomy import CURVES, NOT_LOG

CLIP_MATCH_TOL_CV = 6.0


def _f(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def build_reasoning(sample_id: str, gold_curve: str, evidence: dict,
                    ei: Optional[int] = None, exposure_stops: float = 0.0,
                    source_class: str = "S-main", sensor_clip_applied: bool = True,
                    extra_meta: Optional[dict] = None) -> dict[str, Any]:
    """Templated reasoning steps from measured evidence vs whitepaper anchors.

    ``sensor_clip_applied``: the generator clamped highlights at the curve's
    documented sensor clip (chart / linear_dir).  Re-encodes keep the source
    footage's clip, so a plateau there says nothing about the target curve.
    """
    cc = evidence.get("curve_consistency") or {}
    black = evidence.get("black") or {}
    hl = evidence.get("highlight") or {}
    pred = evidence.get("classical_prediction") or {}
    ranked = pred.get("ranked") or []
    ruled_out = pred.get("ruled_out") or []
    ref = {p["name"]: p for p in (evidence.get("reference_patches") or [])}
    est, fine = black.get("estimate_cv"), black.get("fine_floor_cv")
    tol = black.get("violation_tol_cv") or 5.0
    has_tb = bool(black.get("has_true_black"))
    plateau = hl.get("plateau_cv")

    if gold_curve == NOT_LOG:
        log_blacks = {c: curves.expected_code_values(c)["cv10_float"]["black"] for c in curves.CURVE_IDS}
        remaining = [c for c in curves.CURVE_IDS if c not in ruled_out]
        steps = [{
            "cue": "formula_exclusion",
            "observed": {"black_floor_cv": est, "fine_floor_cv": fine, "has_true_black": has_tb,
                         "plateau_cv": plateau, "ruled_out": ruled_out, "not_ruled_out": remaining},
            "expected_by_curve": log_blacks,
            "conclusion": (
                f"Formula tests exclude {len(ruled_out)}/{len(curves.CURVE_IDS)} log curves"
                + (f"; still compatible: {', '.join(remaining)}" if remaining else "")
                + ". The classical check has no not_log hypothesis, so it can only support not_log "
                  "by excluding every log curve."),
        }]
        sufficient = not remaining
        summary = ("Generator label not_log: a Rec.709 OETF render or a desaturated pseudo-log grade, "
                   f"not a camera log encoding. Black floor {_f(est)} "
                   f"({'true black' if has_tb else 'no true black'}); "
                   + ("formula tests exclude every log curve."
                      if sufficient else
                      f"formula tests leave {', '.join(remaining)} open, so the evidence alone does not "
                      "establish not_log."))
        return _pack(sample_id, gold_curve, steps, summary, ranked, source_class, extra_meta,
                     sufficient, None)

    exp = curves.expected_code_values(gold_curve, ei)
    cv = exp["cv10_float"]
    name = CURVES[gold_curve].name if gold_curve in CURVES else gold_curve
    steps: list[dict[str, Any]] = []

    # 1) black level: two-sided only with a true black, otherwise one-sided (floor is a scene property)
    b = cv["black"]
    if est is None:
        black_concl = "No black-level estimate."
    elif has_tb:
        d = est - b
        black_concl = (f"True black present: floor {est:.1f} vs {name} black {b:.1f} "
                       f"(delta {d:+.1f}, tol {tol:.1f}): "
                       + ("consistent." if abs(d) <= tol else "inconsistent; this test rules the curve out."))
    elif fine is not None and fine < b - tol:
        black_concl = (f"No true black, but averaged shadows reach {fine:.1f}, below {name} black "
                       f"{b:.1f} - {tol:.1f}: incompatible.")
    else:
        black_concl = (f"No true black: darkest averaged shadows ({_f(fine)}) sit at or above {name} black "
                       f"{b:.1f}; compatible but not diagnostic (one-sided).")
    steps.append({
        "cue": "black_level",
        "observed": {"estimate_cv": est, "fine_floor_cv": fine, "has_true_black": has_tb,
                     "violation_tol_cv": tol},
        "expected_by_curve": {gold_curve: b},
        "conclusion": black_concl,
    })

    # 2) 18 % grey: only a known reference patch counts; scene "neutrals" have unknown reflectance
    grey_meas = None
    if "grey18" in ref:
        grey_meas = float(sum(ref["grey18"]["measured_cv"]) / 3.0)
        grey_expected = float(curves.encode(gold_curve, 0.18 * (2.0 ** exposure_stops), ei)) * 1023.0
        fit = (cc.get(gold_curve) or {}).get("reference_fit") or {}
        steps.append({
            "cue": "grey_18_percent",
            "observed": {"mean_cv": round(grey_meas, 2),
                         "fitted_exposure_stops": fit.get("exposure_stops"),
                         "neutral_rms_cv": fit.get("neutral_rms_cv")},
            "expected_by_curve": {gold_curve: round(grey_expected, 2)},
            "note": f"whitepaper grey18 at nominal exposure: {cv['grey18']}",
            "conclusion": (f"Grey card at {grey_meas:.1f} CV; {name} puts 18% grey at {grey_expected:.1f} CV "
                           f"at the generator exposure ({exposure_stops:+g} stop), delta "
                           f"{grey_meas - grey_expected:+.2f}. Neutral ladder fit: RMS "
                           f"{_f(fit.get('neutral_rms_cv'), 2)} CV at {_f(fit.get('exposure_stops'), 2)} stop."),
        })
    else:
        steps.append({
            "cue": "grey_18_percent",
            "observed": {"mean_cv": None, "reason": "no known grey reference in frame"},
            "expected_by_curve": {gold_curve: cv["grey18"]},
            "note": "whitepaper grey18 at nominal exposure; scene exposure unknown",
            "conclusion": (f"No grey card: exposure is unknown, so {name}'s 18% grey anchor "
                           f"({cv['grey18']:.1f} CV at nominal exposure) cannot be checked."),
        })

    # 3) highlight / clip
    clip_exp = cv["clip"]
    if plateau is None:
        clip_concl = (f"No clip plateau (robust max {hl.get('robust_max_cv')}); clip-level cue unavailable.")
    elif not sensor_clip_applied:
        clip_concl = (f"Plateau {plateau} is inherited from the re-encoded source footage; {name}'s camera "
                      f"clip ({clip_exp}) is not simulated, so the plateau is not diagnostic.")
    else:
        d = plateau - clip_exp
        clip_concl = (f"Plateau {plateau} vs {name} clip {clip_exp} (delta {d:+.1f}): "
                      + ("matches." if abs(d) <= CLIP_MATCH_TOL_CV else "does not match."))
    steps.append({
        "cue": "highlight_clip",
        "observed": {"plateau_cv": plateau, "max_cv": hl.get("max_cv"),
                     "clipped_fraction": hl.get("clipped_fraction"),
                     "sensor_clip_simulated": sensor_clip_applied},
        "expected_by_curve": {gold_curve: clip_exp},
        "conclusion": clip_concl,
    })

    # 4) formula-only ranking
    entry = cc.get(gold_curve) or {}
    gold_rank = ranked.index(gold_curve) + 1 if gold_curve in ranked else None
    top = ranked[0] if ranked else None
    gold_ruled = gold_curve in ruled_out
    sufficient = top == gold_curve and not gold_ruled
    if top is None:
        rank_concl = "No classical ranking."
    elif sufficient:
        rank_concl = f"The formula check alone recovers {name} (rank 1 of {len(ranked)})."
    elif gold_ruled:
        terms = {k: v for k, v in (entry.get("penalty_terms") or {}).items() if v >= 1.0}
        rank_concl = (f"The formula check rules {name} out ({terms}); the label comes from the generator, "
                      "so this frame is a known failure case of the classical check.")
    else:
        rank_concl = (f"Not decisive: {top} ranks first (penalty {cc.get(top, {}).get('penalty')}), {name} "
                      f"ranks {gold_rank} of {len(ranked)} (penalty {entry.get('penalty')}); the label comes "
                      "from the generator, not from this evidence.")
    steps.append({
        "cue": "formula_rank",
        "observed": {
            "classical_top3": ranked[:3],
            "gold_rank": gold_rank,
            "gold_ruled_out": gold_ruled,
            "gold_penalty": entry.get("penalty"),
            "gold_score": entry.get("score"),
            "reference_fit": entry.get("reference_fit"),
        },
        "expected_by_curve": {gold_curve: "lowest combined physical+reference penalty"},
        "conclusion": rank_concl,
    })

    # 5) competitors with similar black
    near = []
    for cid in curves.CURVE_IDS:
        if cid == gold_curve:
            continue
        if abs(curves.expected_code_values(cid)["cv10_float"]["black"] - b) <= 5:
            near.append(cid)
    if near:
        steps.append({
            "cue": "confusable_black_family",
            "observed": {"similar_black_curves": near},
            "expected_by_curve": {c: curves.expected_code_values(c)["cv10_float"]["black"] for c in near},
            "conclusion": ("Several curves share ~the same black; grey slope, clip code and "
                           "ColorChecker chroma (reference_fit) have to separate them."),
        })

    conf = curves.CURVE_SPECS[gold_curve].confidence
    summary = (
        f"Synthetic {source_class} frame, generator label {name}"
        + (f" @EI{ei}" if ei is not None else "")
        + (f", exposure {exposure_stops:+g} stop" if exposure_stops else "")
        + f". Whitepaper anchors (10-bit): black {cv['black']}, grey18 {cv['grey18']}, "
          f"white90 {cv['white90']}, clip {cv['clip']}. "
        + f"Measured: black floor {_f(est)} ({'true black' if has_tb else 'no true black'}), "
          f"grey card {_f(grey_meas)}, plateau {_f(plateau)}. "
        + (f"Formula-only check: top-1 {top}, gold rank {gold_rank}/{len(ranked)}"
           + ("; consistent with the label. " if sufficient else
              "; the evidence alone does not single out the label. ")
           if top else "")
        + f"Formula confidence for this curve: {conf}."
    )
    return _pack(sample_id, gold_curve, steps, summary, ranked, source_class, extra_meta,
                 sufficient, gold_rank)


def _pack(sample_id, gold, steps, summary, ranked, source_class, extra_meta, sufficient, gold_rank):
    return {
        "sample_id": sample_id,
        "gold_curve_id": gold,
        "source_class": source_class,
        "steps": steps,
        "expert_summary": summary,
        "evidence_sufficient": sufficient,
        "gold_rank": gold_rank,
        "classical_ranked": ranked,
        "meta": extra_meta or {},
    }
