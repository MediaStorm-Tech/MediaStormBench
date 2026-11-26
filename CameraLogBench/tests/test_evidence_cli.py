"""The formula baseline CLI on the fixture dataset (8-bit frames)."""
from __future__ import annotations

import json
from pathlib import Path

from cameralogbench import curves, evidence
from cameralogbench.eval import score

MINI = Path(__file__).parent / "fixtures" / "mini"


def test_formula_baseline_runs_on_released_8bit_frames(tmp_path):
    out = tmp_path / "formula.jsonl"
    try:
        assert evidence.main(["--annotations", str(MINI / "annotations" / "real.jsonl"),
                              str(MINI / "annotations" / "synth_chart.jsonl"), "--data-root", str(MINI),
                              "--out", str(out), "--workers", "1"]) == 0
        # the 17-option items make the candidates the 15 "curve / gamut" encodings
        assert len(evidence.CANDIDATES) == len(curves.CURVE_IDS) + 3
    finally:
        evidence.set_candidates(curves.CURVE_IDS)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["item_id"] for r in recs] == ["R0001__curve", "S0001__curve"]
    assert all(r["choice_id"] in r["options"] and r["model"] == "formula-baseline" for r in recs)
    ev = [json.loads(l) for l in (tmp_path / "formula.evidence.jsonl").read_text().splitlines()]
    assert any("8-bit" in w for w in ev[0]["warnings"])
    assert score.main(["--results", str(out), "--report", str(tmp_path / "rep")]) == 0
