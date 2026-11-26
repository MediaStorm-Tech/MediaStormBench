"""Regression tests for cameralogbench.eval (runner / score), the frame extractor and the fixture dataset."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from cameralogbench import frames
from cameralogbench.eval import prompts, runner, score
from cameralogbench.taxonomy import OPTION_SET_BRAND, OPTION_SET_CURVE, OPTION_SET_CURVE_EXT2

MINI = Path(__file__).parent / "fixtures" / "mini"


@pytest.mark.parametrize("options", [OPTION_SET_CURVE, OPTION_SET_BRAND])
def test_rotations_are_distinct_right_shifts(options):
    for k in range(1, len(options) + 1):
        layouts = [tuple(runner.rotate_options(options, r, k)) for r in range(k)]
        assert len(set(layouts)) == k
        for r, lay in enumerate(layouts):
            s = r * len(options) // k
            assert lay[s] == options[0]


def test_letter_mapping_follows_rotated_list():
    opts = runner.rotate_options(OPTION_SET_CURVE, 1, 2)
    text = prompts.render_options(opts)
    for line in text.splitlines():
        letter = line.split(".")[0]
        assert runner.letter_to_id(letter, opts)[1] == opts[prompts.LETTERS.index(letter)]


@pytest.mark.parametrize("model,max_tokens,expected", [
    ("gemini-3.1-pro-preview", 4000, {"max_tokens": 8192, "temperature": 0}),
    ("gemini-3.1-pro-preview", 16000, {"max_tokens": 16000, "temperature": 0}),
    ("gemini-3.8-flash", 4000, {"max_tokens": 4000, "temperature": 0}),
    ("gpt-6-sol", 4000, {"max_tokens": 4000}),
    ("gpt-6.1-sol", 4000, {"max_completion_tokens": 4000}),
    ("gpt-5.6-mini", 8192, {"max_completion_tokens": 8192}),
    ("o3-mini", 4000, {"max_completion_tokens": 4000}),
    ("claude-sonnet-5-5", 4000, {"max_tokens": 4000, "temperature": 0}),
    ("kimi-k3", 4000, {"max_tokens": 4000, "temperature": 1}),
])
def test_request_params(model, max_tokens, expected):
    assert runner.request_params(model, max_tokens) == expected


def test_claude_with_effort_uses_the_messages_route():
    p = runner.request_params("claude-opus-5-5", 32000, "high")
    assert p == {"_route": runner.ANTHROPIC_ROUTE, "max_tokens": 32000, "thinking": {"type": "adaptive"},
                 "output_config": {"effort": "high"}}
    assert runner.route_of(p) == runner.ANTHROPIC_ROUTE
    assert runner.route_of(runner.request_params("gpt-6.1-sol", 32000, "high")) == "openai"


def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    assert runner.credentials("openai") == ("https://example.test/v1", "k1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        runner.credentials(runner.ANTHROPIC_ROUTE)


@pytest.mark.parametrize("pred,gold,hit", [
    ("DJI Ronin 4D 8K", "DJI Ronin 4D", True),
    ("DJI Ronin 4D (Zenmuse X9-8K)", "DJI Ronin 4D", True),
    ("Mavic 3 Pro", "DJI Mavic 3 series", True),
    ("DJI Air 3S", "DJI Air 3 series", True),
    ("Blackmagic Design PYXIS 6K", "Blackmagic PYXIS 6K", True),
    ("Canon EOS C70", "Canon C70", True),
    ("Panasonic Lumix S5II", "Panasonic S5 II", True),
    ("DJI Zenmuse X7", "Zenmuse X5S/X7", True),
    ("Nikon ZR", "Nikon ZR (RED colour science)", True),
    ("Blackmagic", "Blackmagic PYXIS 6K", False),
    ("PYXIS", "Blackmagic PYXIS 6K", False),
    ("Nikon Z", "Nikon ZR", False),
    ("DJI Mavic", "DJI Mavic 3 series", False),
    ("Sony FX30", "Sony FX3", False),
    ("Canon C300 Mk III", "Canon C300 Mk II", False),
])
def test_camera_match(pred, gold, hit):
    assert score.camera_match([pred], [gold]) is hit


def _write_items(path: Path) -> None:
    rows = []
    for i in range(3):
        rows.append({"item_id": f"c{i}__curve", "clip_id": f"c{i}", "task": "curve",
                     "input": {}, "options": list(OPTION_SET_CURVE),
                     "gold": {"primary": "dji_dlog", "accept": ["dji_dlog"]}, "twin_of": None, "meta": {}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run(tmp_path: Path, *extra: str) -> int:
    return runner.main(["--annotations", str(tmp_path / "items.jsonl"), "--data-root", str(tmp_path),
                        "--out", str(tmp_path / "t.jsonl"), "--models", "m", "--tiers", "text_only",
                        "--tasks", "curve", "--rotations", "2", "--dry-run", "--workers", "1", *extra])


def _lines(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_resume_keeps_parse_failures_and_retry_reasks_every_z(tmp_path):
    _write_items(tmp_path / "items.jsonl")
    assert _run(tmp_path, "--dry-run-fail-rate", "0") == 0
    recs = _lines(tmp_path / "t.jsonl")
    assert len(recs) == 6 and all(r["request_params"] == {"max_tokens": 4000, "temperature": 0} for r in recs)
    # turn every answer into an unmappable letter (parsed != None, choice_id None)
    for r in recs:
        r.update(choice_id=None, choice_letter=None, correct=None, top2_hit=None, parse_status="invalid_choice")
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert _run(tmp_path, "--dry-run-fail-rate", "0") == 0
    assert len(_lines(tmp_path / "t.jsonl")) == 6  # a parse failure is a measured behaviour, not re-asked
    assert _run(tmp_path, "--dry-run-fail-rate", "0", "--retry-parse-failures") == 0
    assert len(_lines(tmp_path / "t.jsonl")) == 12
    kept, superseded = score.load_records([tmp_path / "t.jsonl"])
    assert superseded == 6 and all(r["choice_id"] for r in kept)


def test_rotations_above_option_count_rejected(tmp_path):
    _write_items(tmp_path / "items.jsonl")
    assert _run(tmp_path, "--rotations", str(len(OPTION_SET_CURVE) + 1)) == 2


def test_latest_record_wins_by_timestamp_not_path_order(tmp_path):
    base = {"model": "m", "tier": "zero_shot", "image_variant": "webp", "item_id": "x", "options": ["a", "b"]}
    (tmp_path / "a_new.jsonl").write_text(json.dumps(dict(base, ts="2026-10-20T16:00:00+08:00", choice_id="b")) + "\n")
    (tmp_path / "z_old.jsonl").write_text(json.dumps(dict(base, ts="2026-10-20T13:00:00+08:00", choice_id="a")) + "\n")
    kept, superseded = score.load_records([tmp_path / "a_new.jsonl", tmp_path / "z_old.jsonl"])
    assert superseded == 1 and kept[0]["choice_id"] == "b"


def test_curve_absent_dual_pairs_on_rotation_and_breaks_down_by_source_curve():
    base = {"model": "m", "tier": "zero_shot", "image_variant": "webp", "ts": "2026-10-22T10:00:00+08:00"}
    src_opts = list(OPTION_SET_CURVE)
    twin_opts = [o for o in src_opts if o != "red_log3g10"]
    recs = []
    for rot, (src_ok, twin_choice) in enumerate([(True, "undetermined"), (True, "arri_logc4")]):
        recs.append(dict(base, item_id="c__curve", task="curve", rotation=rot, options=runner.rotate_options(src_opts, rot, 2),
                         gold={"primary": "red_log3g10", "accept": ["red_log3g10"]}, accept_ids=["red_log3g10"],
                         choice_id="red_log3g10" if src_ok else "arri_logc4", correct=src_ok, top2_hit=src_ok,
                         choice_letter="A", twin_of=None, n_rotations=2))
        recs.append(dict(base, item_id="c__curve_absent", task="curve_absent", rotation=rot,
                         options=runner.rotate_options(twin_opts, rot, 2), twin_of="c__curve",
                         gold={"primary": "undetermined", "accept": ["undetermined"]}, accept_ids=["undetermined"],
                         choice_id=twin_choice, correct=twin_choice == "undetermined", top2_hit=None,
                         choice_letter="A", n_rotations=2))
    items = {"c__curve": {"item_id": "c__curve", "clip_id": "c", "task": "curve", "twin_of": None,
                          "gold": {"primary": "red_log3g10"}},
             "c__curve_absent": {"item_id": "c__curve_absent", "clip_id": "c", "task": "curve_absent",
                                 "twin_of": "c__curve", "gold": {"primary": "undetermined"}}}
    assert score.attach_annotations(recs, items) == 0
    index = {(r["model"], score.tier_label(r), r["item_id"], tuple(r["options"])): r for r in recs}
    index_rot = {(r["model"], score.tier_label(r), r["item_id"], r["rotation"]): r for r in recs}
    twins = [r for r in recs if r["task"] == "curve_absent"]
    m = score.group_metrics(twins, index, index_rot)
    assert m["dual_n"] == 2 and m["dual"] == 0.5 and m["src_correct"] == 1.0 and m["abstain_given_src_correct"] == 0.5
    assert m["per_class"] == {"red_log3g10": "1/2"}  # grouped by the removed source curve, not by "undetermined"
    assert m["confusion"][("red_log3g10", "arri_logc4")] == 1
    # without the rotation index curve_absent cannot be paired, but nothing breaks
    assert score.group_metrics(twins, index)["dual_n"] == 0


# --------------------------------------------------------------------------- #
# End to end on the fixture dataset
# --------------------------------------------------------------------------- #
def test_dry_run_end_to_end_on_fixture_dataset_then_score(tmp_path):
    out = tmp_path / "results" / "mini.jsonl"
    args = ["--annotations", str(MINI / "annotations" / "real.jsonl"), str(MINI / "annotations" / "synth_chart.jsonl"),
            "--data-root", str(MINI), "--out", str(out), "--models", "gpt-x", "claude-y",
            "--tiers", "zero_shot", "with_spec", "with_reference", "--frames", "k3", "--rotations", "2",
            "--effort", "high", "--max-tokens", "32000", "--dry-run", "--workers", "2"]
    assert runner.main(args) == 0
    recs = _lines(out)
    # 2 models x 3 tiers x 4 items x 2 rotations
    assert len(recs) == 48 and not any(r["error"] for r in recs)
    for r in recs:
        assert set(r["item_meta"]) <= {"subset", "source_class"} and "endpoint" not in r
        assert r["image_paths"] and all(not Path(p).is_absolute() for p in r["image_paths"])
        assert r["prompt_version"] == prompts.PROMPT_VERSION and r["dry_run"]
    real = [r for r in recs if r["clip_id"] == "R0001"]
    chart = [r for r in recs if r["clip_id"] == "S0001"]
    assert all(r["frames_mode"] == "k3" and r["frame_positions"] == ["p25", "p50", "p75"] for r in real)
    assert all(r["frames_mode"] == "single" and r["image_paths"] == ["frames/synth_chart/S0001/frame.webp"]
               for r in chart)  # single-image subsets stay single under --frames k3
    # the digest follows each clip's option set: 17 options -> digest_ext2, 14 -> digest
    sha = {n: runner.sha1_text(prompts.load_reference(runner.REFERENCE_DIR / n)[0])
           for n in ("digest.md", "digest_spec.md", "digest_ext2.md", "digest_ext2_spec.md")}
    for r in recs:
        if r["tier"] == "zero_shot":
            assert r["reference_sha1"] is None
            continue
        want = ("digest_ext2" if r["clip_id"] == "R0001" else "digest") + ("_spec" if r["tier"] == "with_spec" else "")
        assert r["reference_sha1"] == sha[want + ".md"] and r["reference_used"]
    assert {r["request_params"].get("_route") for r in recs if r["model"] == "claude-y"} == {runner.ANTHROPIC_ROUTE}
    # resume: nothing left to do
    assert runner.main(args) == 0 and len(_lines(out)) == 48

    report = tmp_path / "reports" / "mini"
    assert score.main(["--results", str(out), "--annotations", str(MINI / "annotations" / "real.jsonl"),
                       str(MINI / "annotations" / "synth_chart.jsonl"), "--report", str(report)]) == 0
    md = (tmp_path / "reports" / "mini.md").read_text(encoding="utf-8")
    assert "dry-run" in md and "## 1. Accuracy" in md
    rows = list(csv.DictReader((tmp_path / "reports" / "mini.csv").open(encoding="utf-8")))
    groups = {(r["model"], r["tier"], r["task"]) for r in rows}
    assert ("gpt-x", "zero_shot{k3}", "curve") in groups and ("claude-y", "with_spec{k3}", "curve_absent") in groups
    absent = next(r for r in rows if r["task"] == "curve_absent" and r["tier"] == "zero_shot{k3}")
    assert json.loads(absent["per_class"]).keys() == {"sony_slog3"}  # twin grouped by its source curve


def test_oracle_dry_run_scores_perfectly(tmp_path):
    items = [json.loads(l) for l in (MINI / "annotations" / "real.jsonl").read_text().splitlines()]
    for it in items:
        it["meta"]["dry_run_oracle"] = True
    ann = tmp_path / "oracle.jsonl"
    ann.write_text("".join(json.dumps(it) + "\n" for it in items))
    out = tmp_path / "oracle_results.jsonl"
    assert runner.main(["--annotations", str(ann), "--data-root", str(MINI), "--out", str(out), "--models", "m",
                        "--tiers", "zero_shot", "--frames", "each", "--rotations", "3", "--dry-run"]) == 0
    recs = _lines(out)
    assert len(recs) == 3 * 3 * 3 and all(r["correct"] for r in recs)
    assert {r["frame_pos"] for r in recs} == {"p25", "p50", "p75"}


# --------------------------------------------------------------------------- #
# Frame extraction from video
# --------------------------------------------------------------------------- #
def test_frames_extracts_three_lossless_frames_and_writes_annotations(tmp_path):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vids = tmp_path / "videos"
    vids.mkdir()
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
                    "-frames:v", "12", "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
                    str(vids / "A001.mov")], check=True)
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "curve", "camera_model"])
        w.writerow(["A001.mov", "S-Log3", "Sony FX3"])
    root = tmp_path / "data"
    assert frames.main(["--videos", str(vids), "--out-root", str(root), "--subset", "mine",
                        "--labels", str(tmp_path / "labels.csv"), "--workers", "1"]) == 0
    for pos in ("p25", "p50", "p75"):
        assert (root / "frames" / "mine" / "A001" / f"{pos}.webp").is_file()
    # p50 = frame n // 2, decoded at 16 bit, >> 8 (the frame is narrower than 1920, so no resize)
    img16, _ = frames.extract_frame16(vids / "A001.mov", 6)
    import cv2
    got = cv2.imread(str(root / "frames" / "mine" / "A001" / "p50.webp"), cv2.IMREAD_UNCHANGED)
    assert (got == frames.to8(img16)).all()
    items = _lines(root / "annotations" / "mine.jsonl")
    assert [it["task"] for it in items] == ["curve", "camera", "curve_absent"]
    cur, cam, absent = items
    assert cur["gold"]["primary"] == "sony_slog3" and cur["options"] == OPTION_SET_CURVE_EXT2
    assert cam["gold"]["primary"] == "sony" and cam["options"] == OPTION_SET_BRAND
    assert absent["twin_of"] == cur["item_id"] and "sony_slog3" not in absent["options"]
    assert len(absent["options"]) == len(OPTION_SET_CURVE_EXT2) - 1 and absent["input"] == cur["input"]
    # the prompt is the plain curve prompt, just with one option fewer
    assert prompts.build_user_text("curve_absent", "zero_shot", absent["options"]).count("\n") == \
        prompts.build_user_text("curve", "zero_shot", cur["options"]).count("\n") - 1
    # and the runner reads the result directly
    out = tmp_path / "r.jsonl"
    assert runner.main(["--annotations", str(root / "annotations" / "mine.jsonl"), "--data-root", str(root),
                        "--out", str(out), "--models", "m", "--tiers", "zero_shot", "--frames", "k3",
                        "--rotations", "1", "--dry-run"]) == 0
    assert len(_lines(out)) == 3
