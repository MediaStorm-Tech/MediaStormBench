"""Download public scene-linear HDR sources for the synthetic natural-scene set.

Sources:
  * Fairchild HDR Photographic Survey: OpenEXR, Nikon D2x camera RGB, public domain for
    non-commercial research (markfairchild.org/HDR.html).
  * Poly Haven HDRIs: equirectangular linear Rec.709 OpenEXR, CC0 (polyhaven.com).

Writes ``data/synth_sources/sources.jsonl`` (one row per source: dataset, url, licence,
sha1, size). Re-running skips files already on disk with matching size.

Example::

    python -m cameralogbench.synth.fetch_sources --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from cameralogbench._io import ROOT, write_jsonl

OUT = ROOT / "data" / "synth_sources"
FAIRCHILD_BASE = "http://markfairchild.org/HDRPS/"
POLYHAVEN_API = "https://api.polyhaven.com"
UA = {"User-Agent": "CameraLogBench/1.0 (research)"}

# Poly Haven quota per category, filled in this order (an asset counts once).
# The trailing ("indoor", 1) tops up to 100 sources: Fairchild's RoundBarnInside page
# has no EXR, so the survey yields 76.
POLYHAVEN_QUOTA = (("night", 3), ("sunrise-sunset", 3), ("indoor", 7),
                   ("urban", 5), ("nature", 5), ("indoor", 1))
POLYHAVEN_EXCLUDE = {"pure skies", "studio"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fairchild_rows() -> list[dict]:
    thumbs = _get(FAIRCHILD_BASE + "HDRthumbs.html").decode("latin-1")
    scenes = sorted(set(re.findall(r"Scenes/([A-Za-z0-9_\-]+)\.html", thumbs)))
    rows = []
    for name in scenes:
        page = _get(f"{FAIRCHILD_BASE}Scenes/{name}.html").decode("latin-1")
        # a few pages carry a malformed href="http:../EXRs/..."
        m = re.search(r'href="(?:http:)?\.\./EXRs/([^"]+\.exr)"', page, re.I)
        if not m:
            print(f"  fairchild {name}: no EXR link, skipped", flush=True)
            continue
        rows.append({
            "source_id": f"fc_{name}", "dataset": "fairchild_hdrps",
            "url": f"{FAIRCHILD_BASE}EXRs/{m.group(1)}",
            "path": f"data/synth_sources/fairchild/exr/{m.group(1)}",
            "licence": "public domain for non-commercial research (Fairchild HDRPS)",
            "colour": "nikon_d2x_camera_rgb", "projection": "perspective",
            "categories": [],
        })
    return rows


def polyhaven_rows(n: int, res: str) -> list[dict]:
    assets = json.loads(_get(f"{POLYHAVEN_API}/assets?t=hdris"))
    ranked = sorted(assets.items(), key=lambda kv: (-kv[1].get("download_count", 0), kv[0]))
    picked: list[tuple[str, dict]] = []
    taken: set[str] = set()
    for cat, quota in POLYHAVEN_QUOTA:
        got = 0
        for aid, a in ranked:
            cats = set(a.get("categories", []))
            if got >= quota or aid in taken or cat not in cats or cats & POLYHAVEN_EXCLUDE:
                continue
            picked.append((aid, a))
            taken.add(aid)
            got += 1
    picked = picked[:n]
    rows = []
    for aid, a in picked:
        files = json.loads(_get(f"{POLYHAVEN_API}/files/{aid}"))
        f = files["hdri"][res]["exr"]
        rows.append({
            "source_id": f"ph_{aid}", "dataset": "polyhaven_hdri",
            "url": f["url"], "path": f"data/synth_sources/polyhaven/exr/{aid}_{res}.exr",
            "licence": "CC0 (Poly Haven)", "colour": "rec709_linear",
            "projection": "equirectangular", "categories": sorted(a.get("categories", [])),
            "expected_size": f.get("size"),
        })
    return rows


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
    return int(n) if n else None


def download(row: dict) -> dict:
    dst = ROOT / row["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Fairchild's server sometimes closes the connection early without an error, so
    # both the skip test and the finished download are checked against the remote size.
    exp = row.get("expected_size") or remote_size(row["url"])
    if not (dst.is_file() and (exp is None or dst.stat().st_size == exp)):
        tmp = dst.with_suffix(dst.suffix + ".part")
        req = urllib.request.Request(row["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        if exp is not None and tmp.stat().st_size != exp:
            raise RuntimeError(f"truncated: {tmp.stat().st_size} of {exp} bytes")
        tmp.replace(dst)
    h = hashlib.sha1()
    with open(dst, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return {**row, "size": dst.stat().st_size, "sha1": h.hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--polyhaven-n", type=int, default=24)
    ap.add_argument("--polyhaven-res", default="8k")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    rows = fairchild_rows()
    print(f"fairchild scenes with EXR: {len(rows)}", flush=True)
    ph = polyhaven_rows(args.polyhaven_n, args.polyhaven_res)
    print(f"polyhaven picked: {len(ph)}", flush=True)
    rows += ph

    done, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                done.append(fut.result())
                print(f"  {i}/{len(rows)} ok   {r['source_id']}", flush=True)
            except Exception as e:  # keep going; re-run resumes
                failed.append(r["source_id"])
                print(f"  {i}/{len(rows)} FAIL {r['source_id']}: {e}", flush=True)
    done.sort(key=lambda r: r["source_id"])
    for r in done:
        r.pop("expected_size", None)
    write_jsonl(OUT / "sources.jsonl", done)
    print(f"wrote {OUT / 'sources.jsonl'}: {len(done)} ok, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
