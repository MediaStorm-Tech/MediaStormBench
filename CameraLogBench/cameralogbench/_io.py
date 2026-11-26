"""Small file helpers shared by the command-line tools.

Relative paths are resolved against the current working directory (``ROOT``); run the tools from the
repository root so that the default ``data/`` and ``work/`` folders land inside it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

ROOT = Path.cwd()


def repo_rel(p: Path) -> str:
    """Path relative to ROOT when inside it, else absolute."""
    p = Path(os.path.abspath(p))
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def resolve_repo_path(s: Optional[str]) -> Optional[Path]:
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
