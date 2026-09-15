"""Shared helpers for the archive tools: iterate archive designs, load indexed TIFFs, write profile
fields back. An archive design dir holds design.tiff (+ palette.csv); a synthetic/real pair dir
holds truth.tiff."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.job import CONFIG_DIR, DATA_DIR, load_yaml  # noqa: E402


def iter_designs(archive_dir: Path | None = None):
    """Yields (id, indexed (H,W) int32, palette_rgb) for every design.tiff / truth.tiff under the dir."""
    d = Path(archive_dir) if archive_dir else DATA_DIR / "archive" / "designs"
    for p in sorted(d.iterdir()) if d.exists() else []:
        if not p.is_dir():
            continue
        f = p / "design.tiff" if (p / "design.tiff").exists() else p / "truth.tiff"
        if not f.exists():
            continue
        idx, rgb = read_indexed_tiff(f)
        yield p.name, idx, rgb


def merge_same_rgb(indexed: np.ndarray, palette_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Archive designs may hold several indices with identical RGB; merge them (lowest index survives)."""
    if palette_rgb is None:
        return indexed, palette_rgb
    seen, remap, keep = {}, {}, []
    for i, c in enumerate(palette_rgb):
        key = tuple(int(x) for x in c)
        if key in seen:
            remap[i] = seen[key]
        else:
            seen[key] = len(keep)
            remap[i] = len(keep)
            keep.append(i)
    out = np.vectorize(lambda v: remap.get(int(v), 0))(indexed).astype(np.int32)
    return out, palette_rgb[keep]


def write_profile_fields(fields: dict, profile: str = "wilton_default") -> Path:
    """Rewrite scalar fields in config/looms/<profile>.yaml in place (comments preserved)."""
    p = CONFIG_DIR / "looms" / f"{profile}.yaml"
    lines = p.read_text().splitlines()
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\w+):\s*([^#]*)(#.*)?$", line)
        if not m or m.group(1) not in fields:
            continue
        k = m.group(1)
        comment = m.group(3) or ""
        val = fields[k]
        sval = "null" if val is None else (str(val).lower() if isinstance(val, bool) else str(val))
        lines[i] = f"{k}: {sval}" + (f"  {comment}" if comment else "")
        done.add(k)
    for k, v in fields.items():
        if k not in done:
            lines.append(f"{k}: {v}")
    p.write_text("\n".join(lines) + "\n")
    return p
