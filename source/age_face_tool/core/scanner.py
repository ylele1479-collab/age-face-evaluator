"""文件夹扫描。"""
from __future__ import annotations

import os
from pathlib import Path

from .imageio import is_image_file


def scan_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    """扫描图片文件，返回按路径自然排序的列表。"""
    root = Path(folder)
    if not root.is_dir():
        return []
    out: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if is_image_file(fn):
                    out.append(Path(dirpath) / fn)
    else:
        for entry in root.iterdir():
            if entry.is_file() and is_image_file(entry.name):
                out.append(entry)
    out.sort(key=lambda p: _natural_key(str(p).lower()))
    return out


def _natural_key(s: str):
    """让 2.jpg 排在 10.jpg 前面。"""
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
