"""按界面误差口径选择图片，仅复制到新目录，不改原图或标注。"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .review_data import ReviewItem


def error_images(items: list[ReviewItem]) -> list[str]:
    """任一张脸的取整误差绝对值为 3 或 4 时提取该图；每张只取一次。"""
    paths: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.status != "完成" or item.err_i is None or abs(item.err_i) not in (3, 4):
            continue
        key = os.path.normcase(os.path.realpath(item.image_path))
        if key not in seen:
            seen.add(key)
            paths.append(item.image_path)
    return paths


@dataclass(frozen=True)
class ExtractResult:
    folder: str
    copied: int
    failures: tuple[tuple[str, str], ...]


def copy_error_images(paths: list[str], destination: str) -> ExtractResult:
    parent = Path(destination)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 每次提取单独建目录，绝不覆盖先前提取结果。
    for number in range(10000):
        suffix = f"_{number}" if number else ""
        folder = parent / f"误差3-4_{stamp}{suffix}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise OSError("无法创建新的提取目录")

    copied = 0
    failures: list[tuple[str, str]] = []
    for raw_path in paths:
        source = Path(raw_path)
        destination_path = folder / source.name
        created = False
        try:
            # 不同原目录可以有同名图；用独占创建保留每一张。
            for number in range(10000):
                suffix = f"_{number}" if number else ""
                destination_path = folder / f"{source.stem}{suffix}{source.suffix}"
                try:
                    output = destination_path.open("xb")
                    created = True
                    break
                except FileExistsError:
                    continue
            else:
                raise OSError("同名图片过多，无法创建目标文件")
            with output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            copied += 1
        except OSError as exc:
            if created:
                try:
                    destination_path.unlink()
                except OSError:
                    pass
            failures.append((str(source), str(exc)))
    return ExtractResult(str(folder), copied, tuple(failures))
