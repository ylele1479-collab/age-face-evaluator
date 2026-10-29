"""断点续跑：把每张图片的结果追加到本地 JSONL 状态文件。

- 状态文件位于 %LOCALAPPDATA%\\AgeFaceTool\\state\\<文件夹哈希>.jsonl
- **不会写入用户的图片目录**，原始图片只读不改。
- 中断后再次开始，会按 (路径, 文件大小, 修改时间) 跳过已完成项。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from ..config import state_dir
from .pipeline import ImageResult


def _folder_key(folder: str | Path) -> str:
    norm = os.path.normcase(os.path.abspath(str(folder)))
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


class ResultStore:
    def __init__(self, folder: str | Path) -> None:
        self.folder = str(folder)
        self.path = state_dir() / f"{_folder_key(folder)}.jsonl"
        self._lock = threading.Lock()
        self._fh = None

    # ------------------------------------------------------------------ #
    def load(self) -> dict[str, ImageResult]:
        """读取已有结果，返回 {绝对路径: ImageResult}。"""
        out: dict[str, ImageResult] = {}
        if not self.path.exists():
            return out
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        r = ImageResult.from_dict(d)
                    except Exception:
                        continue
                    out[os.path.normcase(r.path)] = r
        except OSError:
            return out
        return out

    def append(self, result: ImageResult) -> None:
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")
            self._fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            self._fh.flush()

    def compact(self, results: dict[str, ImageResult]) -> None:
        """把当前结果整体重写（去重、压缩文件体积）。"""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in results.values():
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self.path)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def clear(self) -> None:
        self.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def is_fresh(result: ImageResult, path: Path) -> bool:
    """已有结果是否仍然对应当前文件（大小/修改时间一致）。"""
    try:
        st = path.stat()
    except OSError:
        return False
    if result.size and st.st_size != result.size:
        return False
    if result.mtime and abs(st.st_mtime - result.mtime) > 1.0:
        return False
    return True
