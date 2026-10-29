"""同步诊断只写本机应用数据目录，不保存截图或改动标注数据。"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from threading import Lock

from ..config import app_data_dir

_lock = Lock()
_logger: logging.Logger | None = None


def sync_log(event: str, **fields) -> None:
    global _logger
    try:
        with _lock:
            if _logger is None:
                logger = logging.getLogger("age_face_review.sync")
                logger.setLevel(logging.INFO)
                logger.propagate = False
                folder = app_data_dir() / "logs"
                folder.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    folder / "review-sync.log", maxBytes=1_000_000,
                    backupCount=2, encoding="utf-8")
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s %(threadName)s %(message)s"))
                logger.addHandler(handler)
                _logger = logger
        # %r 把换行转义，便于按一次请求追踪，不允许日志内容伪造新行。
        _logger.info("%s %r", event, fields)
    except Exception:
        # 无法写日志不能影响快捷键或同步。
        pass
