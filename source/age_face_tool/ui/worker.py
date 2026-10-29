"""后台线程：批量推断 + 缩略图加载。"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal

from ..core.age_backend import get_backend
from ..core.detector import FaceDetector
from ..core.imageio import make_thumbnail
from ..core.pipeline import ST_FAIL, ImageResult, Pipeline, PipelineConfig, load_backend
from ..core.store import ResultStore


class ThumbSignals(QObject):
    ready = Signal(str, object)  # path, QImage | None


class ThumbTask(QRunnable):
    def __init__(self, path: str, size: int, signals: ThumbSignals) -> None:
        super().__init__()
        self.path = path
        self.size = size
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D102
        from .preview import image_from_bgr

        arr = make_thumbnail(self.path, self.size)
        if arr is None:
            self.signals.ready.emit(self.path, None)
        else:
            self.signals.ready.emit(self.path, image_from_bgr(arr))


class ThumbnailLoader(QObject):
    """带并发上限的缩略图加载器，避免一次性解码几千张图。"""

    def __init__(self, parent=None, size: int = 96, max_threads: int = 3) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max_threads)
        self.size = size
        self.signals = ThumbSignals()
        self.pending: set[str] = set()

    def request(self, path: str) -> None:
        if path in self.pending:
            return
        self.pending.add(path)
        self.pool.start(ThumbTask(path, self.size, self.signals))

    def clear(self) -> None:
        self.pending.clear()


class BatchWorker(QThread):
    sig_progress = Signal(int, int)      # 已完成, 总数
    sig_done = Signal(object)            # ImageResult
    sig_log = Signal(str, str)           # level, message
    sig_model = Signal(bool, str)        # ready, description
    sig_finished = Signal(bool)          # 是否被用户停止
    sig_stage = Signal(str)              # 当前阶段文字

    def __init__(self, paths: list[str], cfg: PipelineConfig, store: ResultStore, parent=None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.cfg = cfg
        self.store = store
        self._stop = False
        self._pause = threading.Event()
        self._pause.set()
        self.done_count = 0
        self.face_count = 0
        self.fail_count = 0

    # ------------------------------------------------------------------ #
    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def stop(self) -> None:
        self._stop = True
        self._pause.set()

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: C901
        stopped_by_user = False
        try:
            self.sig_stage.emit("正在加载年龄模型…")
            t0 = time.perf_counter()
            backend = load_backend(self.cfg.model)
            self.sig_model.emit(True, backend.describe())
            self.sig_log.emit("info", f"年龄模型就绪（{time.perf_counter() - t0:.1f}s）：{backend.describe()}")

            second = None
            if self.cfg.compare_with:
                t1 = time.perf_counter()
                second = load_backend(self.cfg.compare_with)
                self.sig_log.emit("info", f"对比模型就绪（{time.perf_counter() - t1:.1f}s）：{second.describe()}")

            detector = FaceDetector()
            self.sig_log.emit("info", f"人脸检测器就绪：YuNet ({detector.model_path.name})")
            pipeline = Pipeline(detector, backend, self.cfg, second_backend=second)
            self.sig_stage.emit("正在处理图片…")

            total = len(self.paths)
            for i, path in enumerate(self.paths):
                if self._stop:
                    stopped_by_user = True
                    break
                # 暂停：等待恢复
                while not self._pause.wait(0.05):
                    if self._stop:
                        break
                if self._stop:
                    stopped_by_user = True
                    break

                try:
                    res = pipeline.process(path)
                except Exception as e:  # noqa: BLE001  单张失败不影响整体
                    res = ImageResult(path=str(path), rel=Path(path).name, status=ST_FAIL,
                                      error=f"{type(e).__name__}: {e}")
                    self.sig_log.emit("error", f"{Path(path).name}: {e}")
                    self.sig_log.emit("debug", traceback.format_exc())

                try:
                    self.store.append(res)
                except Exception as e:  # noqa: BLE001
                    self.sig_log.emit("warn", f"写入断点记录失败: {e}")

                self.done_count = i + 1
                self.face_count += res.n_faces
                if res.status not in ("成功",):
                    self.fail_count += 1
                self.sig_done.emit(res)
                self.sig_progress.emit(self.done_count, total)

        except Exception as e:  # noqa: BLE001
            self.sig_log.emit("error", f"批量任务异常终止: {type(e).__name__}: {e}")
            self.sig_log.emit("debug", traceback.format_exc())
            self.sig_model.emit(False, str(e))
        finally:
            self.store.close()
            self.sig_stage.emit("已停止" if stopped_by_user else "完成")
            self.sig_finished.emit(stopped_by_user)
