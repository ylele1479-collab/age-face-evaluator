"""年龄复查助手：只读辅助界面。

给"人工复查年龄标注"用：拖进文件夹/图片 → 逐张显示模型估计年龄与类别倾向，
旁边同时列出 json 里已有的候选（原始 / TS / Cake），帮人判断该选哪个。

查看和同步不改动原图或标注；只有明确的保存/撤销按钮写入 json。
提取只复制原图到新文件夹，日志只写本机应用数据目录。
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from threading import Lock

from PySide6.QtCore import QFileSystemWatcher, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from ..config import APP_NAME
from ..core.anno_writer import backup_root, is_gender_save_candidate, restore_from_map, save_defaults
from ..core.imageio import imread_bgr
from ..core.review_data import GENDER_TEXT, ReviewItem, build_items, collect_images, parse_box, valid_age
from ..core.win_hook import KeyWatcher, _force_foreground, find_window_by_title, is_foreground
from ..core.legacy_sync import LegacySyncWorker
from ..core.legacy_buttons import SYNC_REVISION
from ..core.error_extract import copy_error_images, error_images
from ..core.sync_log import sync_log
from .preview import FacePreview, image_from_bgr

ACCENT = QColor("#3d82f0")
OK = QColor("#00E5A0")          # 误差 ≤2
YELLOW = QColor("#FFD166")      # 误差 ≤5
WARN = QColor("#FF9F43")        # 误差 ≤10
RED = QColor("#FF5C5C")         # 误差 >10
MUTED = QColor("#8b93a1")
BAND_COLORS = {"10-19": QColor("#3d82f0"), "20-29": QColor("#FF9F43")}


# ---------------------------------------------------------------------- #
# 类别概率条
# ---------------------------------------------------------------------- #
class ProbBars(QWidget):
    """FairFace 9 个粗类别的概率条，重点突出 10-19 / 20-29。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: list[str] = []
        self._probs: list[float] = []
        self.scale = 1.0
        self.setMinimumHeight(150)
        self.setMaximumHeight(160)

    def set_probs(self, labels: list[str], probs: list[float]) -> None:
        self._labels = list(labels)
        self._probs = list(probs)
        self.update()

    def clear(self) -> None:
        self._labels, self._probs = [], []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#202027"))
        if not self._labels:
            p.setPen(MUTED)
            p.setFont(QFont("Microsoft YaHei UI", max(6, round(9 * self.scale))))
            p.drawText(self.rect(), Qt.AlignCenter, "等待模型结果…")
            p.end()
            return

        row_h = self.height() / max(1, len(self._labels))
        label_w = round(62 * self.scale)
        pct_w = round(44 * self.scale)
        bar_x = label_w
        bar_w = max(20, self.width() - label_w - pct_w - 8)
        font = QFont("Microsoft YaHei UI", max(6, round(8 * self.scale)))
        p.setFont(font)
        for i, (lab, prob) in enumerate(zip(self._labels, self._probs)):
            y = i * row_h
            cy = y + row_h / 2
            hot = lab in BAND_COLORS
            p.setPen(BAND_COLORS.get(lab, QColor("#b9c4d4")) if hot else QColor("#8b93a1"))
            f = QFont("Microsoft YaHei UI", max(6, round(8 * self.scale)))
            f.setBold(hot)
            p.setFont(f)
            p.drawText(QRectF(0, y, label_w - 6, row_h), Qt.AlignRight | Qt.AlignVCenter, lab)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#2b2b33"))
            p.drawRoundedRect(QRectF(bar_x, cy - 5, bar_w, 10), 5, 5)
            w = max(2.0, bar_w * max(0.0, min(1.0, prob)))
            p.setBrush(BAND_COLORS.get(lab, QColor("#4a5a72")))
            p.drawRoundedRect(QRectF(bar_x, cy - 5, w, 10), 5, 5)
            p.setPen(QColor("#e6e6e6") if hot else QColor("#9aa4b2"))
            f2 = QFont("Microsoft YaHei UI", max(6, round(8 * self.scale)))
            f2.setBold(hot)
            p.setFont(f2)
            p.drawText(QRectF(bar_x + bar_w + 4, y, pct_w, row_h),
                       Qt.AlignRight | Qt.AlignVCenter, f"{prob * 100:.0f}%")
        p.end()


# ---------------------------------------------------------------------- #
# 后台推理
# ---------------------------------------------------------------------- #
BATCH = 12          # 每次前向一起算多少张脸（批量比逐张快 3~5 倍）


class ReviewWorker(QThread):
    computed = Signal(int, dict)
    state_changed = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue: deque[int] = deque()
        self._prio: deque[int] = deque()
        self._items: dict[int, ReviewItem] = {}
        self._lock = Lock()
        self._stop = False
        self._backend = None
        self._detector = None
        self.ready = False
        self.load_error = ""
        self._senior_reference = None
        self._reference_error = ""
        self._feature_error = ""
        self._detector_error = ""

    def _reference_output(self, age: float, features) -> str:
        if age < 50:
            return "60/70/80 岁参考图仅用于老年段；当前沿用原模型估计。"
        if features is None:
            return "年龄参考辅助暂不可用；沿用原模型估计。"
        try:
            if self._senior_reference is None:
                if self._reference_error:
                    raise RuntimeError(self._reference_error)
                from ..core.senior_reference import SeniorAgeReference
                try:
                    self._senior_reference = SeniorAgeReference(self._backend)
                except Exception as exc:
                    self._reference_error = str(exc)
                    raise
            return self._senior_reference.describe(age, features)
        except Exception as exc:
            sync_log("reference_error", error=str(exc))
            return "年龄参考辅助暂不可用；沿用原模型估计。"

    def _predict_with_optional_features(self, patches):
        """辅助特征失败时回退原年龄接口，不能使年龄和误差一起失效。"""
        if not self._feature_error:
            try:
                ages, probs, labels, features = self._backend.predict_probs_with_features(patches)
                if (len(ages) != len(patches) or len(probs) != len(patches)
                        or features is None or len(features) != len(patches)):
                    raise RuntimeError("参考特征接口返回的结果数量不匹配")
                return ages, probs, labels, features
            except Exception as exc:  # noqa: BLE001
                self._feature_error = f"{type(exc).__name__}: {exc}"
                sync_log("review_feature_fallback", error=self._feature_error)
        ages, probs, labels = self._backend.predict_probs(patches)
        return ages, probs, labels, None

    def _emit_failure(self, uid: int, error: str) -> None:
        item = self._items.get(uid)
        if item is not None:
            self.computed.emit(uid, {
                "status": "失败", "error": error, "pred": None,
                "probs": [], "labels": [], "box": item.box,
                "box_signature": item.box_signature,
            })

    # ---------------------------------------------------------------- #
    def push(self, items: list[ReviewItem]) -> None:
        with self._lock:
            for it in items:
                self._items[it.uid] = it
                self._queue.append(it.uid)
            queued = len(self._queue) + len(self._prio)
        sync_log("review_preload_queued", added=len(items), queued=queued)

    def clear_pending(self) -> None:
        """列表清空时丢弃旧目录的排队任务；已开始的一批允许自然结束。"""
        with self._lock:
            discarded = len(self._queue) + len(self._prio)
            self._queue.clear()
            self._prio.clear()
            self._items.clear()
        sync_log("review_preload_cleared", discarded=discarded)

    def prioritize(self, uid: int) -> None:
        """把 uid 提到最前面。

        注意：不能只留一个"优先目标"——用户翻页比推理快时，被顶掉的那张会既不在
        优先位也不在主队列里，导致永远不算（这就是"只有当前这张有误差"的原因）。
        这里用一个小优先队列，超出的退回主队列，保证每张最终都会算。
        """
        with self._lock:
            if uid not in self._items:
                return
            try:
                self._queue.remove(uid)
            except ValueError:
                pass
            try:
                self._prio.remove(uid)
            except ValueError:
                pass
            while len(self._prio) >= 2:            # 只保留最近 2 个优先目标
                # 退到队尾，不能让频繁翻页产生的旧优先目标挤占未浏览图片。
                self._queue.append(self._prio.popleft())
            self._prio.append(uid)

    def stop(self) -> None:
        # 推理或加载模型可能持续数秒；由关闭流程异步等待线程结束。
        self._stop = True

    # ---------------------------------------------------------------- #
    def _next_batch(self, size: int) -> list[int]:
        """优先目标搭配一张后台图片，连续翻页也要推进整目录预计算。"""
        with self._lock:
            if self._prio:
                batch = [self._prio.popleft()]
                if size > 1 and self._queue:
                    batch.append(self._queue.popleft())
                return batch
            n = min(size, len(self._queue))
            return [self._queue.popleft() for _ in range(n)]

    def run(self) -> None:
        self.state_changed.emit("loading", "正在加载 ViT 年龄模型…")
        sync_log("review_model_loading")
        try:
            from ..core.pipeline import MODEL_VIT, load_backend

            self._backend = load_backend(MODEL_VIT)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            sync_log("review_model_failed", error=self.load_error)
        if self._backend is not None:
            try:
                from ..core.detector import FaceDetector

                self._detector = FaceDetector()
            except Exception as exc:  # noqa: BLE001
                self._detector_error = f"{type(exc).__name__}: {exc}"
                sync_log("review_detector_failed", error=self._detector_error)
        self.ready = True
        self.state_changed.emit("failed" if self.load_error else "ready", self.load_error)
        if not self.load_error:
            sync_log("review_model_ready", detector_ready=self._detector is not None)

        while not self._stop:
            uids = self._next_batch(BATCH)
            if not uids:
                self.msleep(30)
                continue
            if self._backend is None:
                for uid in uids:
                    self._emit_failure(uid, f"年龄模型加载失败：{self.load_error}")
                continue
            t0 = time.monotonic()
            with self._lock:
                queued = len(self._queue) + len(self._prio)
            sync_log("review_preload_batch_started", count=len(uids), queued=queued)
            try:
                self._compute_batch(uids)
            except Exception as exc:  # noqa: BLE001
                error = f"后台计算异常：{type(exc).__name__}: {exc}"
                sync_log("review_batch_failed", error=error, count=len(uids))
                self.state_changed.emit("error", error)
                for uid in uids:
                    self._emit_failure(uid, error)
            finally:
                sync_log("review_preload_batch_finished", count=len(uids),
                         elapsed_ms=round((time.monotonic() - t0) * 1000))

    def _compute_batch(self, uids: list[int]) -> None:  # noqa: C901
        t0 = time.time()

        # ---- 1) 逐个读图 + 取景（批量前向，比一张一张快 3~5 倍） ----
        prepared: list[tuple[int, ReviewItem, tuple, list]] = []
        extras: dict[int, dict] = {}
        for uid in uids:
            item = self._items.get(uid)
            if item is None:
                continue
            out: dict = {"status": "完成", "error": "", "pred": None,
                         "probs": [], "labels": [], "box": item.box,
                         "box_signature": item.box_signature}
            try:
                img = imread_bgr(item.image_path)
                if img is None:
                    out.update(status="失败", error="图片读取失败")
                    self.computed.emit(uid, out)
                    continue
                box = item.box
                if not box or box[2] <= 0 or box[3] <= 0:
                    if self._detector is None and self._detector_error:
                        raise RuntimeError(f"图片没有有效人脸框，且检测模型加载失败：{self._detector_error}")
                    faces = self._detector.detect(img) if self._detector else []
                    if not faces:
                        out.update(status="无人脸", error="未检测到人脸")
                        self.computed.emit(uid, out)
                        continue
                    f0 = faces[0]
                    box = (f0.x, f0.y, f0.w, f0.h) if hasattr(f0, "x") else tuple(f0)
                out["box"] = box
                if self._backend is None:
                    out.update(status="失败", error=self.load_error)
                    self.computed.emit(uid, out)
                    continue
                views = self._backend.prepare_views(img, box)
                if not views:
                    raise RuntimeError("模型输入取景为空")
                extras[uid] = out
                prepared.append((uid, item, box, [p for _, p in views]))
            except Exception as e:  # noqa: BLE001
                out.update(status="失败", error=f"{type(e).__name__}: {e}")
                self.computed.emit(uid, out)

        # ---- 2) 一次前向算完这一批 ----
        done_map: dict[int, dict] = {}
        for uid, item, box, patches in prepared:
            done_map[uid] = dict(extras[uid], box=box)
        if prepared:
            all_patches = [p for _, _, _, ps in prepared for p in ps]
            try:
                ages, probs, labels, features = self._predict_with_optional_features(all_patches)
                off = 0
                for uid, item, box, ps in prepared:
                    n = len(ps)
                    a = ages[off:off + n]
                    pr = probs[off:off + n].mean(axis=0)
                    feature = features[off:off + n] if features is not None else None
                    off += n
                    d = done_map[uid]
                    d["pred"] = float(sum(a) / n)
                    d["probs"] = [float(x) for x in pr]
                    d["labels"] = labels
                    d["reference_hint"] = self._reference_output(d["pred"], feature)
            except Exception as exc:  # noqa: BLE001
                # 批量失败就退回逐张，保证不卡住
                sync_log("review_batch_fallback", error=f"{type(exc).__name__}: {exc}")
                for uid, item, box, ps in prepared:
                    try:
                        ages, probs, labels, features = self._predict_with_optional_features(ps)
                        done_map[uid].update(
                            pred=float(sum(ages) / len(ages)),
                            probs=[float(x) for x in probs.mean(axis=0)],
                            labels=labels)
                        done_map[uid]["reference_hint"] = self._reference_output(done_map[uid]["pred"], features)
                    except Exception as e2:  # noqa: BLE001
                        error = f"{type(e2).__name__}: {e2}"
                        done_map[uid].update(status="失败", error=error, pred=None,
                                             probs=[], labels=[], reference_hint="")
                        sync_log("review_item_failed", uid=uid, error=error)
        for uid, d in done_map.items():
            d["ms"] = int((time.time() - t0) * 1000 / max(1, len(done_map)))
            self.computed.emit(uid, d)


# ---------------------------------------------------------------------- #
# 主窗口
# ---------------------------------------------------------------------- #
class ExtractWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, paths: list[str], destination: str, parent=None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.destination = destination

    def run(self) -> None:
        try:
            self.completed.emit(copy_error_images(self.paths, self.destination))
        except Exception as exc:
            self.failed.emit(str(exc))


class ReviewWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} · 年龄复查助手（改进版源码）· 修订 {SYNC_REVISION}")
        self.resize(1160, 760)
        self.setMinimumSize(560, 380)
        self.setAcceptDrops(True)

        self.items: list[ReviewItem] = []
        self.by_uid: dict[int, ReviewItem] = {}
        self._item_indices: dict[int, int] = {}
        self._items_by_path: dict[str, list[ReviewItem]] = {}
        self._rows_by_uid: dict[int, QTreeWidgetItem] = {}
        self._uid = 0
        self._filter = "all"
        self._saved_filter = "all"
        self._extra_filter = "all"
        self._computed = 0
        self._compute_notice = ""
        self._t_start = time.time()
        self._cur_uid: int | None = None
        self._img_cache: dict[str, object] = {}
        self._mtimes: dict[str, float] = {}
        self._watcher: KeyWatcher | None = None
        self._programmatic = False   # True 表示这次选中是程序驱动的，不要反向发按键
        self._prev_uid: int | None = None   # 本次选中之前的 uid
        self._old_pos_uid: int | None = None  # 我们认定的"原工具当前显示的那张"
        self._sync_worker: LegacySyncWorker | None = None
        self._pending_sync: int | None = None
        self._sync_blocked = False
        self._sync_returns_to_new = True
        self._retiring_sync_workers: list[LegacySyncWorker] = []
        self._retiring_watchers: list[KeyWatcher] = []
        self._sync_inflight = None  # 已停线程中仍未确认的一步，窗口运行期间保留。
        self._sync_restart_when_stopped = False
        self._closing = False
        self._model_stop_requested = False
        self._last_backup: tuple[str, dict[str, str], dict[str, str]] | None = None
        self._export_worker: ExtractWorker | None = None
        self._base_qss = QApplication.instance().styleSheet()
        self._data_width = 340

        self._build_ui()

        # 后台结果可能成批到达，只合并进度统计；每行和筛选仍立即更新。
        self._compute_status_timer = QTimer(self)
        self._compute_status_timer.setSingleShot(True)
        self._compute_status_timer.setInterval(150)
        self._compute_status_timer.timeout.connect(self._refresh_compute_status)
        self._sync_restart_timer = QTimer(self)
        self._sync_restart_timer.setSingleShot(True)
        self._sync_restart_timer.setInterval(150)
        self._sync_restart_timer.timeout.connect(self._on_retired_sync_finished)

        # 监听目录里 json 的变化：原工具一保存，这里就能看到新值并自动对齐
        self._fsw = QFileSystemWatcher(self)
        self._fsw.directoryChanged.connect(self._on_dir_changed)

        self.worker = ReviewWorker(self)
        self.worker.computed.connect(self._on_computed)
        self.worker.state_changed.connect(self._on_worker_state)
        self.worker.start()

        # 目录事件未必报告已有 JSON 的内容变化；分批补查，当前预览优先。
        self._json_poll_cursor = 0
        self._json_poll_timer = QTimer(self)
        self._json_poll_timer.setInterval(800)
        self._json_poll_timer.timeout.connect(self._poll_json_changes)
        self._json_poll_timer.start()

    # ---------------------------------------------------------------- #
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(8)

        # 顶部工具条
        bar = QHBoxLayout()
        toolbar = QWidget()
        toolbar.setLayout(bar)
        bar.setContentsMargins(2, 2, 2, 2)
        filter_bar = QHBoxLayout()
        filter_toolbar = QWidget()
        filter_toolbar.setLayout(filter_bar)
        filter_bar.setContentsMargins(2, 2, 2, 2)
        self.btn_dir = QPushButton("添加文件夹…")
        self.btn_files = QPushButton("添加图片…")
        self.btn_clear = QPushButton("清空")
        self.btn_save = QPushButton("一键保存")
        self.btn_save.setObjectName("primary")
        self.btn_save.setToolTip(
            "把所有【未保存】的图片按 Cake 年龄写入 anno_age。\n"
            "缺失性别采用已有 Cake 候选；无候选时采用用户设定默认男并标明来源。\n"
            "已有 anno 的一张都不动；写入前自动备份，可撤销。")
        self.btn_undo = QPushButton("撤销")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("撤销上一次「一键保存」，从备份恢复原文件")
        self.chk_rec = QCheckBox("含子目录")
        self.chk_rec.setChecked(True)
        self.chk_sync = QCheckBox("A/D 双向同步")
        self.chk_sync.setChecked(True)
        self.chk_sync.setToolTip(
            "勾选后：在新版或旧版按 A/D，都按新版当前筛选列表翻页，两边显示同一张。\n"
            "例如筛选后只剩原编号 201、999，D 从 201 跳到 999，A 返回 201。\n"
            "新版的左右键也会同步旧版；旧版按 S 保存仍照常生效。")
        self.chk_rev = QCheckBox("点击同步旧版")
        self.chk_rev.setChecked(True)
        self.chk_rev.setToolTip(
            "默认开启。单击或双击筛选列表中的图片，旧版就跟到同一张；以新版选中的位置为准。\n"
            "点击同步完成后焦点返回新版；编号定位也会同步旧版。\n"
            "两边 A/D 是否按当前筛选列表同步翻页，由「A/D 双向同步」控制。")
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["全部误差", "误差 ≤2（含正负）", "误差 3–4（含正负）",
                                  "误差 >5（橙/红）", "误差 >10（红）", "自定义误差"])
        self.cmb_saved = QComboBox()
        self.cmb_saved.addItems(["全部保存状态", "未保存", "已保存"])
        self.cmb_extra = QComboBox()
        self.cmb_extra.addItems(["全部图片", "未看过", "性别字段冲突", "有疑问"])
        self.cmb_extra.setToolTip(
            "性别字段冲突：对照已有标注、Cake 候选和可识别的文件名身份证性别位。\n"
            "缺失或默认值本身不算冲突；结果仅供人工核对，不代表已确认错误。\n"
            "未从照片验证性别；没有数据冲突不代表全部正确。")
        self.cmb_err_compare = QComboBox()
        self.cmb_err_compare.addItems(["绝对误差 ≥", "绝对误差 ≤"])
        self.spin_err_abs = QSpinBox()
        self.spin_err_abs.setRange(0, 1000)
        self.spin_err_abs.setValue(2)
        self.spin_err_abs.setKeyboardTracking(False)
        self._custom_error_edited = False
        self.spin_err_abs.setToolTip("输入数字并回车或离开输入框，即自动启用自定义绝对误差筛选；含等号，允许 0。")
        self.lbl_ro = QLabel("只读：不修改任何图片与 json")
        self.lbl_ro.setObjectName("hint")
        self.btn_about = QPushButton("口径说明")
        for w in (self.btn_dir, self.btn_files, self.btn_clear, self.btn_save, self.btn_undo):
            bar.addWidget(w)
        bar.addWidget(self.chk_rec)
        bar.addWidget(self.chk_sync)
        bar.addWidget(self.chk_rev)
        filter_bar.addWidget(QLabel("筛选："))
        filter_bar.addWidget(self.cmb_filter)
        filter_bar.addWidget(self.cmb_err_compare)
        filter_bar.addWidget(self.spin_err_abs)
        filter_bar.addWidget(self.cmb_saved)
        filter_bar.addWidget(self.cmb_extra)
        filter_bar.addSpacing(10)
        filter_bar.addWidget(QLabel("跳到第"))
        self.spin_go = QSpinBox()
        self.spin_go.setRange(1, 1)
        self.spin_go.setFixedWidth(84)
        self.spin_go.setKeyboardTracking(False)
        self.spin_go.setToolTip(
            "按全部图片的原始编号定位；目标被筛掉时仍显示预览，保留筛选。\n"
            "开启点击同步时，旧版也定位到同一张图片。输入后按回车或点定位。")
        self.btn_go = QPushButton("定位")
        self.btn_go.setFixedWidth(52)
        filter_bar.addWidget(self.spin_go)
        filter_bar.addWidget(self.btn_go)
        filter_bar.addWidget(QLabel("张"))
        self.btn_align = QPushButton("对齐原工具")
        self.btn_align.setFixedWidth(84)
        self.btn_align.setToolTip(
            "只告诉程序「原工具现在停在第几张」，不移动本窗口。\n"
            "用于两边位置不一致时：例如本窗口在第 1 张，原工具在第 50 张，\n"
            "点这里填 50，之后点本窗口第 1 张就会让它后退 49 张。")
        filter_bar.addWidget(self.btn_align)
        filter_bar.addStretch(1)
        self.lbl_pos = QLabel("第 - / - 张")
        self.lbl_pos.setObjectName("hint")
        filter_bar.addWidget(self.lbl_pos)
        bar.addStretch(1)
        bar.addWidget(self.btn_about)
        # 长工具条可横向滚动，不再把整窗最小宽度撑到一千多像素。
        self.toolbar_scroll = QScrollArea()
        self.toolbar_scroll.setWidgetResizable(True)
        self.toolbar_scroll.setFrameShape(QFrame.NoFrame)
        self.toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.toolbar_scroll.setWidget(toolbar)
        self.toolbar_scroll.setMinimumWidth(0)
        self.toolbar_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        outer.addWidget(self.toolbar_scroll)
        self.filter_scroll = QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setFrameShape(QFrame.NoFrame)
        self.filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.filter_scroll.setWidget(filter_toolbar)
        self.filter_scroll.setMinimumWidth(0)
        self.filter_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        outer.addWidget(self.filter_scroll)

        tools = QHBoxLayout()
        tools_widget = QWidget()
        tools_widget.setLayout(tools)
        tools.setContentsMargins(2, 2, 2, 2)
        self.btn_extract = QPushButton("提取误差 3–4 的图…")
        self.btn_extract.setToolTip(
            "提取已完成计算、取整误差绝对值为 3 或 4 的图片，包含正负误差。\n"
            "基准依次采用已标注、Cake、原始、TS 中第一个有效年龄。\n"
            "从全部已加载图片选择，不受当前筛选影响；一图多脸只复制一次。\n"
            "仅复制图片到你选择的位置下的新文件夹，不复制或修改 json。")
        tools.addWidget(self.btn_extract)
        self.btn_save_small = QPushButton("保存误差 ≤2")
        self.btn_save_small.setToolTip(
            "一键接受默认值：保存全部已加载、计算完成且 −2≤取整误差≤2 的未保存标注。\n"
            "不受当前筛选影响；已有标注不改动，其他误差和未计算条目不保存。\n"
            "仅保存有 Cake 年龄的条目；缺失性别采用已有候选或默认男并标明来源。自动备份，可撤销。")
        tools.addWidget(self.btn_save_small)
        self.btn_gender = QPushButton("补全缺失性别")
        self.btn_gender.setToolTip(
            "只补全缺失的 anno_gender，已有人工性别不改动，年龄不改动。\n"
            "优先采用已有 Cake 性别；也缺失时采用用户设定默认男，标明来源。\n"
            "不从图片推断性别；点击此按钮才写入。\n"
            "自动备份，可撤销。")
        tools.addWidget(self.btn_gender)
        self.btn_retry_sync = QPushButton("重试同步")
        self.btn_retry_sync.setToolTip("重新核对当前预览，让旧版定位到这张图片")
        tools.addWidget(self.btn_retry_sync)
        tools.addStretch(1)
        self.cmb_scale = QComboBox()
        for percent in (75, 85, 100, 115):
            self.cmb_scale.addItem(f"界面 {percent}%", percent)
        self.cmb_scale.setCurrentIndex(2)
        self.cmb_scale.setToolTip("缩放界面的文字和控件；图片仍自动适应预览区域")
        tools.addWidget(self.cmb_scale)
        self.btn_data = QPushButton("收起数据")
        self.btn_data.setCheckable(True)
        self.btn_data.setChecked(True)
        self.tools_scroll = QScrollArea()
        self.tools_scroll.setWidgetResizable(True)
        self.tools_scroll.setFrameShape(QFrame.NoFrame)
        self.tools_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tools_scroll.setWidget(tools_widget)
        self.tools_scroll.setMinimumWidth(0)
        self.tools_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        control_row = QHBoxLayout()
        control_row.addWidget(self.tools_scroll, 1)
        control_row.addWidget(self.btn_data)
        outer.addLayout(control_row)

        split = QSplitter(Qt.Horizontal)
        self._split = split
        outer.addWidget(split, 1)

        # 左：列表
        left = QWidget()
        left.setMinimumWidth(180)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        self.lbl_list = QLabel("筛选结果 0 项 / 全部 0 项")
        self.lbl_list.setWordWrap(True)
        self.lbl_list.setToolTip("数量按列表条目统计；一图多脸时可有多项。原始编号不随筛选重新编号。")
        self.lbl_list.setObjectName("hint")
        lv.addWidget(self.lbl_list)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["原始编号", "文件", "模型", "误差", "保存"])
        self.tree.headerItem().setToolTip(0, "全部数据中的原始编号，不是筛选结果的数量；筛选后不重新编号。")
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setAlternatingRowColors(False)
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(0, 68)
        self.tree.setColumnWidth(2, 58)
        self.tree.setColumnWidth(3, 40)
        self.tree.setColumnWidth(4, 40)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        # 中：预览
        self.preview = FacePreview()
        self.preview.setMinimumSize(160, 140)
        split.addWidget(self.preview)

        # 右：结论面板
        right = QWidget()
        self._data_content = right
        right.setMinimumWidth(0)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 4, 4, 4)
        rv.setSpacing(8)

        cap = QLabel("模型估计")
        cap.setObjectName("hint")
        rv.addWidget(cap)
        self.lbl_age = QLabel("—")
        f = QFont("Microsoft YaHei UI", 26)
        f.setBold(True)
        self.lbl_age.setFont(f)
        self.lbl_age.setStyleSheet("color:#8fc1ff;")
        rv.addWidget(self.lbl_age)
        self.lbl_sub = QLabel("ViT-Age（FairFace 微调）· 原始输出未校准 · 实测平均误差约 2.3 岁")
        self.lbl_sub.setObjectName("hint")
        self.lbl_sub.setWordWrap(True)
        rv.addWidget(self.lbl_sub)
        self.lbl_reference = QLabel("")
        self.lbl_reference.setWordWrap(True)
        self.lbl_reference.setObjectName("hint")
        self.lbl_reference.setToolTip(
            "原模型年龄为主，参考人像特征为辅，联合提示 60/70/80 左右档位。\n"
            "参考相似度不是年龄概率，未在本批数据校准；误差与筛选仍使用原模型年龄。")
        rv.addWidget(self.lbl_reference)
        self.btn_reference = QPushButton("查看年龄参考图")
        rv.addWidget(self.btn_reference)

        self.bars = ProbBars()
        rv.addWidget(self.bars)

        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setMinimumHeight(38)
        rv.addWidget(self.lbl_verdict)

        self.lbl_err = QLabel("")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet("font-size:13px;")
        rv.addWidget(self.lbl_err)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#33333c;")
        rv.addWidget(line)

        cap2 = QLabel("json 里的候选（只读展示）")
        cap2.setObjectName("hint")
        rv.addWidget(cap2)
        self.cand_rows: list[QLabel] = []
        for _ in range(3):
            lb = QLabel("")
            lb.setTextFormat(Qt.RichText)
            lb.setMinimumHeight(26)
            lb.setWordWrap(True)
            lb.setStyleSheet("padding:3px 6px; border-radius:4px;")
            rv.addWidget(lb)
            self.cand_rows.append(lb)

        self.lbl_gender = QLabel("")
        self.lbl_gender.setObjectName("hint")
        self.lbl_gender.setWordWrap(True)
        rv.addWidget(self.lbl_gender)
        self.lbl_cand_hint = QLabel("候选之间相差不到 4 岁时，模型的排序不可靠，请直接看图判断")
        self.lbl_cand_hint.setObjectName("hint")
        self.lbl_cand_hint.setWordWrap(True)
        rv.addWidget(self.lbl_cand_hint)
        rv.addStretch(1)

        tip = QLabel("快捷键：← → / A D 切换　空格 标记已看并下一张　1/2/3 记录你的选择（仅本机内存）")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        rv.addWidget(tip)
        self._data_panel = QScrollArea()
        self._data_panel.setWidgetResizable(True)
        self._data_panel.setWidget(right)
        self._data_panel.setMinimumWidth(0)
        self._data_panel.setMaximumWidth(420)
        self._data_panel.setFrameShape(QFrame.NoFrame)
        split.addWidget(self._data_panel)
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        split.setCollapsible(2, True)
        split.setStretchFactor(1, 1)
        split.setSizes([310, 500, 340])

        # 底部
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.lbl_status = QLabel("把文件夹或图片拖进窗口，或点左上角按钮")
        self.lbl_status.setObjectName("hint")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setMinimumWidth(0)
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_legend = QLabel(
            '误差：<span style="color:#00E5A0">≤2 绿</span> · '
            '<span style="color:#FFD166">≤5 黄</span> · '
            '<span style="color:#FF9F43">≤10 橙</span> · '
            '<span style="color:#FF5C5C">&gt;10 红</span>'
            '　（依次相对已标注、Cake、原始或 TS 年龄）')
        self.lbl_legend.setObjectName("hint")
        self.lbl_legend.setWordWrap(True)
        self.lbl_legend.setMinimumWidth(0)
        self.lbl_legend.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.progress.setMaximumWidth(130)
        bottom.addWidget(self.lbl_status, 1)
        bottom.addWidget(self.lbl_legend, 1)
        bottom.addWidget(self.progress, 0)
        outer.addLayout(bottom)
        self.lbl_compute = QLabel("自动计算：正在加载 ViT 年龄模型…")
        self.lbl_compute.setObjectName("hint")
        self.lbl_compute.setWordWrap(True)
        self.lbl_compute.setMinimumWidth(0)
        self.lbl_compute.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        outer.addWidget(self.lbl_compute)
        self.lbl_sync = QLabel("同步：等待加载图片")
        self.lbl_sync.setObjectName("hint")
        self.lbl_sync.setWordWrap(True)
        self.lbl_sync.setMinimumWidth(0)
        self.lbl_sync.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        outer.addWidget(self.lbl_sync)

        # 信号
        self.btn_dir.clicked.connect(self._pick_dir)
        self.btn_files.clicked.connect(self._pick_files)
        self.btn_clear.clicked.connect(self._clear)
        self.cmb_filter.currentIndexChanged.connect(self._on_filter)
        self.cmb_saved.currentIndexChanged.connect(self._on_filter)
        self.cmb_extra.currentIndexChanged.connect(self._on_filter)
        self.cmb_err_compare.activated.connect(self._on_custom_error)
        self.spin_err_abs.valueChanged.connect(self._on_custom_error)
        self.spin_err_abs.lineEdit().textEdited.connect(
            lambda _text: setattr(self, "_custom_error_edited", True))
        self.spin_err_abs.editingFinished.connect(self._commit_custom_error)
        self.tree.currentItemChanged.connect(self._on_select)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.btn_about.clicked.connect(self._show_about)
        self.btn_save.clicked.connect(self._save_defaults)
        self.btn_undo.clicked.connect(self._undo_save)
        self.chk_sync.toggled.connect(self._on_sync_toggled)
        self.chk_rev.toggled.connect(self._on_sync_toggled)
        self.btn_go.clicked.connect(self._go_to_value)
        self.btn_align.clicked.connect(self._align_old_pos)
        self.btn_extract.clicked.connect(self._extract_error_images)
        self.btn_save_small.clicked.connect(lambda: self._save_defaults(small_error_only=True))
        self.btn_gender.clicked.connect(lambda: self._save_defaults(gender_only=True))
        self.btn_reference.clicked.connect(self._show_reference_image)
        self.btn_retry_sync.clicked.connect(self._retry_sync)
        self.btn_data.toggled.connect(self._set_data_visible)
        self._split.splitterMoved.connect(self._on_splitter_moved)
        self.cmb_scale.currentIndexChanged.connect(self._set_interface_scale)
        self.spin_go.lineEdit().returnPressed.connect(self._go_to_value)

        for key, slot in (("Left", lambda: self._step(-1, sync_legacy=True)),
                          ("A", lambda: self._step(-1, sync_legacy=True)),
                          ("Right", lambda: self._step(1, sync_legacy=True)),
                          ("D", lambda: self._step(1, sync_legacy=True)),
                          ("Space", self._mark_seen_next)):
            self._bind(key, slot)
        for i in range(3):
            self._bind(str(i + 1), lambda idx=i: self._mark_pick(idx))
        self._set_interface_scale()

    def _set_interface_scale(self, _index: int = 0) -> None:
        scale = self.cmb_scale.currentData() / 100
        app = QApplication.instance()
        app.setStyleSheet(re.sub(
            r"(\d+(?:\.\d+)?)px",
            lambda match: f"{max(1, round(float(match[1]) * scale))}px",
            self._base_qss))
        app.setFont(QFont("Microsoft YaHei UI", max(7, round(9 * scale))))
        font = QFont("Microsoft YaHei UI", max(14, round(26 * scale)))
        font.setBold(True)
        self.lbl_age.setFont(font)
        self.bars.scale = scale
        self.bars.setMinimumHeight(round(150 * scale))
        self.bars.setMaximumHeight(round(160 * scale))
        self.bars.update()
        self.lbl_err.setStyleSheet(f"font-size:{round(13 * scale)}px;")
        self.lbl_verdict.setMinimumHeight(round(38 * scale))
        for label in self.cand_rows:
            label.setMinimumHeight(round(26 * scale))
        self.spin_go.setFixedWidth(round(84 * scale))
        self.btn_go.setFixedWidth(round(52 * scale))
        self.btn_align.setFixedWidth(round(84 * scale))
        for column, width in ((0, 68), (2, 58), (3, 40), (4, 40)):
            self.tree.setColumnWidth(column, round(width * scale))
        for scroll in (self.toolbar_scroll, self.filter_scroll, self.tools_scroll):
            height = scroll.widget().sizeHint().height()
            scroll.setFixedHeight(height + scroll.horizontalScrollBar().sizeHint().height() + 6)

    def _set_data_visible(self, visible: bool) -> None:
        sizes = self._split.sizes()
        if not visible and sizes[2] > 0:
            self._data_width = sizes[2]
        self._data_panel.setVisible(visible)
        if visible:
            self._split.setSizes([sizes[0], max(160, sizes[1] - self._data_width),
                                  self._data_width])
        self.btn_data.setText("收起数据" if visible else "展开数据")

    def _on_splitter_moved(self, _position: int, _index: int) -> None:
        visible = self._data_panel.isVisible() and self._split.sizes()[2] > 0
        blocked = self.btn_data.blockSignals(True)
        try:
            self.btn_data.setChecked(visible)
            self.btn_data.setText("收起数据" if visible else "展开数据")
        finally:
            self.btn_data.blockSignals(blocked)

    def _extract_error_images(self) -> None:
        if self._export_worker is not None:
            return
        # 先完成计算再提取，避免把尚未有预测值的图片悄悄漏掉。
        if any(it.status == "待计算" for it in self.items):
            QMessageBox.information(self, "提取图片", "请等后台计算完成后再提取，避免漏图。")
            return
        paths = error_images(self.items)
        if not paths:
            QMessageBox.information(self, "提取图片", "没有取整误差绝对值为 3 或 4 的图片。")
            return
        destination = QFileDialog.getExistingDirectory(self, f"选择提取位置（共 {len(paths)} 张）")
        if not destination:
            return
        worker = self._export_worker = ExtractWorker(paths, destination, self)
        worker.completed.connect(self._on_extract_completed)
        worker.failed.connect(self._on_extract_failed)
        worker.finished.connect(self._on_extract_finished)
        worker.finished.connect(worker.deleteLater)
        self.btn_extract.setEnabled(False)
        self.lbl_status.setText(f"正在提取 {len(paths)} 张图片…")
        worker.start()

    def _on_extract_completed(self, result) -> None:
        message = f"已提取 {result.copied} 张图片到：\n{result.folder}"
        uncomputed = sum(it.status != "完成" for it in self.items)
        if uncomputed:
            message += f"\n另有 {uncomputed} 条计算失败或无人脸，未纳入提取。"
        if result.failures:
            message += f"\n复制失败 {len(result.failures)} 张："
            message += "\n" + "\n".join(f"{Path(p).name}：{error}" for p, error in result.failures[:3])
        self.lbl_status.setText(message.replace("\n", " "))
        QMessageBox.information(self, "提取结果", message)

    def _on_extract_failed(self, message: str) -> None:
        self.lbl_status.setText(f"提取失败：{message}")
        QMessageBox.warning(self, "提取失败", message)

    def _on_extract_finished(self) -> None:
        self._export_worker = None
        self.btn_extract.setEnabled(True)

    def _retry_sync(self) -> None:
        current = self._current()
        if current is not None:
            self._reverse_sync(self._old_pos_uid, current.uid, force=True)

    def _show_reference_image(self) -> None:
        from ..core.senior_reference import REFERENCE_IMAGE
        dialog = QDialog(self)
        dialog.setWindowTitle("年龄辅助参考 · 60 / 70 / 80 岁")
        dialog.resize(800, 650)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        label = QLabel()
        label.setPixmap(QPixmap(str(REFERENCE_IMAGE)))
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.exec()

    def _bind(self, key: str, slot) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(slot)

    def _show_about(self) -> None:
        QMessageBox.information(self, "模型口径说明", (
            "以下数字全部在本机、用你已标注的 1575 张人脸实测，未做任何挑选：\n\n"
            "• 与人工标注的平均误差 2.3 岁，中位 1.7 岁；误差 ≤5 岁占 90.4%\n"
            "• 13~19 岁段平均误差 2.2 岁；20~29 岁段 4.2 岁\n"
            "• 20 岁分界（十几 vs 二十几）：平衡准确只有 55~59%，"
            "二十几的召回仅 24%\n"
            "　→ 模型说“十几”时，并不能说明他真的是十几\n"
            "• 曾用你的标注做保序回归矫正：平均误差降到 1.6 岁，"
            "但二十几召回归零，故不采用（不把数字做漂亮）\n\n"
            "所以本工具只做两件事：\n"
            "1) 给一个参考年龄，帮你看“大概多大”\n"
            "2) 标出模型与候选差得离谱的样本，提示复核\n\n"
            "不做的事：不替你判断十几/二十几，不写任何文件。"
        ))

    # ---------------------------------------------------------------- #
    # 输入
    # ---------------------------------------------------------------- #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_paths(paths)

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            self.add_paths([d])

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.webp *.bmp);;所有文件 (*)")
        if files:
            self.add_paths(files)

    def add_paths(self, paths: list[str]) -> None:
        images = collect_images(paths, recursive=self.chk_rec.isChecked())
        if not images:
            self.lbl_status.setText("没有找到图片文件")
            return
        known = {os.path.normcase(os.path.realpath(it.image_path)) for it in self.items}
        new_images = [p for p in images if os.path.normcase(os.path.realpath(p)) not in known]
        if not new_images:
            self.lbl_status.setText("这些图片已经在列表里了")
            return
        new_items = build_items(new_images, start_uid=self._uid)
        self._uid += len(new_items)
        start_index = len(self.items)
        self.items.extend(new_items)
        for index, it in enumerate(new_items, start_index):
            self.by_uid[it.uid] = it
            self._item_indices[it.uid] = index
            self._items_by_path.setdefault(it.image_path, []).append(it)
        self.worker.push(new_items)
        self._watch_dirs(new_items)
        if self.chk_sync.isChecked() or self.chk_rev.isChecked():
            self._start_sync()
        self._refresh_list(select_uid=new_items[0].uid)
        self.progress.setRange(0, len(self.items))
        self._refresh_compute_status()
        self.lbl_status.setText(f"新增 {len(new_images)} 张（共 {len(self.items)} 张）")

    # ---------------------------------------------------------------- #
    # 一键保存（只写未保存的，写前备份，可撤销）
    # ---------------------------------------------------------------- #
    def _save_defaults(self, *, small_error_only: bool = False, gender_only: bool = False) -> None:
        self._cancel_reverse_sync()
        if gender_only:
            todo = [it for it in self.items if is_gender_save_candidate(it)]
        else:
            todo = [it for it in self.items
                    if it.anno is None and it.json_path and valid_age(it.cake)]
        if small_error_only:
            todo = [it for it in todo if it.status == "完成"
                    and it.err_i is not None and abs(it.err_i) <= 2]
        if not todo:
            message = ("没有缺失性别的标注。" if gender_only else
                       "没有已计算、未保存、有 Cake 年龄且误差在 −2 至 2 的标注。"
                       if small_error_only else "没有可保存的图片（未保存项需要有效 Cake 年龄）。")
            QMessageBox.information(self, "一键保存", message)
            return
        n_have = sum(1 for it in self.items if it.anno is not None)
        if not small_error_only and not gender_only:
            ans = QMessageBox.question(
                self, "一键保存",
                f"将把 {len(todo)} 张【未保存】图片的年龄写成 Cake 值。\n"
                "缺失性别沿用 Cake 候选；无候选时采用用户设定默认男并标明来源。\n"
                f"已有 anno 的 {n_have} 张不会被改动。\n\n"
                f"写入前自动备份到：\n{backup_root()}\n\n确定继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = save_defaults(todo, max_error=2 if small_error_only else None,
                                gender_only=gender_only)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "一键保存失败", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()

        # 自己写的文件要同步 mtime，否则目录监听会以为"原工具保存了"而跳图
        for jp in {it.json_path for it in todo if it.json_path}:
            try:
                self._mtimes[jp] = os.path.getmtime(jp)
            except OSError:
                pass
        for it in self.items:
            if it.json_path:
                self._reload_json(it)
        self._refresh_list()
        if res["mapping"]:
            self._last_backup = (res["backup_dir"], res["mapping"], res["written_hashes"])
            self.btn_undo.setEnabled(True)

        msg = f"已写入 {res['written']} 条标注。"
        if small_error_only:
            msg += "\n范围：−2 ≤ 取整误差 ≤ 2；写入 Cake 默认值。"
            skipped = len(todo) - res["written"]
            if skipped and not res["errors"]:
                msg += f"\n另有 {skipped} 条在写入前已保存或数据有变化，已跳过。"
        if gender_only:
            msg += "\n只补全缺失性别，未修改年龄或已有人工性别。"
            if res["skipped"]:
                msg += f"\n另有 {res['skipped']} 条标注文件不存在或不可用，已跳过。"
        if res["default_genders"]:
            msg += f"\n其中 {res['default_genders']} 条采用用户设定默认男，已标明来源。"
        if res["errors"]:
            msg += f"\n\n失败 {len(res['errors'])} 个文件（可能被原工具占用）：\n"
            msg += "\n".join(f"{Path(p).name}: {e}" for p, e in res["errors"][:5])
        msg += f"\n\n备份目录：\n{res['backup_dir']}"
        QMessageBox.information(self, "一键保存", msg)

    def _undo_save(self) -> None:
        if not self._last_backup:
            QMessageBox.information(self, "撤销", "没有可撤销的保存记录（重启后会清空）。")
            return
        bdir, mapping, hashes = self._last_backup
        ans = QMessageBox.question(
            self, "撤销上一次保存",
            f"将从备份恢复 {len(mapping)} 个 json 文件：\n{bdir}\n\n确定？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        res = restore_from_map(bdir, mapping, expected_hashes=hashes)
        touched = set(mapping.values())
        for it in self.items:
            if it.json_path in touched:
                self._reload_json(it)
                try:
                    self._mtimes[it.json_path] = os.path.getmtime(it.json_path)
                except OSError:
                    pass
        self._refresh_list()
        restored_paths = set(res["restored_paths"])
        remaining = {name: path for name, path in mapping.items() if path not in restored_paths}
        if not remaining:
            self._last_backup = None
            self.btn_undo.setEnabled(False)
        else:
            self._last_backup = (bdir, remaining, hashes)
        message = f"已恢复 {res['restored']} 个文件。"
        if res["errors"]:
            message += f"\n未恢复 {len(res['errors'])} 个文件，已保留撤销记录：\n"
            message += "\n".join(f"{Path(path).name}：{error}" for path, error in res["errors"][:5])
        QMessageBox.information(self, "撤销", message)

    def _clear(self) -> None:
        self._stop_sync()
        self.worker.clear_pending()
        self.items.clear()
        self.by_uid.clear()
        self._item_indices.clear()
        self._items_by_path.clear()
        self._mtimes.clear()
        try:
            dirs = self._fsw.directories()
            if dirs:
                self._fsw.removePaths(dirs)
        except Exception:  # noqa: BLE001
            pass
        self._computed = 0
        self._t_start = time.time()
        self._cur_uid = None
        self._old_pos_uid = None
        self._prev_uid = None
        self._pending_sync = None
        self._img_cache.clear()
        self._rows_by_uid.clear()
        self.tree.clear()
        self.preview.clear("拖入文件夹或图片开始")
        self._reset_panel()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._update_list_count()
        self.lbl_status.setText("已清空")
        self._refresh_compute_status()

    # ---------------------------------------------------------------- #
    # 列表
    # ---------------------------------------------------------------- #
    def _passes(self, it: ReviewItem) -> bool:
        if self._saved_filter == "noanno" and it.anno is not None:
            return False
        if self._saved_filter == "saved" and it.anno is None:
            return False
        if self._extra_filter == "unseen" and it.seen:
            return False
        if self._extra_filter == "sex_review" and not it.data_gender_conflict:
            return False
        if self._extra_filter == "issue" and not it.issues():
            return False
        if self._filter == "err2":
            return it.err_i is not None and abs(it.err_i) <= 2
        if self._filter == "err34":
            return it.err_i is not None and abs(it.err_i) in (3, 4)
        if self._filter == "custom":
            error = it.err_i
            if error is None:
                return False
            threshold = self.spin_err_abs.value()
            return (abs(error) >= threshold if self.cmb_err_compare.currentIndex() == 0
                    else abs(error) <= threshold)
        if self._filter == "err5":
            return it.err_i is not None and abs(it.err_i) > 5
        if self._filter == "err10":
            return it.err_i is not None and abs(it.err_i) > 10
        return True

    def _has_filter(self) -> bool:
        return any(value != "all" for value in (self._filter, self._saved_filter, self._extra_filter))

    def _update_list_count(self) -> None:
        count = self.tree.topLevelItemCount()
        text = f"筛选结果 {count} 项 / 全部 {len(self.items)} 项"
        if self._extra_filter == "sex_review":
            text += "\n已有字段存在冲突，需人工核对；不代表已确认错误"
        self.lbl_list.setText(text)

    def _refresh_list(self, select_uid: int | None = None) -> None:
        keep = select_uid if select_uid is not None else self._cur_uid
        self.tree.blockSignals(True)
        self._rows_by_uid.clear()
        self.tree.clear()
        for idx, it in enumerate(self.items):
            if not self._passes(it):
                continue
            tw = self._make_row(it, idx)
            self.tree.addTopLevelItem(tw)
            self._rows_by_uid[it.uid] = tw
            if it.uid == keep:
                self.tree.setCurrentItem(tw)
        self.tree.blockSignals(False)
        self._update_list_count()
        if (self._cur_uid is None and self.tree.currentItem() is None
                and self.tree.topLevelItemCount()):
            # 只在尚无预览时初始化高亮；筛选外的实际位置不能借用另一行的高亮。
            blocked = self.tree.blockSignals(True)
            try:
                self.tree.setCurrentItem(self.tree.topLevelItem(0))
            finally:
                self.tree.blockSignals(blocked)
        # 重建列表时信号被屏蔽，这里补一次选中回调，保证右侧面板和预览跟着更新。
        # 首次或显式选择时才更新预览，普通刷新保留筛选外的实际位置。
        cur = self.tree.currentItem()
        if cur is not None and (self._cur_uid is None or select_uid is not None):
            self._programmatic = True
            try:
                self._on_select(cur, None)
            finally:
                self._programmatic = False
        self._update_pos()

    def _make_row(self, it: ReviewItem, index: int) -> QTreeWidgetItem:
        row = QTreeWidgetItem([str(index + 1), it.name, self._age_text(it),
                              self._err_text(it), self._save_text(it)])
        row.setData(0, Qt.UserRole, it.uid)
        tip = self._row_tip(it)
        row.setToolTip(0, tip)
        row.setToolTip(1, tip)
        self._paint_row(row, it)
        return row

    def _age_text(self, it: ReviewItem) -> str:
        if it.status == "完成" and it.pred is not None:
            return f"{it.pred:.0f} 岁"
        return it.status if it.status != "待计算" else "…"

    def _err_text(self, it: ReviewItem) -> str:
        e = it.err_i
        if e is None:
            if it.status == "完成" and it.pred is not None and it.baseline is None:
                return "缺基准"
            return ""
        return f"{e:+d}" if e else "0"

    def _row_color(self, it: ReviewItem) -> QColor | None:
        """误差分级（按取整后的岁数）：≤2 绿 / ≤5 黄 / ≤10 橙 / >10 红。"""
        if it.status in ("失败", "无人脸"):
            return RED
        e = it.err_i
        if e is None:
            return None
        a = abs(e)
        if a <= 2:
            return OK
        if a <= 5:
            return YELLOW
        if a <= 10:
            return WARN
        return RED

    def _row_tip(self, it: ReviewItem) -> str:
        lines = [it.image_path, f"json: {it.json_path or '（无）'}"]
        b = it.baseline
        if it.anno is not None:
            g = GENDER_TEXT.get(it.anno_gender, "?")
            age = f"{float(it.anno):.0f} 岁" if valid_age(it.anno) else "年龄无效"
            lines.append(f"已保存：{age} · {g}")
        else:
            lines.append("尚未保存（json 里没有 anno）")
        if it.pred is not None and b is not None:
            lines.append(f"模型 {it.pred:.1f} 岁，误差 {it.pred - b[0]:+.1f} 岁（相对{b[1]}）")
        elif it.pred is not None:
            lines.append("年龄已计算；缺少已标注、Cake、原始和 TS 的有效年龄基准。")
        if it.issues():
            lines.append("疑问：" + "；".join(it.issues()))
        if it.gender_review_reasons:
            lines.append("性别待核对原因：" + "；".join(it.gender_review_reasons))
        return "\n".join(lines)

    def _save_text(self, it: ReviewItem) -> str:
        return "✓" if it.anno is not None else ""

    def _paint_row(self, tw: QTreeWidgetItem, it: ReviewItem) -> None:
        color = self._row_color(it)
        for c in range(4):
            tw.setForeground(c, color if color is not None else QColor("#e6e6e6"))
        # 保存列：已保存=绿勾，未保存=灰
        tw.setForeground(4, OK if it.anno is not None else QColor("#5a5a66"))

    def _on_custom_error(self, *_args) -> None:
        self._custom_error_edited = False
        custom_index = self.cmb_filter.findText("自定义误差")
        if self.cmb_filter.currentIndex() != custom_index:
            self.cmb_filter.setCurrentIndex(custom_index)
        else:
            self._on_filter(custom_index)

    def _commit_custom_error(self) -> None:
        if self._custom_error_edited:
            self._on_custom_error()

    def _on_filter(self, idx: int) -> None:
        self._cancel_reverse_sync()
        self._filter = ["all", "err2", "err34", "err5", "err10", "custom"][self.cmb_filter.currentIndex()]
        self._saved_filter = ["all", "noanno", "saved"][self.cmb_saved.currentIndex()]
        self._extra_filter = ["all", "unseen", "sex_review", "issue"][self.cmb_extra.currentIndex()]
        self._refresh_list()
        row = self.tree.currentItem()
        if row is None and self.tree.topLevelItemCount():
            row = self.tree.topLevelItem(0)
        if row is not None:
            uid = int(row.data(0, Qt.UserRole))
            if uid != self._cur_uid:
                self._show_item(self.by_uid[uid])
        else:
            self._cur_uid = None
            self.preview.clear("当前筛选下没有图片")
            self._reset_panel()
            self._update_pos()
            self.lbl_status.setText("当前筛选下没有图片")

    def _update_pos(self) -> None:
        total = len(self.items)
        index = self._index_of(self._cur_uid)
        idx = index if index is not None else -1
        txt = f"第 {idx + 1 if idx >= 0 else '-'} / {total or '-'} 张"
        if self._has_filter():
            txt += f"（筛选内共 {self.tree.topLevelItemCount()} 行）"
        self.lbl_pos.setText(txt)
        if total:
            self.spin_go.blockSignals(True)
            self.spin_go.setRange(1, total)
            if (idx >= 0 and not self.spin_go.hasFocus()
                    and not self.spin_go.lineEdit().hasFocus()):
                self.spin_go.setValue(idx + 1)
            self.spin_go.blockSignals(False)

    def _align_old_pos(self) -> None:
        """告诉程序"原工具现在停在第几张"，不移动本窗口。"""
        if not self.items:
            QMessageBox.information(self, "对齐原工具", "先添加图片。")
            return
        cur = 1
        if self._old_pos_uid is not None:
            cur = next((i + 1 for i, it in enumerate(self.items)
                        if it.uid == self._old_pos_uid), 1)
        n, ok = QInputDialog.getInt(
            self, "对齐原工具",
            f"原工具（年龄 / 性别属性标注）底部现在显示第几张？\n共 {len(self.items)} 张：",
            cur, 1, len(self.items), 1)
        if not ok:
            return
        self._old_pos_uid = self.items[n - 1].uid
        self.lbl_status.setText(
            f"已手动记录旧版第 {n} 张；下次同步仍会回读编号和文件名确认。")

    def _go_to_value(self) -> None:
        if not self.items:
            return
        self.spin_go.interpretText()
        n = max(1, min(len(self.items), int(self.spin_go.value())))
        target = self.items[n - 1]
        if not self.chk_rev.isChecked():
            self._cancel_reverse_sync()
        self._show_item(target)
        if self.chk_rev.isChecked():
            self._reverse_sync(self._old_pos_uid, target.uid)

    def _step(self, delta: int, *, sync_legacy: bool = False,
              restore_focus: bool = True) -> None:
        """按新版可见列表翻页；两边的 A/D 共用该顺序和定位入口。"""
        # 新目标马上替换旧目标时不先把焦点抢回新版，避免旧请求仍在读数、
        # 新请求又切回旧版所形成的焦点来回切换。
        if delta == 0:
            return
        visible: list[ReviewItem] = []
        for row_index in range(self.tree.topLevelItemCount()):
            row = self.tree.topLevelItem(row_index)
            if row.isHidden():
                continue
            item = self.by_uid.get(int(row.data(0, Qt.UserRole)))
            if item is not None:
                visible.append(item)
        if not visible:
            self._cancel_reverse_sync(restore_focus=restore_focus)
            if restore_focus:
                self._restore_new_focus()
            self.lbl_status.setText("当前筛选下没有图片")
            return
        visible_uids = [it.uid for it in visible]
        if self._cur_uid in visible_uids:
            i = visible_uids.index(self._cur_uid)
            j = max(0, min(len(visible) - 1, i + delta))
            target = visible[j]
        else:
            # 标记已看或编号定位后，预览可能位于筛选外；从其原始位置找可见邻项。
            indices = {it.uid: i for i, it in enumerate(self.items)}
            current_index = indices.get(self._cur_uid)
            if current_index is None:
                target = visible[0] if delta > 0 else visible[-1]
            elif delta > 0:
                target = next((it for it in visible if indices[it.uid] > current_index), visible[-1])
            else:
                target = next((it for it in reversed(visible) if indices[it.uid] < current_index), visible[0])
        if not sync_legacy or not self.chk_sync.isChecked():
            self._cancel_reverse_sync(restore_focus=restore_focus)
        if target.uid != self._cur_uid:
            self._show_item(target)
        if sync_legacy and self.chk_sync.isChecked():
            # 相同目标由 request_target 合并；目标变化只更换代数，不先 cancel 清队列。
            self._reverse_sync(self._old_pos_uid, target.uid, force=True,
                               restore_focus=restore_focus)

    def _mark_seen_next(self) -> None:
        it = self._current()
        if it is None:
            return
        it.seen = True
        self._refresh_list()
        self._step(1, sync_legacy=True)

    def _mark_pick(self, idx: int) -> None:
        it = self._current()
        if it is None:
            return
        cands = it.candidates
        if idx < len(cands):
            it.pick = idx
            it.seen = True
            self.lbl_status.setText(f"已记下：{cands[idx][0]} = {cands[idx][1]:.0f} 岁（仅本机内存）")
            self._refresh_list()
            self._step(1, sync_legacy=True)

    def _current(self) -> ReviewItem | None:
        return self.by_uid.get(self._cur_uid) if self._cur_uid is not None else None

    # ---------------------------------------------------------------- #
    # 跟随原工具（A/D 翻页、S 保存）
    # ---------------------------------------------------------------- #
    def _match_texts(self) -> list[str]:
        """前台窗口标题的匹配串。

        实测 face_attribute_anno 的窗口标题是「年龄 / 性别属性标注」，
        不是文件夹路径，所以这里同时认几个特征词 + 当前数据集的公共路径。
        """
        out = ["属性标注", "face_attribute"]
        if self.items:
            paths = [it.image_path for it in self.items]
            try:
                common = os.path.commonpath(paths)
            except ValueError:
                common = os.path.dirname(paths[0])
            if os.path.isfile(common):
                common = os.path.dirname(common)
            if len(common) >= 7:
                out.append(common)
        return out

    def _on_sync_toggled(self, on: bool) -> None:
        if not self.chk_rev.isChecked() and self._pending_sync is not None:
            self._cancel_reverse_sync()
        if self.chk_sync.isChecked() or self.chk_rev.isChecked():
            self._start_sync()
        else:
            self._stop_sync()
            self.lbl_sync.setText("同步：已关闭")

    def _collect_stopped_sync_workers(self) -> bool:
        """只在线程结束后接收其最终债务；运行中禁止并发启动新线程。"""
        still_running = []
        for worker in self._retiring_sync_workers:
            try:
                running = worker.isRunning()
            except Exception:  # noqa: BLE001
                running = True
            if running:
                still_running.append(worker)
            else:
                self._sync_inflight = worker._inflight
        self._retiring_sync_workers = still_running
        return not still_running

    def _collect_stopped_watchers(self) -> bool:
        still_running = []
        for watcher in self._retiring_watchers:
            try:
                running = watcher.isRunning()
            except Exception:  # noqa: BLE001
                running = True
            if running:
                still_running.append(watcher)
        self._retiring_watchers = still_running
        return not still_running

    def _start_sync(self) -> bool:
        if self._closing or not self.items:
            return False
        workers_stopped = self._collect_stopped_sync_workers()
        watchers_stopped = self._collect_stopped_watchers()
        if not workers_stopped or not watchers_stopped:
            self._sync_restart_when_stopped = True
            if not self._sync_restart_timer.isActive():
                self._sync_restart_timer.start()
            self.lbl_sync.setText("同步：上一次操作正在停止，请稍后重试")
            return False
        self._sync_restart_when_stopped = False
        self._sync_restart_timer.stop()
        paths = [it.image_path for it in self.items]
        if self._sync_worker is None:
            try:
                self._sync_worker = LegacySyncWorker(paths, self, initial_inflight=self._sync_inflight)
                self._sync_worker.position.connect(self._on_legacy_position)
                self._sync_worker.completed.connect(self._on_sync_completed)
                self._sync_worker.warning.connect(self._on_sync_warning)
                self._sync_worker.start()
            except Exception:  # noqa: BLE001
                if self._sync_worker is not None:
                    self._stop_sync()
                self.lbl_sync.setText("同步：启动失败，请稍后重试")
                return False
            self._sync_inflight = None  # 所有权已转给新线程。
        else:
            if self._sync_worker.set_paths(paths):
                self._cancel_reverse_sync()
                self._old_pos_uid = None
        if self._watcher is not None and self._watcher.isRunning():
            self._watcher.set_match_text(self._match_texts())
            self._update_navigation_routing()
            return True
        self._watcher = KeyWatcher(self._match_texts(), self)
        try:
            self._watcher.set_own_hwnd(int(self.winId()))
        except Exception:  # noqa: BLE001
            pass
        self._watcher.key.connect(self._on_global_key)
        self._watcher.local_nav.connect(self._on_local_navigation_key)
        self._watcher.failed.connect(
            lambda m: self.lbl_sync.setText(f"同步：按键不可用：{m}（仍可手动选图）"))
        self._update_navigation_routing()
        self._watcher.start()
        return True

    def _on_retired_sync_finished(self) -> None:
        """线程退出后只恢复监听；已取消的定位请求不会自动重发。"""
        workers_stopped = self._collect_stopped_sync_workers()
        watchers_stopped = self._collect_stopped_watchers()
        if not workers_stopped or not watchers_stopped:
            # finished 通知到达时底层线程可能还在收尾，不与它并发。
            if self._sync_restart_when_stopped and not self._closing:
                if not self._sync_restart_timer.isActive():
                    self._sync_restart_timer.start()
            return
        if not self._sync_restart_when_stopped:
            return
        self._sync_restart_when_stopped = False
        if (not self._closing and self.items
                and (self.chk_sync.isChecked() or self.chk_rev.isChecked())):
            if self._start_sync():
                self.lbl_sync.setText("同步：已恢复，请重新选择目标图片")

    def _update_navigation_routing(self) -> None:
        if self._watcher is not None:
            # 正常使用旧版时也接管 A/D，不能在一次定位完成后就退回旧版目录顺序。
            self._watcher.set_local_navigation(
                not self._sync_blocked and (self.chk_sync.isChecked() or self._pending_sync is not None))

    def _stop_sync(self) -> None:
        self._sync_restart_when_stopped = False
        self._sync_restart_timer.stop()
        try:
            self._cancel_reverse_sync()
        except Exception:  # noqa: BLE001
            self._pending_sync = None
        if self._sync_worker is not None:
            worker = self._sync_worker
            for signal in (worker.position, worker.completed, worker.warning):
                try:
                    signal.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            try:
                worker.finished.connect(self._on_retired_sync_finished)
            except Exception:  # noqa: BLE001
                pass
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._sync_worker = None
            self._retiring_sync_workers.append(worker)
            self._collect_stopped_sync_workers()
        if self._watcher is not None:
            watcher = self._watcher
            try:
                watcher.key.disconnect()
                watcher.local_nav.disconnect()
                watcher.failed.disconnect()
            except Exception:  # noqa: BLE001
                pass
            try:
                watcher.finished.connect(self._on_retired_sync_finished)
                watcher.stop()
            except Exception:  # noqa: BLE001
                pass
            self._watcher = None
            self._retiring_watchers.append(watcher)
            self._collect_stopped_watchers()

    def _on_global_key(self, ch: str) -> None:
        # 钩子只用于唤醒读取，绝不把一次按键当作一次成功翻页。
        if self._sync_worker is not None:
            if ch in ("a", "d"):
                self._sync_worker.cancel()
                self._pending_sync = None
                self._sync_returns_to_new = True
                self._update_navigation_routing()
            self._sync_worker.wake()

    def _on_local_navigation_key(self, delta: int) -> None:
        sync_log("physical_navigation", delta=delta, current=self._cur_uid)
        # 旧版里的物理 A/D 也按新版筛选列表选目标，随后让旧版定位到该目标。
        # 若前台是新版发起同步时临时借用的，完成后仍将焦点归还新版。
        restore_focus = (not self.chk_sync.isChecked()
                         or (self._pending_sync is not None and self._sync_returns_to_new))
        self._step(delta, sync_legacy=True, restore_focus=restore_focus)

    def _on_legacy_position(self, position) -> None:
        # 读到位置不代表导航已恢复；失败后保留旧版原生 A/D，显式重试才恢复接管。
        matches = self._items_by_path.get(position.image_path, [])
        if not matches:
            return
        current = self._current()
        it = current if current in matches else matches[0]
        changed = self._old_pos_uid not in {match.uid for match in matches}
        self._old_pos_uid = it.uid
        if (changed and self._pending_sync is None and self.chk_sync.isChecked()
                and not is_foreground(int(self.winId()))):
            if self._cur_uid != it.uid:
                self._show_item(it)

    def _on_sync_completed(self, generation: int, ok: bool, message: str) -> None:
        if generation != self._pending_sync:
            return
        restore_focus = self._sync_returns_to_new
        self._pending_sync = None
        self._sync_returns_to_new = True
        self._sync_blocked = not ok
        self._update_navigation_routing()
        restored = self._restore_new_focus() if restore_focus else True
        if not ok:
            self._old_pos_uid = None
            message += "（已暂停接管旧版 A/D，可先用旧版原生按键手动对齐）"
        if not restored:
            message += "（请点回新版窗口后继续使用 A/D）"
        self.lbl_sync.setText(f"同步：{'完成' if ok else '失败'} · {message}")
        sync_log("ui_completed", generation=generation, ok=ok,
                 current=self._cur_uid, message=message)

    def _on_sync_warning(self, message: str) -> None:
        self._old_pos_uid = None
        if self._pending_sync is None:
            self.lbl_sync.setText(f"同步：{message}")

    def _index_of(self, uid: int | None) -> int | None:
        if uid is None:
            return None
        return self._item_indices.get(uid)

    def _show_item(self, it: ReviewItem) -> None:
        """把某一张显示到中间/右侧，并尽量在列表里高亮。

        关键：**筛选只影响列表显示，不影响预览**。
        原工具翻到一张被筛掉的图时，这里照样显示它，筛选保持不变。
        """
        self._programmatic = True
        try:
            self._cur_uid = it.uid
            self._show_image(it)
            self._show_panel(it)
            selected = self._select_uid(it.uid)
            if not selected and it.status == "待计算":
                self.worker.prioritize(it.uid)
        finally:
            self._programmatic = False
        self._update_pos()
        if not self._passes(it):
            idx = self._index_of(it.uid)
            self.lbl_status.setText(
                f"第 {(idx or 0) + 1} 张不在当前筛选内（筛选保持不变，预览仍跟随原工具）")

    def _select_uid(self, uid: int) -> bool:
        tw = self._rows_by_uid.get(uid)
        if tw is not None and not tw.isHidden():
            programmatic = self._programmatic
            self._programmatic = True
            try:
                self.tree.setCurrentItem(tw)
                self.tree.scrollToItem(tw, QAbstractItemView.PositionAtCenter)
            finally:
                self._programmatic = programmatic
            return True
        # 实际预览位于筛选外时没有对应行；清掉217等旧高亮，而不是暗示它是当前图。
        blocked = self.tree.blockSignals(True)
        try:
            self.tree.setCurrentItem(None)
            self.tree.clearSelection()
        finally:
            self.tree.blockSignals(blocked)
        return False

    # ---------------------------------------------------------------- #
    # json 变更（原工具保存后自动对齐 + 刷新 anno）
    # ---------------------------------------------------------------- #
    def _watch_dirs(self, items: list[ReviewItem]) -> None:
        dirs = set()
        for it in items:
            if not it.json_path:
                continue
            d = os.path.dirname(it.json_path)
            dirs.add(d)
            if it.json_path not in self._mtimes:
                try:
                    self._mtimes[it.json_path] = os.path.getmtime(it.json_path)
                except OSError:
                    self._mtimes[it.json_path] = 0.0
        exist = set(self._fsw.directories())
        add = [d for d in dirs if d and d not in exist and os.path.isdir(d)]
        if add:
            self._fsw.addPaths(add)

    def _on_dir_changed(self, path: str) -> None:
        self._refresh_changed_json({it.json_path for it in self.items
                                    if it.json_path and os.path.dirname(it.json_path) == path})

    def _poll_json_changes(self) -> None:
        paths = list(dict.fromkeys(it.json_path for it in self.items if it.json_path))
        if not paths:
            return
        start = self._json_poll_cursor % len(paths)
        size = min(256, len(paths))
        batch = {paths[(start + offset) % len(paths)] for offset in range(size)}
        self._json_poll_cursor = (start + size) % len(paths)
        current = self._current()
        if current is not None and current.json_path:
            batch.add(current.json_path)
        self._refresh_changed_json(batch)

    def _refresh_changed_json(self, paths: set[str]) -> None:
        changed_paths: dict[str, float] = {}
        for json_path in paths:
            try:
                m = os.path.getmtime(json_path)
            except OSError:
                continue
            if m != self._mtimes.get(json_path, 0.0):
                changed_paths[json_path] = m
        if not changed_paths:
            return
        refreshed = set(changed_paths)
        # 批量保存/其他写入也能触发目录事件，mtime 不能证明旧版停在哪张。
        for it in self.items:
            if it.json_path in changed_paths:
                if not self._reload_json(it):
                    refreshed.discard(it.json_path)
                self._update_row(it)
        # 只有读成功才记住版本；文件暂时被占用或正写入时，下轮仍会重试。
        for json_path in refreshed:
            self._mtimes[json_path] = changed_paths[json_path]
        current = self._current()
        if current is not None:
            self._show_panel(current)
        if self._sync_worker is not None:
            self._sync_worker.wake()

    def _reload_json(self, it: ReviewItem) -> bool:
        if not it.json_path:
            return False
        try:
            obj = json.loads(Path(it.json_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False
        res = (obj.get("result") or []) if isinstance(obj, dict) else []
        if not isinstance(res, list) or it.face_index >= len(res) or not isinstance(res[it.face_index], dict):
            return False
        r = res[it.face_index]
        signature = json.dumps(r.get("box"), sort_keys=True, ensure_ascii=False)
        if signature != it.box_signature:
            it.box_signature = signature
            it.box = parse_box(r.get("box"))
            it.pred = None
            it.probs = []
            it.labels = []
            it.status = "待计算"
            it.reference_hint = ""
            self.worker.prioritize(it.uid)
        it.ori = r.get("ori_age")
        it.ts = r.get("ts_age")
        it.cake = r.get("cake_age")
        it.cake_gender = r.get("cake_gender")
        it.anno = r.get("anno_age")
        it.anno_gender = r.get("anno_gender")
        it.anno_gender_source = r.get("anno_gender_source", "")
        self._schedule_compute_status()
        return True

    # ---------------------------------------------------------------- #
    # 选择 / 渲染
    # ---------------------------------------------------------------- #
    def _on_select(self, cur: QTreeWidgetItem, prev: QTreeWidgetItem) -> None:
        if cur is None:
            return
        uid = int(cur.data(0, Qt.UserRole))
        it = self.by_uid.get(uid)
        if it is None:
            return
        old_uid = self._cur_uid
        if old_uid is not None and old_uid != uid:
            self._prev_uid = old_uid
        self._cur_uid = uid
        self._show_image(it)
        self._show_panel(it)
        self._update_pos()
        if it.status == "待计算":
            self.worker.prioritize(uid)
        if not self._programmatic:
            self._cancel_reverse_sync()

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        # 以新版筛选列表点击的位置为目标；原始编号与完整目录顺序仍用于旧版定位。
        if item is not None:
            uid = int(item.data(0, Qt.UserRole))
            if uid in self.by_uid:
                # 双击后的第二次释放也可能产生 clicked，不能取消刚请求的显式同步。
                if self._cur_uid != uid:
                    self._cancel_reverse_sync()
                self._show_item(self.by_uid[uid])
                if self.chk_rev.isChecked():
                    self._reverse_sync(self._old_pos_uid, uid)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        """双击与单击共用同一个定位队列，重复目标会合并。"""
        if item is None:
            return
        uid = int(item.data(0, Qt.UserRole))
        if uid in self.by_uid:
            self._show_item(self.by_uid[uid])
        if self.chk_rev.isChecked():
            self._reverse_sync(self._old_pos_uid, uid)

    def _cancel_reverse_sync(self, *, restore_focus: bool = True) -> None:
        if self._pending_sync is not None:
            if self._sync_worker is not None:
                self._sync_worker.cancel()
            self._pending_sync = None
            if restore_focus:
                self._restore_new_focus()
        self._sync_returns_to_new = True
        self._update_navigation_routing()

    def _restore_new_focus(self) -> bool:
        own_hwnd = int(self.winId())
        if is_foreground(own_hwnd):
            return True
        legacy_hwnd = find_window_by_title(["属性标注", "face_attribute"])
        # 用户已经切去其他程序时不抢回来；只归还本次定位临时取走的焦点。
        if legacy_hwnd and is_foreground(legacy_hwnd):
            if _force_foreground(own_hwnd):
                self.tree.setFocus(Qt.OtherFocusReason)
                return True
            return False
        return True

    def _reverse_sync(self, old_uid: int | None, new_uid: int,
                      max_steps: int = 200, force: bool = False, *,
                      restore_focus: bool = True) -> None:
        """统一进入后台队列；old_uid 仅为兼容入口，不作为位置依据。"""
        if not force and not self.chk_rev.isChecked():
            return
        target = self.by_uid.get(new_uid)
        if target is None:
            return
        if not self._start_sync():
            return
        if self._sync_worker is not None:
            if not self._sync_worker.navigation_pending():
                self._sync_blocked = False
            self._sync_returns_to_new = restore_focus
            self._pending_sync = self._sync_worker.request_target(target.image_path)
            self._update_navigation_routing()
        self.lbl_sync.setText("同步：正在核对旧版位置，可继续选择其他图片…")

    def _show_image(self, it: ReviewItem) -> None:
        img = self._img_cache.get(it.image_path)
        if img is None:
            img = imread_bgr(it.image_path)
            if img is None:
                self.preview.clear("图片读取失败")
                return
            self._img_cache.clear()
            self._img_cache[it.image_path] = img
        pix = QPixmap.fromImage(image_from_bgr(img))
        faces = []
        if it.box and it.box[2] > 0:
            gender = ""
            g_src = it.anno_gender if it.anno_gender is not None else it.cake_gender
            if g_src in GENDER_TEXT:
                gender = "·" + GENDER_TEXT[g_src]
            if it.pred is not None:
                label = f"#{it.face_index + 1}  {it.pred:.0f}岁{gender}"
                status = "成功"
            else:
                label = f"#{it.face_index + 1}  计算中…"
                status = "计算中"
            faces.append({
                "index": it.face_index + 1, "box": it.box,
                "status": status, "age": it.pred, "lo": None, "hi": None,
                "label": label,
            })
        self.preview.set_image(pix, faces, hint="")

    def _reset_panel(self) -> None:
        self.lbl_age.setText("—")
        self.lbl_verdict.setText("")
        self.lbl_err.setText("")
        self.bars.clear()
        for lb in self.cand_rows:
            lb.setText("")
            lb.setStyleSheet("")
        self.lbl_gender.setText("")
        self.lbl_reference.setText("")

    def _show_panel(self, it: ReviewItem) -> None:
        self._reset_panel()
        if it.pred is not None:
            self.lbl_age.setText(f"{it.pred:.1f} 岁")
            self.bars.set_probs(it.labels, it.probs)
        else:
            self.lbl_age.setText("…" if it.status == "待计算" else "—")
            self.bars.clear()

        level, text = it.verdict()
        self.lbl_reference.setText(it.reference_hint)
        color = {"ok": "#00E5A0", "warn": "#FF9F43"}.get(level, "#9aa4b2")
        self.lbl_verdict.setText(f'<span style="color:{color}">{text}</span>')

        e = it.err
        b = it.baseline
        if e is None or b is None:
            self.lbl_err.setText("")
        else:
            c = self._row_color(it)
            cname = c.name() if c is not None else "#9aa4b2"
            self.lbl_err.setText(
                f'<span style="color:{cname};font-weight:bold">误差 {it.err_i:+d} 岁</span>'
                f'<span style="color:#8b93a1">（模型 {it.pred:.1f} − {b[1]} {b[0]:.0f} = {e:+.1f}）</span>')

        near = it.nearest()
        cands = it.candidates
        for i, lb in enumerate(self.cand_rows):
            if i >= len(cands):
                lb.setText("")
                lb.setStyleSheet("")
                continue
            name, val = cands[i]
            if near and near[0] == name:
                delta = f"　与模型差 {abs(val - it.pred):.0f} 岁　← 最接近"
                bg, fg = "#26364f", "#9fd0ff"
            else:
                delta = f"　与模型差 {abs(val - it.pred):.0f} 岁" if it.pred is not None else ""
                bg, fg = "#202027", "#b9c4d4"
            if it.pick == i:
                bg, fg = "#1f3b2e", "#7ef0c0"
            self.cand_rows[i].setText(
                f'<b>{i + 1}. {name}</b>　<span style="font-size:15px">{val:.0f}</span> 岁'
                f'<span style="color:#8b93a1">{delta}</span>')
            self.cand_rows[i].setStyleSheet(
                f"padding:4px 8px; border-radius:4px; background:{bg}; color:{fg};")

        if it.anno is not None:
            g = GENDER_TEXT.get(it.anno_gender, "?")
            age = f"{float(it.anno):.0f} 岁" if valid_age(it.anno) else "年龄无效"
            txt = f"<span style='color:#00E5A0'>已保存：{age} · {g}</span>"
        else:
            txt = "<span style='color:#FF9F43'>尚未保存（json 里没有 anno）</span>"
        if it.cake_gender in GENDER_TEXT:
            cg = GENDER_TEXT[it.cake_gender]
            if it.sex_conflict:
                ag = GENDER_TEXT.get(it.anno_gender, "?")
                txt += f"　|　<span style='color:#FF6B6B'>⚠ 数据性别冲突：Cake {cg} → 已标注 {ag}</span>"
            else:
                txt += f"　|　Cake 性别：{cg}"
        g = it.id_gender
        if g in GENDER_TEXT:
            if it.gender_conflict:
                txt += (f"　|　<span style='color:#FF6B6B'>⚠ 文件名身份证性别：{GENDER_TEXT[g]}"
                        f"（与 json 不一致）</span>")
            else:
                compared = any(value in (1, 2) for value in (it.anno_gender, it.cake_gender))
                note = "与已有字段一致" if compared else "无可比较字段"
                txt += f"　|　文件名身份证：{GENDER_TEXT[g]}（{note}）"
        if it.anno_gender not in (1, 2):
            txt += f"<br>待填性别：{GENDER_TEXT[it.fill_gender]} · {it.fill_gender_source}"
        elif it.anno_gender_source in ("user_default_male", "cake_default"):
            txt += f"<br>性别来源：{it.fill_gender_source}"
        if it.gender_needs_review:
            txt += "<br>性别待复核：" + "；".join(it.gender_review_reasons)
        txt += "<br>仅作已有性别数据对照；未从照片验证性别，无数据冲突不代表已确认正确。"
        self.lbl_gender.setText(txt)

    def _on_worker_state(self, state: str, message: str) -> None:
        if state == "loading":
            self._compute_notice = "正在加载 ViT 年龄模型…"
        elif state == "ready":
            self._compute_notice = ""
        elif state == "failed":
            self._compute_notice = f"年龄模型加载失败：{message}"
        else:
            self._compute_notice = message
        self._refresh_compute_status()

    def _refresh_compute_status(self) -> None:
        """统计全部已加载图片；列表筛选和同步提示不能遮住计算进度。"""
        success = sum(it.status == "完成" and it.pred is not None for it in self.items)
        failed = sum(it.status == "失败" for it in self.items)
        no_face = sum(it.status == "无人脸" for it in self.items)
        pending = len(self.items) - success - failed - no_face
        with_error = sum(it.err_i is not None for it in self.items)
        self._computed = success
        self.progress.setRange(0, max(1, len(self.items)))
        self.progress.setValue(success + failed + no_face)
        if self.items:
            message = (f"自动计算（全部 {len(self.items)} 张）：成功 {success} · "
                       f"失败 {failed} · 无人脸 {no_face} · 待算 {pending}；"
                       f"已算误差 {with_error} 张")
            no_baseline = sum(it.status == "完成" and it.pred is not None
                              and it.baseline is None for it in self.items)
            if no_baseline:
                message += f"，缺少有效年龄基准 {no_baseline} 张"
        elif self.worker.ready and not self.worker.load_error:
            message = "自动计算：模型已就绪，添加图片后自动计算"
        else:
            message = "自动计算：等待添加图片"
        if self._compute_notice:
            message += "；" + self._compute_notice.splitlines()[0]
        errors = [it.error for it in self.items if it.status == "失败" and it.error]
        details = self._compute_notice
        if errors:
            details += ("\n" if details else "") + "最近失败：" + errors[-1]
        self.lbl_compute.setText(message)
        self.lbl_compute.setToolTip(details)

    def _schedule_compute_status(self) -> None:
        """合并同一批结果的统计，防止大量图片逐条扫描整个目录。"""
        if not self._compute_status_timer.isActive():
            self._compute_status_timer.start()

    def _on_computed(self, uid: int, out: dict) -> None:
        it = self.by_uid.get(uid)
        if it is None:
            return
        if out.get("box_signature", it.box_signature) != it.box_signature:
            # 运行中旧版已更改框，丢弃旧取景的结果并按新框重算。
            self.worker.prioritize(uid)
            return
        it.status = out.get("status", "完成")
        it.error = out.get("error", "")
        it.pred = out.get("pred")
        it.probs = out.get("probs") or []
        it.labels = out.get("labels") or []
        it.reference_hint = out.get("reference_hint", "")
        if out.get("box"):
            it.box = out["box"]
        self._schedule_compute_status()

        # 只更新对应行，避免整表重建导致选中跳动
        self._update_row(it)
        if uid == self._cur_uid:
            self._show_image(it)
            self._show_panel(it)

    def _update_row(self, it: ReviewItem) -> None:
        row = self._rows_by_uid.get(it.uid)
        passes = self._passes(it)
        membership_changed = False
        blocked = self.tree.blockSignals(True)
        try:
            if row is not None and not passes:
                self._rows_by_uid.pop(it.uid, None)
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(row))
                membership_changed = True
            elif row is None and passes:
                index = self._index_of(it.uid)
                if index is None:
                    return
                # 可见行始终按原始编号排列；二分查找避免预加载期间反复扫描整表。
                lo, hi = 0, self.tree.topLevelItemCount()
                while lo < hi:
                    middle = (lo + hi) // 2
                    if int(self.tree.topLevelItem(middle).text(0)) < index + 1:
                        lo = middle + 1
                    else:
                        hi = middle
                insert_at = lo
                row = self._make_row(it, index)
                self.tree.insertTopLevelItem(insert_at, row)
                self._rows_by_uid[it.uid] = row
                membership_changed = True
            elif row is not None:
                row.setText(2, self._age_text(it))
                row.setText(3, self._err_text(it))
                row.setText(4, self._save_text(it))
                tip = self._row_tip(it)
                row.setToolTip(0, tip)
                row.setToolTip(1, tip)
                self._paint_row(row, it)
            if membership_changed:
                current_row = self._rows_by_uid.get(self._cur_uid)
                if self._cur_uid is not None and self.tree.currentItem() is not current_row:
                    self._select_uid(self._cur_uid)
                self._update_pos()
            if membership_changed or self._extra_filter == "sex_review":
                self._update_list_count()
        finally:
            self.tree.blockSignals(blocked)

    # ---------------------------------------------------------------- #
    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        if self._export_worker is not None and self._export_worker.isRunning():
            self.lbl_status.setText("正在复制图片，完成后关闭…")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        try:
            self._json_poll_timer.stop()
            self._stop_sync()
        except Exception:  # noqa: BLE001
            pass
        if (any(worker.isRunning() for worker in self._retiring_sync_workers)
                or any(watcher.isRunning() for watcher in self._retiring_watchers)):
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if not self._model_stop_requested:
            self.worker.stop()
            self._model_stop_requested = True
        if self.worker.isRunning():
            self.lbl_status.setText("正在等待模型计算结束，完成后关闭…")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        super().closeEvent(event)
