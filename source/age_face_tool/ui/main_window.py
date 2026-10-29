"""主窗口。"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_NAME,
    DEFAULT_HALF_WIDTH,
    MAX_HALF_WIDTH,
    MIN_HALF_WIDTH,
    VERSION,
    calibration_file,
    default_export_path,
)
from ..core.age_backend import backend_status
from ..core.exporter import export_excel
from ..core.interval import MODE_BASE, MODE_CALIBRATED, Calibration
from ..core.pipeline import (
    MODEL_DEEPFACE,
    MODEL_LABELS,
    MODEL_VIT,
    ST_NO_FACE,
    ST_OK,
    ImageResult,
    PipelineConfig,
    other_model,
)
from ..core.scanner import rel_path, scan_folder
from ..core.store import ResultStore, is_fresh
from .preview import FacePreview
from .worker import BatchWorker, ThumbnailLoader

COLOR_PENDING = QColor("#9aa0a6")
COLOR_OK = QColor("#3ddc97")
COLOR_NOFACE = QColor("#ffb020")
COLOR_FAIL = QColor("#ff6b6b")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.resize(1360, 860)

        self.settings = QSettings("AgeFaceTool", "AgeFaceTool")
        self.folder: Path | None = None
        self.files: list[str] = []
        self.results: dict[str, ImageResult] = {}
        self.items: dict[str, QListWidgetItem] = {}
        self.thumbs: dict[str, QIcon] = {}
        self.store: ResultStore | None = None
        self.worker: BatchWorker | None = None
        self.calib: Calibration | None = None
        self.t_start: float = 0.0
        self.current_path: str | None = None
        self._expanded = False

        self.thumb_loader = ThumbnailLoader(self, size=96, max_threads=3)
        self.thumb_loader.signals.ready.connect(self._on_thumb_ready)

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._update_calib_label()
        self._set_running_state(False)

        self._thumb_timer = QTimer(self)
        self._thumb_timer.setInterval(260)
        self._thumb_timer.timeout.connect(self._load_visible_thumbs)
        self._thumb_timer.start()

    # ================================================================== #
    # UI 构建
    # ================================================================== #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 6)
        root.setSpacing(8)

        # ---------- 顶部：数据源 + 参数 ---------- #
        top = QHBoxLayout()
        top.setSpacing(10)

        gb_src = QGroupBox("数据源")
        lay_src = QHBoxLayout(gb_src)
        lay_src.setContentsMargins(10, 6, 10, 8)
        self.ed_folder = QLineEdit()
        self.ed_folder.setReadOnly(True)
        self.ed_folder.setPlaceholderText("尚未选择文件夹")
        self.btn_folder = QPushButton("选择文件夹…")
        self.btn_folder.clicked.connect(self.choose_folder)
        self.chk_recursive = QCheckBox("包含子文件夹")
        self.chk_recursive.setChecked(True)
        self.chk_recursive.stateChanged.connect(lambda _: self.rescan())
        self.btn_rescan = QPushButton("重新扫描")
        self.btn_rescan.clicked.connect(self.rescan)
        lay_src.addWidget(self.ed_folder, 1)
        lay_src.addWidget(self.btn_folder)
        lay_src.addWidget(self.chk_recursive)
        lay_src.addWidget(self.btn_rescan)

        gb_par = QGroupBox("推断设置")
        lay_par = QHBoxLayout(gb_par)
        lay_par.setContentsMargins(10, 6, 10, 8)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("人工参考范围（基础）", MODE_BASE)
        self.cmb_mode.addItem("校准区间（进阶/保形预测）", MODE_CALIBRATED)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.spn_width = QSpinBox()
        self.spn_width.setRange(MIN_HALF_WIDTH, MAX_HALF_WIDTH)
        self.spn_width.setValue(DEFAULT_HALF_WIDTH)
        self.spn_width.setSuffix(" 岁")
        self.spn_width.setPrefix("± ")
        self.spn_width.setToolTip("基础模式下区间为 预测值 ± 该数值")
        self.chk_tta = QCheckBox("多视图采样")
        self.chk_tta.setToolTip("对同一张脸做多次轻微变换后分别预测，用于进阶模式的自适应区间")
        self.chk_skip = QCheckBox("跳过已完成")
        self.chk_skip.setChecked(True)
        self.chk_skip.setToolTip("中断后再次开始时，跳过已完成且文件未变化的图片")
        lay_par.addWidget(QLabel("区间模式:"))
        lay_par.addWidget(self.cmb_mode)
        lay_par.addWidget(QLabel("区间半宽:"))
        lay_par.addWidget(self.spn_width)
        lay_par.addWidget(self.chk_tta)
        lay_par.addWidget(self.chk_skip)
        lay_par.addStretch(1)

        gb_model = QGroupBox("年龄模型")
        lay_model = QHBoxLayout(gb_model)
        lay_model.setContentsMargins(10, 6, 10, 8)
        self.cmb_model = QComboBox()
        self.cmb_model.addItem(MODEL_LABELS[MODEL_DEEPFACE], MODEL_DEEPFACE)
        self.cmb_model.addItem(MODEL_LABELS[MODEL_VIT], MODEL_VIT)
        self.cmb_model.setToolTip(
            "DeepFace：VGGFace 骨干 + 101 类年龄分布；\n"
            "ViT：FairFace 微调的 ViT 年龄分类器。\n"
            "两者在 UTKFace 上的实测 MAE 见 reports/model_comparison.md"
        )
        self.chk_compare = QCheckBox("同时用第二个模型对比")
        self.chk_compare.setToolTip("对每张脸同时运行另一个年龄模型，结果表中显示对比年龄")
        self.chk_compare.stateChanged.connect(self._apply_column_visibility)
        lay_model.addWidget(QLabel("主模型:"))
        lay_model.addWidget(self.cmb_model, 1)
        lay_model.addWidget(self.chk_compare)
        lay_model.addStretch(1)

        top.addWidget(gb_src, 3)
        top.addWidget(gb_par, 4)
        root.addLayout(top)

        mid = QHBoxLayout()
        mid.setSpacing(10)
        mid.addWidget(gb_model, 1)
        root.addLayout(mid)

        # ---------- 操作栏 ---------- #
        bar = QHBoxLayout()
        self.btn_start = QPushButton("开始推断")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_batch)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_batch)
        self.btn_export = QPushButton("导出 Excel")
        self.btn_export.clicked.connect(self.do_export)
        self.btn_export.setEnabled(False)
        self.lbl_disclaimer = QLabel(
            "提示：基础区间是人为设定的参考范围，不是统计置信区间；所有数值均来自本地模型实际推断。"
        )
        self.lbl_disclaimer.setObjectName("hint")
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_export)
        bar.addSpacing(12)
        bar.addWidget(self.lbl_disclaimer, 1)
        root.addLayout(bar)

        # ---------- 主体 ---------- #
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        self.lbl_count = QLabel("图片列表")
        lv.addWidget(self.lbl_count)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setIconSize(QSize(96, 96))
        self.list.setGridSize(QSize(124, 148))
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setSpacing(2)
        self.list.currentItemChanged.connect(lambda cur, prev: self._on_select(cur))
        self.list.verticalScrollBar().valueChanged.connect(lambda _: self._load_visible_thumbs())
        lv.addWidget(self.list, 1)
        splitter.addWidget(left)

        right = QSplitter(Qt.Vertical)
        splitter.addWidget(right)

        # 预览
        pw = QWidget()
        pv = QVBoxLayout(pw)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(4)
        self.lbl_preview = QLabel("当前图片")
        pv.addWidget(self.lbl_preview)
        self.preview = FacePreview()
        pv.addWidget(self.preview, 1)
        right.addWidget(pw)

        # 结果表 + 日志
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(4)

        head = QHBoxLayout()
        head.addWidget(QLabel("当前图片人脸结果"))
        self.chk_expand = QCheckBox("展开模型内部预测值")
        self.chk_expand.stateChanged.connect(self._toggle_expand)
        head.addWidget(self.chk_expand)
        head.addStretch(1)
        self.lbl_detail = QLabel("")
        self.lbl_detail.setObjectName("hint")
        head.addWidget(self.lbl_detail)
        bv.addLayout(head)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["#", "预测年龄", "年龄区间", "区间半宽(±)", "检测置信度", "多视图预测值", "标准差", "对比年龄", "状态"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        bv.addWidget(self.table, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setPlaceholderText("运行日志")
        self.log.setFixedHeight(110)
        bv.addWidget(self.log)
        right.addWidget(bottom)
        right.setSizes([420, 380])
        splitter.setSizes([300, 1000])

        # ---------- 状态栏 ---------- #
        status = QWidget()
        sl = QHBoxLayout(status)
        sl.setContentsMargins(0, 2, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setFixedWidth(260)
        self.lbl_stat = QLabel("等待开始")
        self.lbl_device = QLabel("设备: 未加载模型")
        self.lbl_device.setObjectName("hint")
        sl.addWidget(self.progress)
        sl.addWidget(self.lbl_stat, 1)
        sl.addWidget(self.lbl_device)
        root.addWidget(status)

    def _build_menu(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("文件")
        a_open = QAction("选择图片文件夹…", self)
        a_open.triggered.connect(self.choose_folder)
        a_export = QAction("导出 Excel…", self)
        a_export.triggered.connect(self.do_export)
        a_clear = QAction("清除本文件夹的断点记录", self)
        a_clear.triggered.connect(self.clear_state)
        a_quit = QAction("退出", self)
        a_quit.triggered.connect(self.close)
        for a in (a_open, a_export):
            m_file.addAction(a)
        m_file.addSeparator()
        m_file.addAction(a_clear)
        m_file.addSeparator()
        m_file.addAction(a_quit)

        m_help = mb.addMenu("帮助")
        a_cal = QAction("加载校准文件…", self)
        a_cal.triggered.connect(self.import_calibration)
        a_about = QAction("关于 / 使用说明", self)
        a_about.triggered.connect(self.show_about)
        m_help.addAction(a_cal)
        m_help.addAction(a_about)

    # ================================================================== #
    # 设置持久化
    # ================================================================== #
    def _restore_settings(self) -> None:
        folder = self.settings.value("folder", "", type=str)
        if folder and Path(folder).is_dir():
            self.folder = Path(folder)
            self.ed_folder.setText(folder)
            self.rescan()
        self.chk_recursive.setChecked(self.settings.value("recursive", True, type=bool))
        self.spn_width.setValue(int(self.settings.value("half_width", DEFAULT_HALF_WIDTH)))
        self.chk_tta.setChecked(self.settings.value("tta", False, type=bool))
        self.chk_skip.setChecked(self.settings.value("skip", True, type=bool))
        mode = self.settings.value("mode", MODE_BASE, type=str)
        idx = self.cmb_mode.findData(mode)
        if idx >= 0:
            self.cmb_mode.setCurrentIndex(idx)
        model = self.settings.value("model", MODEL_VIT, type=str)
        idx = self.cmb_model.findData(model)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)
        self.chk_compare.setChecked(self.settings.value("compare", False, type=bool))

    def _save_settings(self) -> None:
        self.settings.setValue("folder", str(self.folder) if self.folder else "")
        self.settings.setValue("recursive", self.chk_recursive.isChecked())
        self.settings.setValue("half_width", self.spn_width.value())
        self.settings.setValue("tta", self.chk_tta.isChecked())
        self.settings.setValue("skip", self.chk_skip.isChecked())
        self.settings.setValue("mode", self.cmb_mode.currentData())
        self.settings.setValue("model", self.cmb_model.currentData())
        self.settings.setValue("compare", self.chk_compare.isChecked())

    # ================================================================== #
    # 文件夹
    # ================================================================== #
    def choose_folder(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, APP_NAME, "请先停止当前任务。")
            return
        start = str(self.folder) if self.folder else str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "选择包含人脸照片的文件夹", start)
        if not d:
            return
        self.folder = Path(d)
        self.ed_folder.setText(str(self.folder))
        self.rescan()

    def rescan(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self.folder:
            return
        self.files = [str(p) for p in scan_folder(self.folder, self.chk_recursive.isChecked())]
        self.results.clear()
        self.items.clear()
        self.thumbs.clear()
        self.thumb_loader.clear()
        self.list.clear()
        self.table.setRowCount(0)
        self.preview.clear("请在左侧选择一张图片")
        self.store = ResultStore(self.folder)

        if not self.files:
            self.lbl_count.setText("图片列表（0 张）")
            self.log_line("warn", f"该文件夹下没有找到支持的图片：{self.folder}")
            self._set_running_state(False)
            return

        for path in self.files:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setForeground(COLOR_PENDING)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setSizeHint(QSize(120, 144))
            item.setToolTip(path)
            self.list.addItem(item)
            self.items[path] = item
        self.lbl_count.setText(f"图片列表（{len(self.files)} 张）")
        self.log_line("info", f"已扫描到 {len(self.files)} 张图片：{self.folder}")

        # 载入断点记录
        cached = self.store.load()
        reused = 0
        for path in self.files:
            r = cached.get(str(Path(path)).lower()) or cached.get(str(Path(path)))
            if r and is_fresh(r, Path(path)):
                self.results[path] = r
                self._update_item(path, r)
                reused += 1
        if reused:
            self.log_line("info", f"已恢复 {reused} 张图片的历史结果（断点续跑）")
            self.btn_export.setEnabled(True)
        self._update_counters()
        if self.files:
            self.list.setCurrentRow(0)
        self._set_running_state(False)

    # ================================================================== #
    # 运行控制
    # ================================================================== #
    def start_batch(self) -> None:
        if not self.files:
            QMessageBox.information(self, APP_NAME, "请先选择一个包含图片的文件夹。")
            return
        if self.worker and self.worker.isRunning():
            return

        mode = self.cmb_mode.currentData()
        calib = None
        if mode == MODE_CALIBRATED:
            calib = self._load_calibration()
            if calib is None:
                QMessageBox.warning(
                    self,
                    APP_NAME,
                    "未找到校准文件，无法使用进阶模式。\n本次将自动回退到「人工参考范围」模式。\n\n"
                    "（可用 帮助 → 加载校准文件… 导入 calibration.json）",
                )
                mode = MODE_BASE

        cfg = PipelineConfig(
            mode=mode,
            half_width=self.spn_width.value(),
            use_tta=self.chk_tta.isChecked() or (mode == MODE_CALIBRATED and (calib.strategy if calib else "") == "scaled"),
            calib=calib,
            model=self.cmb_model.currentData(),
            compare_with=other_model(self.cmb_model.currentData()) if self.chk_compare.isChecked() else None,
        )

        todo = self.files
        if self.chk_skip.isChecked():
            todo = [p for p in self.files if not (p in self.results and is_fresh(self.results[p], Path(p)))]
        if not todo:
            QMessageBox.information(self, APP_NAME, "所有图片都已处理完成。\n如需重新推断，请取消勾选「跳过已完成」。")
            return

        if self.store is None:
            self.store = ResultStore(self.folder)
        self.t_start = time.perf_counter()
        self.worker = BatchWorker(todo, cfg, self.store, self)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_done.connect(self._on_image_done)
        self.worker.sig_log.connect(self.log_line)
        self.worker.sig_model.connect(self._on_model)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.sig_stage.connect(self._on_stage)
        self._set_running_state(True)
        self.progress.setRange(0, len(todo))
        self.progress.setValue(0)
        self.log_line("info", f"开始处理 {len(todo)} 张图片（模式：{self.cmb_mode.currentText()}）")
        self.worker.start()

    def toggle_pause(self) -> None:
        if not (self.worker and self.worker.isRunning()):
            return
        if self.worker.paused:
            self.worker.resume()
            self.btn_pause.setText("暂停")
            self.log_line("info", "已继续")
        else:
            self.worker.pause()
            self.btn_pause.setText("继续")
            self.log_line("info", "已暂停（当前图片处理完后暂停）")

    def stop_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_line("warn", "正在停止…")
            self.btn_stop.setEnabled(False)

    def _on_stage(self, text: str) -> None:
        self.lbl_stat.setText(text)

    def _on_model(self, ok: bool, desc: str) -> None:
        if ok:
            self.lbl_device.setText(f"设备: {desc}")
        else:
            self.lbl_device.setText("设备: 模型加载失败")
            self.log_line("error", f"模型加载失败: {desc}")

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        elapsed = time.perf_counter() - self.t_start
        speed = done / elapsed if elapsed > 0 and done else 0.0
        eta = (total - done) / speed if speed > 0 else 0
        self.lbl_stat.setText(
            f"已完成 {done}/{total} 张 · 人脸 {sum(r.n_faces for r in self.results.values())} 个 · "
            f"{speed:.2f} 张/秒 · 剩余约 {int(eta)} 秒"
        )

    def _on_image_done(self, res: ImageResult) -> None:
        self.results[res.path] = res
        self._update_item(res.path, res)
        if res.status in ("读取失败", "处理失败", "部分人脸失败"):
            self.log_line("warn", f"{Path(res.path).name}: {res.status} {res.error}")
        if self.current_path == res.path:
            self._render_current()
        self.btn_export.setEnabled(bool(self.results))

    def _on_finished(self, stopped: bool) -> None:
        self._set_running_state(False)
        self.progress.setValue(self.progress.maximum())
        n_img = len(self.results)
        n_face = sum(r.n_faces for r in self.results.values())
        msg = f"{'已停止' if stopped else '处理完成'}：共 {n_img} 张图片，{n_face} 张人脸"
        self.lbl_stat.setText(msg)
        self.log_line("info", msg)
        self.btn_export.setEnabled(bool(self.results))
        self.worker = None
        self._update_counters()

    def _set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_folder.setEnabled(not running)
        self.btn_rescan.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        self.btn_pause.setText("暂停")
        self.cmb_mode.setEnabled(not running)
        self.cmb_model.setEnabled(not running)
        self.chk_compare.setEnabled(not running)
        self.spn_width.setEnabled(not running)
        if not running:
            self.btn_export.setEnabled(bool(self.results))

    # ================================================================== #
    # 列表与预览
    # ================================================================== #
    def _summary_text(self, res: ImageResult | None) -> str:
        if res is None:
            return "待处理"
        if res.status == ST_NO_FACE:
            return "未检测到人脸"
        if res.status in ("读取失败", "处理失败"):
            return res.status
        ages = [f.age for f in res.faces if f.age is not None]
        if not ages:
            return res.status
        if len(ages) == 1:
            f = res.faces[0]
            return f"{f.age:.0f}岁 [{f.lo}-{f.hi}]"
        return f"{len(ages)}人 {min(ages):.0f}-{max(ages):.0f}岁"

    def _update_item(self, path: str, res: ImageResult) -> None:
        item = self.items.get(path)
        if item is None:
            return
        item.setText(f"{Path(path).name}\n{self._summary_text(res)}")
        if res.status == ST_OK:
            item.setForeground(COLOR_OK)
        elif res.status == ST_NO_FACE:
            item.setForeground(COLOR_NOFACE)
        else:
            item.setForeground(COLOR_FAIL)

    def _on_select(self, cur: QListWidgetItem | None) -> None:
        if cur is None:
            return
        path = cur.data(Qt.UserRole)
        self.current_path = path
        self._render_current()

    def _render_current(self) -> None:
        path = self.current_path
        if not path:
            return
        res = self.results.get(path)
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        orig_w, orig_h = (size.width(), size.height()) if size.isValid() else (0, 0)
        if size.isValid():
            long_side = max(size.width(), size.height())
            if long_side > 1500:
                scale = 1500 / long_side
                reader.setScaledSize(QSize(int(size.width() * scale), int(size.height() * scale)))
        img = reader.read()
        if img.isNull():
            self.preview.clear("无法显示该图片（可能已损坏或格式不支持）")
            self.table.setRowCount(0)
            self.lbl_preview.setText(f"当前图片：{Path(path).name}")
            return
        if res is not None and res.width and res.height:
            orig_w, orig_h = res.width, res.height
        elif orig_w == 0:
            orig_w, orig_h = img.width(), img.height()

        faces = []
        if res is not None:
            faces = [f.to_dict() | {"status": f.status} for f in res.faces]
        self.preview.set_image(QPixmap.fromImage(img), faces, "", coord_size=(orig_w, orig_h))
        self.lbl_preview.setText(
            f"当前图片：{Path(path).name}    {orig_w}×{orig_h}    {Path(path).parent}"
        )

        self._fill_table(res)
        if res is None:
            self.lbl_detail.setText("尚未处理")
        else:
            self.lbl_detail.setText(
                f"状态: {res.status}   人脸: {res.n_faces}   耗时: {res.elapsed:.2f}s"
                + (f"   错误: {res.error}" if res.error else "")
            )

    def _fill_table(self, res: ImageResult | None) -> None:
        self.table.setRowCount(0)
        if res is None:
            return
        for f in res.faces:
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [
                str(f.index),
                f"{f.age:.1f}" if f.age is not None else "-",
                f"{f.lo} ~ {f.hi}" if f.age is not None else "-",
                f"±{f.half:.1f}",
                f"{f.det_score:.3f}",
                ", ".join(f"{v:.1f}" for v in f.views) if len(f.views) > 1 else "-",
                f"{f.sigma:.2f}" if f.sigma is not None else "-",
                f"{f.age2:.1f}" if f.age2 is not None else "-",
                f.status + (f"（{f.error}）" if f.error else ""),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c != 8:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)
        self._apply_column_visibility()

    def _toggle_expand(self, state: int) -> None:
        self._expanded = bool(state)
        self._apply_column_visibility()

    def _apply_column_visibility(self) -> None:
        self.table.setColumnHidden(3, not self._expanded)
        self.table.setColumnHidden(4, not self._expanded)
        self.table.setColumnHidden(5, not self._expanded)
        self.table.setColumnHidden(7, not self.chk_compare.isChecked())
        self.table.setColumnHidden(6, not self._expanded)

    def _load_visible_thumbs(self) -> None:
        if not self.files:
            return
        vp = self.list.viewport().rect()
        for i in range(self.list.count()):
            item = self.list.item(i)
            path = item.data(Qt.UserRole)
            if path in self.thumbs:
                continue
            rect = self.list.visualItemRect(item)
            if rect.bottom() < vp.top() - 200 or rect.top() > vp.bottom() + 200:
                continue
            self.thumb_loader.request(path)

    def _on_thumb_ready(self, path: str, image) -> None:
        self.thumb_loader.pending.discard(path)
        item = self.items.get(path)
        if item is None or image is None:
            return
        icon = QIcon(QPixmap.fromImage(image))
        if len(self.thumbs) > 900:
            self.thumbs.clear()
        self.thumbs[path] = icon
        item.setIcon(icon)

    def _update_counters(self) -> None:
        n = len(self.files)
        done = len(self.results)
        self.lbl_count.setText(f"图片列表（{n} 张，已完成 {done}）")

    # ================================================================== #
    # 校准文件
    # ================================================================== #
    def _load_calibration(self) -> Calibration | None:
        p = calibration_file()
        if p.exists():
            try:
                self.calib = Calibration.load(p)
                backend = (self.calib.metrics or {}).get("backend")
                if backend and self.cmb_model.currentData() != backend:
                    self.log_line(
                        "warn",
                        f"校准文件是为「{MODEL_LABELS.get(backend, backend)}」标定的，"
                        f"当前主模型是「{MODEL_LABELS.get(self.cmb_model.currentData(), '')}」，"
                        "区间宽度可能与当前模型不匹配，建议切回对应模型或重新校准",
                    )
                return self.calib
            except Exception as e:  # noqa: BLE001
                self.log_line("error", f"校准文件读取失败: {e}")
        return None

    def _update_calib_label(self) -> None:
        c = self.calib or self._load_calibration()
        if c:
            self.lbl_disclaimer.setText(
                f"校准区间：{c.describe()}（覆盖率来自验证集实测，非对本机照片的保证）"
            )
        else:
            self.lbl_disclaimer.setText(
                "提示：基础区间是人为设定的参考范围，不是统计置信区间；所有数值均来自本地模型实际推断。"
            )

    def import_calibration(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "选择校准文件", "", "校准文件 (*.json)")
        if not f:
            return
        try:
            c = Calibration.load(f)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"无法读取校准文件:\n{e}")
            return
        dst = calibration_file()
        try:
            dst.write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, f"保存校准文件失败:\n{e}")
            return
        self.calib = c
        self._update_calib_label()
        self.log_line("info", f"已加载校准文件: {c.describe()}")

    def _on_mode_changed(self, _idx: int) -> None:
        mode = self.cmb_mode.currentData()
        if mode == MODE_CALIBRATED:
            c = self.calib or self._load_calibration()
            if c and c.strategy == "scaled":
                self.chk_tta.setChecked(True)
        self._update_calib_label()

    # ================================================================== #
    # 导出
    # ================================================================== #
    def do_export(self) -> None:
        if not self.results:
            QMessageBox.information(self, APP_NAME, "还没有可导出的结果。")
            return
        default = default_export_path()
        if self.folder:
            default = self.folder / f"年龄区间结果_{time.strftime('%Y%m%d_%H%M')}.xlsx"
        f, _ = QFileDialog.getSaveFileName(self, "导出 Excel", str(default), "Excel 工作簿 (*.xlsx)")
        if not f:
            return
        if not f.lower().endswith(".xlsx"):
            f += ".xlsx"
        ordered = [self.results[p] for p in self.files if p in self.results]
        mode = self.cmb_mode.currentData()
        calib = self._load_calibration() if mode == MODE_CALIBRATED else None
        model_name = self.lbl_device.text().replace("设备: ", "")
        if not model_name or "未加载" in model_name:
            model_name = "DeepFace-Age (TensorFlow, CPU)"
        try:
            out = export_excel(
                ordered,
                f,
                model_name=model_name,
                mode_label=self.cmb_mode.currentText(),
                half_width=self.spn_width.value(),
                calib=calib,
                notes="基础区间=预测值±半宽（人工设定）；校准区间由保形预测在公开验证集上标定。",
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"导出失败:\n{type(e).__name__}: {e}")
            return
        self.log_line("info", f"已导出 {len(ordered)} 条图片结果到 {out}")
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setText(f"导出成功：\n{out}")
        box.setInformativeText("是否打开所在文件夹？")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if box.exec() == QMessageBox.Yes:
            self._open_folder(out.parent)

    @staticmethod
    def _open_folder(path: Path) -> None:
        import os

        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def clear_state(self) -> None:
        if not self.folder:
            return
        if QMessageBox.question(self, APP_NAME, "确定清除本文件夹的断点记录？\n（不会删除任何图片）") != QMessageBox.Yes:
            return
        ResultStore(self.folder).clear()
        self.results.clear()
        for p, item in self.items.items():
            item.setText(f"{Path(p).name}\n待处理")
            item.setForeground(COLOR_PENDING)
        self.log_line("info", "已清除断点记录")

    # ================================================================== #
    def show_about(self) -> None:
        ok, desc = backend_status()
        QMessageBox.information(
            self,
            f"关于 {APP_NAME}",
            f"{APP_NAME} v{VERSION}\n\n"
            "功能：本地批量检测人脸并推断年龄区间，结果可导出 Excel。\n\n"
            "• 检测：OpenCV YuNet（本地 ONNX 模型）\n"
            f"• 年龄：DeepFace 表观年龄模型（本地 TensorFlow，{'已加载' if ok else '未加载'}）\n"
            f"• 状态：{desc}\n\n"
            "重要说明：\n"
            "1. 基础模式的区间 = 预测值 ± 半宽，是人为参考范围，不是统计置信区间。\n"
            "2. 进阶模式的区间由带真实年龄标签的公开数据校准，覆盖率在该验证集上实测。\n"
            "3. 年龄为模型基于外观的推断（表观年龄），与身份证年龄可能不同。\n"
            "4. 图片只在本机处理，不会上传；不推断性别、种族等属性。",
        )

    # ================================================================== #
    def log_line(self, level: str, msg: str) -> None:
        color = {"error": "#ff6b6b", "warn": "#ffb020", "info": "#8ab4f8", "debug": "#7a7a7a"}.get(level, "#cfcfcf")
        ts = time.strftime("%H:%M:%S")
        safe = (
            str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        )
        self.log.appendHtml(f'<span style="color:#666">[{ts}]</span> <span style="color:{color}">{safe}</span>')

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, APP_NAME, "任务正在运行，确定退出？\n（已完成的图片会保留，可下次续跑）") != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait(6000)
        if self.store:
            self.store.close()
        self._save_settings()
        event.accept()
