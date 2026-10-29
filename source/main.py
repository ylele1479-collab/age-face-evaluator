"""程序入口。"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 让 `python main.py` 与 `python -m age_face_tool` 都能正确导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

from age_face_tool.config import APP_NAME, VERSION

QSS = """
QWidget { background-color: #1b1b1f; color: #e6e6e6; font-size: 12px; }
QMainWindow, QDialog { background-color: #1b1b1f; }
QGroupBox {
    border: 1px solid #33333c; border-radius: 6px; margin-top: 10px;
    padding: 8px 8px 6px 8px; background-color: #202027;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fb6d6; }
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTableWidget, QListWidget {
    background-color: #17171b; border: 1px solid #33333c; border-radius: 5px;
    selection-background-color: #2f5d9e; selection-color: #ffffff;
}
QLineEdit { padding: 4px 6px; }
QComboBox { padding: 3px 6px; }
QComboBox QAbstractItemView { background: #202027; selection-background-color: #2f5d9e; }
QPushButton {
    background-color: #2a2a33; border: 1px solid #3a3a46; border-radius: 5px;
    padding: 5px 14px; min-height: 20px;
}
QPushButton:hover { background-color: #34343f; }
QPushButton:pressed { background-color: #24242c; }
QPushButton:disabled { color: #6b6b75; background-color: #232329; border-color: #2c2c34; }
QPushButton#primary {
    background-color: #2f6fdb; border-color: #3d82f0; color: #ffffff; font-weight: bold;
}
QPushButton#primary:hover { background-color: #3b7dea; }
QPushButton#primary:disabled { background-color: #2a3a55; color: #8b93a1; }
QListWidget::item { border-radius: 6px; padding: 2px; }
QListWidget::item:selected { background-color: #2b3b55; border: 1px solid #3d82f0; }
QTableWidget { gridline-color: #2b2b33; }
QHeaderView::section {
    background-color: #24242c; color: #b9c4d4; border: none;
    border-right: 1px solid #2f2f38; border-bottom: 1px solid #2f2f38; padding: 4px;
}
QProgressBar {
    border: 1px solid #33333c; border-radius: 4px; background-color: #17171b;
    text-align: center; color: #d8e0ea; height: 16px;
}
QProgressBar::chunk { background-color: #2f6fdb; border-radius: 3px; }
QSplitter::handle { background-color: #2b2b33; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QCheckBox { spacing: 5px; }
QMenuBar { background-color: #202027; }
QMenuBar::item:selected { background-color: #2f5d9e; }
QMenu { background-color: #202027; border: 1px solid #33333c; }
QMenu::item:selected { background-color: #2f5d9e; }
QScrollBar:vertical { background: #17171b; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3a46; border-radius: 5px; min-height: 26px; }
QScrollBar::handle:vertical:hover { background: #4a4a58; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar:horizontal { background: #17171b; height: 11px; }
QScrollBar::handle:horizontal { background: #3a3a46; border-radius: 5px; min-width: 26px; }
QLabel#hint { color: #8b93a1; }
QStatusBar { background: #202027; }
"""


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#1b1b1f"))
    p.setColor(QPalette.WindowText, QColor("#e6e6e6"))
    p.setColor(QPalette.Base, QColor("#17171b"))
    p.setColor(QPalette.AlternateBase, QColor("#202027"))
    p.setColor(QPalette.Text, QColor("#e6e6e6"))
    p.setColor(QPalette.Button, QColor("#2a2a33"))
    p.setColor(QPalette.ButtonText, QColor("#e6e6e6"))
    p.setColor(QPalette.Highlight, QColor("#2f6fdb"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, QColor("#202027"))
    p.setColor(QPalette.ToolTipText, QColor("#e6e6e6"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#6b6b75"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6b6b75"))
    return p


def _excepthook(exc_type, exc, tb) -> None:
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(msg)
    try:
        QMessageBox.critical(None, APP_NAME, f"程序发生未处理的异常：\n\n{exc}\n\n详情已输出到控制台。")
    except Exception:  # noqa: BLE001
        pass


def run_selftest(folder: str) -> int:
    """无头自检：扫描文件夹并处理全部图片，写 selftest_report.txt。

    用法: AgeFaceTool.exe --selftest <图片文件夹>
    用于在没有界面的环境下验证打包后的程序能否正常运行。
    """
    from age_face_tool.core.detector import FaceDetector
    from age_face_tool.core.exporter import export_excel
    from age_face_tool.core.pipeline import MODEL_DEEPFACE, MODEL_VIT, Pipeline, PipelineConfig, load_backend
    from age_face_tool.core.scanner import scan_folder

    root = Path(folder)
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg)

    exit_code = 3
    try:
        say(f"自检开始: {root}")
        files = [str(p) for p in scan_folder(root, True)]
        say(f"扫描到 {len(files)} 个图片文件")
        if not files:
            say("结果: 失败（没有图片）")
            exit_code = 1
            return exit_code
        backend = load_backend(MODEL_VIT)
        say(f"年龄模型(主): {backend.describe()}")
        try:
            second = load_backend(MODEL_DEEPFACE)
            say(f"年龄模型(对比): {second.describe()}")
        except Exception as e:  # noqa: BLE001
            second = None
            say(f"对比模型不可用（不影响主模型自检）: {type(e).__name__}: {e}")
        pipe = Pipeline(FaceDetector(), backend,
                        PipelineConfig(half_width=4, use_tta=False, model=MODEL_VIT,
                                       compare_with=MODEL_DEEPFACE if second else None),
                        second_backend=second)

        results = []
        ok = noface = failed = faces = 0
        for f in files:
            r = pipe.process(f)
            results.append(r)
            faces += r.n_faces
            if r.status == "成功":
                ok += 1
            elif r.status == "未检测到人脸":
                noface += 1
            else:
                failed += 1
            say(f"  {Path(f).name}: {r.status} 人脸={r.n_faces}")

        out = root.parent / "selftest_result.xlsx"
        try:
            export_excel(results, out, model_name=backend.describe(),
                         mode_label="人工参考范围（基础）", half_width=4)
            say(f"Excel 已导出: {out}")
        except Exception as e:  # noqa: BLE001
            say(f"Excel 导出失败: {e}")

        say(f"汇总: 成功 {ok}，未检测到人脸 {noface}，失败 {failed}，共 {faces} 张人脸")
        exit_code = 0 if failed == 0 else 2
        say("结果: 通过" if failed == 0 else "结果: 部分图片失败（异常已被记录）")
    except Exception as e:  # noqa: BLE001
        import traceback

        say(f"结果: 异常 {type(e).__name__}: {e}")
        say(traceback.format_exc())
        exit_code = 3
    finally:
        log = Path.cwd() / "selftest_report.txt"
        try:
            log.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass
    return exit_code


def main() -> int:
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        folder = sys.argv[i + 1] if len(sys.argv) > i + 1 else "."
        return run_selftest(folder)

    # 同一个 exe 两种模式：
    #   1) 命令行带 --review
    #   2) exe 文件名里含"复查"（把 AgeFaceTool.exe 复制一份改名即可，共享 _internal）
    exe_stem = Path(sys.argv[0]).stem
    review_mode = ("--review" in sys.argv) or ("复查" in exe_stem)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(QSS)
    f = QFont("Microsoft YaHei UI", 9)
    app.setFont(f)
    sys.excepthook = _excepthook

    if review_mode:
        from age_face_tool.ui.review_window import ReviewWindow

        w = ReviewWindow()
        w.show()
        paths = [a for a in sys.argv[1:] if not a.startswith("--")]
        if paths:
            w.add_paths(paths)
        auto = int(os.environ.get("REVIEW_AUTOCLOSE_MS", "0") or 0)
        if auto > 0:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(auto, app.quit)
        return app.exec()

    from age_face_tool.ui.main_window import MainWindow

    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
