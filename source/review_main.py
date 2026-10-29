"""年龄复查助手入口（只读）。

用法:
    D:\\AgeFaceEnv\\venv\\Scripts\\python.exe review_main.py
    D:\\AgeFaceEnv\\venv\\Scripts\\python.exe review_main.py "F:\\yll\\10.21标年龄\\12"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from PySide6.QtCore import QLockFile
from PySide6.QtGui import QFont
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

from age_face_tool.config import APP_NAME, app_data_dir


def main() -> int:
    from main import QSS, _dark_palette  # 复用主程序的深色主题

    app = QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} 复查助手")
    # 每个用户只运行一套复查助手，避免多套全局 A/D 钩子争用同一个旧版窗口。
    server_name = "AgeFaceReviewAssistant-" + os.environ.get("USERNAME", "default")
    instance_lock = QLockFile(str(app_data_dir() / "review-instance.lock"))
    instance_lock.setStaleLockTime(0)
    # Windows 的命名管道允许同名监听；必须先用进程锁排除重复启动。
    if not instance_lock.tryLock(0):
        client = QLocalSocket()
        client.connectToServer(server_name)
        if not client.waitForConnected(1000):
            QMessageBox.critical(None, "启动失败", "无法连接已运行的复查助手，请关闭已有窗口后重新打开。")
            return 1
        client.write(json.dumps(sys.argv[1:], ensure_ascii=False).encode("utf-8") + b"\n")
        while client.bytesToWrite():
            if not client.waitForBytesWritten(1000):
                return 1
        client.disconnectFromServer()
        return 0
    server = QLocalServer(app)
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(server_name):
        QMessageBox.critical(None, "启动失败", f"无法建立窗口唤回通道：{server.errorString()}")
        return 1
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(QSS)
    app.setFont(QFont("Microsoft YaHei UI", 9))

    from age_face_tool.ui.review_window import ReviewWindow
    from age_face_tool.core.sync_log import sync_log
    from age_face_tool.core.legacy_buttons import SYNC_REVISION

    w = ReviewWindow()
    sync_log("application_start", entry=str(Path(__file__).resolve()), pid=os.getpid(),
             revision=SYNC_REVISION)

    def accept_activation():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()

            def receive(connection=connection):
                if not connection.canReadLine():
                    if connection.bytesAvailable() > 1024 * 1024:
                        connection.abort()
                    return
                try:
                    try:
                        paths = json.loads(bytes(connection.readLine()).decode("utf-8"))
                    except (ValueError, UnicodeError):
                        return
                    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                        return
                    if paths:
                        w.add_paths(paths)
                    if w.isMinimized():
                        w.showNormal()
                    else:
                        w.show()
                    w.raise_()
                    w.activateWindow()
                finally:
                    connection.disconnectFromServer()

            connection.readyRead.connect(receive)
            connection.disconnected.connect(connection.deleteLater)
            receive()

    server.newConnection.connect(accept_activation)
    w.show()
    if len(sys.argv) > 1:
        w.add_paths(sys.argv[1:])

    # 便于无头自测：设置 REVIEW_AUTOCLOSE_MS 后到点自动退出
    auto = int(os.environ.get("REVIEW_AUTOCLOSE_MS", "0") or 0)
    if auto > 0:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(auto, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
