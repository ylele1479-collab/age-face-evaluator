"""图片预览控件：按比例缩放绘制原图 + 人脸框 + 年龄区间标签。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

BOX_OK = QColor("#00E5A0")
BOX_FAIL = QColor("#FF6B6B")
BOX_SEL = QColor("#FFD166")
TEXT_COLOR = QColor("#0B0B0B")


class FacePreview(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pix: QPixmap | None = None
        self._coord: tuple[int, int] = (0, 0)
        self._faces: list[dict] = []
        self._hint = "请在左侧选择一张图片"
        self._selected = -1
        self.setAutoFillBackground(False)

    # ------------------------------------------------------------------ #
    def clear(self, hint: str = "请在左侧选择一张图片") -> None:
        self._pix = None
        self._coord = (0, 0)
        self._faces = []
        self._hint = hint
        self.update()

    def set_image(
        self,
        pix: QPixmap | None,
        faces: list[dict] | None = None,
        hint: str = "",
        coord_size: tuple[int, int] | None = None,
    ) -> None:
        self._pix = pix
        self._faces = faces or []
        self._hint = hint
        self._coord = coord_size or ((pix.width(), pix.height()) if pix is not None else (0, 0))
        self.update()

    def select_face(self, index: int) -> None:
        self._selected = index
        self.update()

    # ------------------------------------------------------------------ #
    def _target_rect(self) -> QRectF:
        if self._pix is None:
            return QRectF()
        pw, ph = self._pix.width(), self._pix.height()
        if pw <= 0 or ph <= 0:
            return QRectF()
        aw, ah = self.width() - 8, self.height() - 8
        scale = min(aw / pw, ah / ph)
        w, h = pw * scale, ph * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor("#161616"))

        if self._pix is None:
            p.setPen(QColor("#8a8a8a"))
            f = QFont()
            f.setPointSize(10)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self._hint)
            p.end()
            return

        r = self._target_rect()
        p.drawPixmap(r.toRect(), self._pix)

        if self._faces:
            cw, ch = self._coord if self._coord[0] else (self._pix.width(), self._pix.height())
            sx = r.width() / max(1, cw)
            sy = r.height() / max(1, ch)
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            p.setFont(font)
            fm = QFontMetrics(font)
            for f in self._faces:
                x, y, w, h = f["box"]
                rect = QRectF(r.left() + x * sx, r.top() + y * sy, w * sx, h * sy)
                ok = f.get("status") == "成功" and f.get("age") is not None
                pending = f.get("status") == "计算中"
                color = (BOX_SEL if (pending or f["index"] == self._selected)
                         else (BOX_OK if ok else BOX_FAIL))
                pen = QPen(color)
                pen.setWidth(2)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(rect)

                if ok:
                    if f.get("label"):
                        label = f["label"]
                    elif f.get("lo") is None or f.get("hi") is None:
                        label = f"#{f['index']}  {f['age']:.0f}岁"
                    else:
                        label = f"#{f['index']}  {f['age']:.0f}岁  [{f['lo']}-{f['hi']}]"
                else:
                    label = f"#{f['index']}  {f.get('status', '失败')}"
                tw = fm.horizontalAdvance(label) + 10
                th = fm.height() + 4
                ly = rect.top() - th - 2
                if ly < r.top():
                    ly = rect.top() + 2
                lrect = QRectF(rect.left(), ly, tw, th)
                p.fillRect(lrect, color)
                p.setPen(TEXT_COLOR)
                p.drawText(lrect, Qt.AlignCenter, label)
        elif self._hint:
            p.setPen(QColor("#FF9F43"))
            f = QFont()
            f.setPointSize(10)
            p.setFont(f)
            p.drawText(self.rect().adjusted(0, 0, 0, -20), Qt.AlignHCenter | Qt.AlignBottom, self._hint)
        p.end()


def image_from_bgr(arr) -> QImage:
    """numpy BGR → QImage（不复制两次，最后 copy 一次保证内存独立）。"""
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_BGR888).copy()
