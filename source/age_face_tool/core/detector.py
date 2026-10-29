"""人脸检测：YuNet(ONNX) + 标注抑制 + 多尺度融合。

设计要点
--------
1. 使用 OpenCV 自带的 YuNet 检测器（轻量 ONNX，CPU 实时），首次运行自动下载。
2. 对"已带黄色检测框 / 年龄文字"的图片，额外在**抑制标注后**的副本上再检测一次，
   两次结果做 NMS 合并，降低标注干扰。
3. 多尺度检测，兼顾大图里的中等脸和小图里的小脸。
4. 只做几何合理性过滤（长宽比、最小尺寸），不使用图片中的任何文字信息。
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import YUNET_FILENAME, YUNET_URLS, models_dir


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    # YuNet 输出的 5 个关键点：右眼、左眼、鼻尖、右嘴角、左嘴角（图像坐标）
    landmarks: np.ndarray | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return max(1, self.w * self.h)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        d = {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(float(self.score), 4)}
        if self.landmarks is not None:
            d["landmarks"] = [[round(float(a), 1), round(float(b), 1)] for a, b in self.landmarks]
        return d


def ensure_yunet_model(progress_cb=None) -> Path:
    """确保 YuNet 模型存在；不存在则下载。返回本地路径。"""
    dst = models_dir() / YUNET_FILENAME
    if dst.exists() and dst.stat().st_size > 200_000:
        return dst
    last_err = None
    for url in YUNET_URLS:
        try:
            if progress_cb:
                progress_cb(f"正在下载人脸检测模型: {url}")
            tmp = dst.with_suffix(".part")
            req = urllib.request.Request(url, headers={"User-Agent": "AgeFaceTool/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size < 200_000:
                tmp.unlink(missing_ok=True)
                raise IOError("下载文件过小，疑似失败")
            tmp.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(
        f"无法获取人脸检测模型 {YUNET_FILENAME}（请检查网络或手动放到 {models_dir()}）：{last_err}"
    )


def suppress_annotations(img_bgr: np.ndarray) -> np.ndarray | None:
    """把"纯色标注"（黄色框、彩色文字）尽量抹掉，供第二次检测使用。

    只在标注像素占比合理时返回处理后的图，否则返回 None（表示无需处理）。
    该结果仅作为**额外的检测输入**，不会替换原图，因此误判风险很低。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 黄色 / 橙色标注（检测框、年龄文字最常用的颜色）
    m_yellow = cv2.inRange(hsv, (18, 120, 140), (40, 255, 255))
    # 纯红 / 纯绿（部分标注工具）
    m_red = cv2.inRange(hsv, (0, 160, 130), (8, 255, 255)) | cv2.inRange(hsv, (170, 160, 130), (179, 255, 255))
    m_green = cv2.inRange(hsv, (45, 140, 120), (85, 255, 255))
    mask = m_yellow | m_red | m_green
    frac = float(mask.mean()) / 255.0
    if frac < 0.0008 or frac > 0.40:
        return None
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    try:
        return cv2.inpaint(img_bgr, mask, 4, cv2.INPAINT_TELEA)
    except cv2.error:
        return None


def _nms(boxes: list[FaceBox], iou_thr: float = 0.35) -> list[FaceBox]:
    if not boxes:
        return []
    arr = np.array([[b.x, b.y, b.w, b.h] for b in boxes], dtype=np.float32)
    scores = np.array([b.score for b in boxes], dtype=np.float32)
    idxs = scores.argsort()[::-1]
    keep: list[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(arr[i, 0], arr[rest, 0])
        yy1 = np.maximum(arr[i, 1], arr[rest, 1])
        xx2 = np.minimum(arr[i, 0] + arr[i, 2], arr[rest, 0] + arr[rest, 2])
        yy2 = np.minimum(arr[i, 1] + arr[i, 3], arr[rest, 1] + arr[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = arr[i, 2] * arr[i, 3] + arr[rest, 2] * arr[rest, 3] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        idxs = rest[iou < iou_thr]
    return [boxes[i] for i in keep]


class FaceDetector:
    """线程安全性：每个工作线程应各自持有一个实例（OpenCV DNN 句柄非线程安全）。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        min_face_px: int = 24,
        max_long_side: int = 1600,
    ) -> None:
        self.model_path = Path(model_path) if model_path else ensure_yunet_model()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_px = min_face_px
        self.max_long_side = max_long_side
        self._det = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), score_threshold, nms_threshold, 5000
        )
        self.last_variants: list[str] = []

    # ------------------------------------------------------------------ #
    def _detect_one_scale(self, img: np.ndarray, scale: float) -> list[FaceBox]:
        h, w = img.shape[:2]
        if scale != 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        else:
            nw, nh = w, h
            resized = img
        self._det.setInputSize((nw, nh))
        try:
            _, faces = self._det.detect(resized)
        except cv2.error:
            return []
        out: list[FaceBox] = []
        if faces is None:
            return out
        inv = 1.0 / scale
        for f in faces:
            x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            score = float(f[-1])
            if score < self.score_threshold:
                continue
            lm = np.asarray(f[4:14], dtype=np.float32).reshape(5, 2) * inv
            out.append(
                FaceBox(
                    x=int(round(x * inv)),
                    y=int(round(y * inv)),
                    w=int(round(bw * inv)),
                    h=int(round(bh * inv)),
                    score=score,
                    landmarks=lm,
                )
            )
        return out

    def _scales_for(self, w: int, h: int) -> list[float]:
        long_side = max(w, h)
        if long_side > self.max_long_side:
            base = self.max_long_side / long_side
        elif long_side < 900:
            base = min(2.0, 900.0 / long_side)  # 小图放大，提升小脸召回
        else:
            base = 1.0
        scales = [base]
        up = base * 1.6
        if up <= 2.5 and long_side * up <= self.max_long_side * 1.6:
            scales.append(up)
        return scales

    # ------------------------------------------------------------------ #
    def detect(self, img_bgr: np.ndarray, cleaned: np.ndarray | None = None) -> list[FaceBox]:
        """返回按"从上到下、从左到右"排序的人脸框列表。

        cleaned: 可选的"已抑制标注"图像；调用方重复使用同一张图时会先算好传入，
        避免重复做 inpaint。
        """
        h, w = img_bgr.shape[:2]
        variants: list[tuple[str, np.ndarray]] = [("原图", img_bgr)]
        if cleaned is None:
            cleaned = suppress_annotations(img_bgr)
        if cleaned is not None:
            variants.append(("去标注", cleaned))

        raw: list[FaceBox] = []
        used: list[str] = []
        for vname, vimg in variants:
            scales = self._scales_for(w, h)
            for si, s in enumerate(scales):
                found = self._detect_one_scale(vimg, s)
                if found:
                    used.append(f"{vname}x{s:.2f}")
                    raw.extend(found)
                    # 基准尺度已经检出人脸时，不再尝试放大尺度（省时）
                    if si == 0 and len(scales) > 1:
                        break
        self.last_variants = used

        merged = _nms(raw, iou_thr=0.35)
        merged = [self._clip(b, w, h) for b in merged if self._valid(b)]
        merged.sort(key=lambda b: (int(b.cy / max(1.0, h * 0.22)), b.cx))
        return merged

    def _valid(self, b: FaceBox) -> list[FaceBox]:
        if b.w < self.min_face_px or b.h < self.min_face_px:
            return []
        ratio = b.w / float(b.h)
        if ratio < 0.45 or ratio > 2.2:
            return []
        return [b]

    @staticmethod
    def _clip(b: FaceBox, w: int, h: int) -> FaceBox:
        x1 = max(0, min(b.x, w - 1))
        y1 = max(0, min(b.y, h - 1))
        x2 = max(x1 + 1, min(b.x + b.w, w))
        y2 = max(y1 + 1, min(b.y + b.h, h))
        return FaceBox(x1, y1, x2 - x1, y2 - y1, b.score, b.landmarks)


def expand_box(b: FaceBox, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    """按比例外扩人脸框（用于给人脸模型提供一点上下文）。"""
    cx, cy = b.cx, b.cy
    nw, nh = b.w * (1 + 2 * margin), b.h * (1 + 2 * margin)
    x1 = int(round(max(0, cx - nw / 2)))
    y1 = int(round(max(0, cy - nh / 2)))
    x2 = int(round(min(w, cx + nw / 2)))
    y2 = int(round(min(h, cy + nh / 2)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return img[y : y + h, x : x + w].copy()


def scale_hint(box: FaceBox, img_w: int) -> str:
    """人脸尺寸分档（仅用于界面提示，不影响推断）。"""
    r = box.w / max(1, img_w)
    if r < 0.05:
        return "小"
    if r < 0.15:
        return "中"
    return "大"


def fmt_seconds(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        return "-"
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} 秒"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m} 分 {s} 秒"
    hh, mm = divmod(m, 60)
    return f"{hh} 时 {mm} 分"
