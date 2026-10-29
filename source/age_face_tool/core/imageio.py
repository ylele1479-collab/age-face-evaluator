"""图片读写工具：兼容中文路径、EXIF 方向、透明通道。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..config import IMAGE_EXTS


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def imread_bgr(path: str | Path) -> np.ndarray:
    """读取图片为 BGR ndarray。

    使用 PIL 解码（对中文路径、EXIF 旋转、CMYK、PNG 透明通道都更稳妥），
    再转换为 OpenCV 使用的 BGR 顺序。
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1])


def imwrite_bgr(path: str | Path, img_bgr: np.ndarray) -> None:
    """以中文路径安全的方式写图。"""
    ok, buf = cv2_encode(img_bgr, Path(path).suffix or ".png")
    if not ok:
        raise IOError(f"编码失败: {path}")
    Path(path).write_bytes(buf.tobytes())


def cv2_encode(img_bgr: np.ndarray, ext: str = ".jpg") -> tuple[bool, np.ndarray]:
    import cv2

    return cv2.imencode(ext, img_bgr)


def make_thumbnail(path: str | Path, size: int = 96) -> np.ndarray | None:
    """生成缩略图（BGR），失败返回 None。使用 draft 降低解码开销。"""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        return np.ascontiguousarray(arr[:, :, ::-1])
    except Exception:
        return None
