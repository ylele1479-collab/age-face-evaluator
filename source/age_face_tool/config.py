"""全局配置与路径常量。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "人脸年龄区间推断工具"
APP_ID = "AgeFaceTool"
VERSION = "1.1.0"

# 支持的图片格式（按要求：JPG / JPEG / PNG / WEBP，额外兼容 BMP）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 年龄区间默认设置
DEFAULT_HALF_WIDTH = 4
MIN_HALF_WIDTH = 1
MAX_HALF_WIDTH = 15
AGE_MIN, AGE_MAX = 0, 100

# 模型文件
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

# 区间类型文案
INTERVAL_TYPE_BASE = "人工参考范围"
INTERVAL_TYPE_CALIB = "校准区间(保形预测)"


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    p = Path(base) / APP_ID
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_dir() -> Path:
    p = app_data_dir() / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p


def calibration_file() -> Path:
    return app_data_dir() / "calibration.json"


def project_root() -> Path:
    """返回项目根目录（源码运行 / PyInstaller 打包两种情况都可用）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_export_path() -> Path:
    return Path(os.path.expanduser("~")) / "Desktop" / "年龄区间结果.xlsx"
