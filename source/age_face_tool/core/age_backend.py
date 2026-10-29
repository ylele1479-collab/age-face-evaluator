"""年龄估计后端（DeepFace / TensorFlow，纯本地推理）。

模型：DeepFace 的表观年龄模型（VGGFace 骨干 + 101 类年龄 softmax）。
- 输出的是 0~100 岁每个整数年龄的概率（101 维分布），
  期望值即 DeepFace 官方给出的年龄预测；
- 该分布可直接用于构造"基于模型概率分布"的年龄区间（进阶模式）；
- 只输出年龄，不输出性别、种族等无关属性。
- 模型只加载一次（进程级单例 + 线程锁）。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

# 减少 TensorFlow 的控制台噪声（必须在 import tensorflow 之前设置）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_MODEL_NAME = "DeepFace-Age(VGGFace/101类分布)"
AGE_CLASSES = 101
_AGE_AXIS = np.arange(AGE_CLASSES, dtype=np.float32)


class AgeBackendError(RuntimeError):
    pass


@dataclass
class ViewSpec:
    name: str
    margin: float
    flip: bool = False


DEFAULT_VIEWS: tuple[ViewSpec, ...] = (
    # 主视图：人脸框外扩 45%，接近"头肩像"取景——实测比紧贴人脸的裁剪更准
    ViewSpec("标准取景", 0.45),
    ViewSpec("水平翻转", 0.45, flip=True),
    ViewSpec("近景+30%", 0.30),
    ViewSpec("中景+60%", 0.60),
    ViewSpec("远景+75%", 0.75),
)


def apply_view(img_bgr: np.ndarray, box: tuple[int, int, int, int], spec: ViewSpec) -> np.ndarray:
    """按视图定义裁剪出一个人脸图像（超出图片边界的部分自然裁掉，不做填充）。

    取景比例（margin）对结果影响很大：实测同一张真实照片，人脸占满画面时模型会
    明显低估年龄，而按"头肩像"取景（外扩 45%）时预测更准。因此主视图使用 0.45。
    """
    from .detector import FaceBox, expand_box

    h, w = img_bgr.shape[:2]
    x, y, bw, bh = box
    x1, y1, cw, ch = expand_box(FaceBox(x, y, bw, bh), spec.margin, w, h)
    patch = img_bgr[y1 : y1 + ch, x1 : x1 + cw]
    if patch.size == 0:
        patch = img_bgr[y : y + bh, x : x + bw]
    if spec.flip:
        patch = cv2.flip(patch, 1)
    return np.ascontiguousarray(patch)


class DeepFaceAgeBackend:
    """DeepFace 年龄模型的薄封装（直接取 101 类概率分布）。

    对外接口与 ViTAgeBackend 一致，便于互换与对比：
      prepare_views(img, box, landmarks) -> [(视图名, 输入图)]
      predict_batch(patches)             -> (每张的期望年龄, (N,101) 分布或 None)
    """

    name = _MODEL_NAME
    has_distribution = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None
        self._model = None
        self._mode = ""
        self._deepface = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            from deepface import DeepFace
        except Exception as e:  # noqa: BLE001
            raise AgeBackendError(
                "无法导入 deepface / tensorflow，请确认依赖已正确安装。\n" f"原始错误: {e}"
            ) from e
        self._deepface = DeepFace
        try:
            client = DeepFace.build_model("Age", task="facial_attribute")
            self._client = client
            self._model = getattr(client, "model", None)
            if self._model is None:
                raise AgeBackendError("未取到年龄模型的底层 Keras 模型")
            self._mode = "dist"
        except Exception as e:  # noqa: BLE001
            self._mode = "analyze"
            self._load_error = str(e)

        # 预热（同时确认权重可用）
        dummy = np.full((160, 160, 3), 128, dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception as e:  # noqa: BLE001
            raise AgeBackendError(f"年龄模型预热失败: {e}") from e

    # ------------------------------------------------------------------ #
    @staticmethod
    def _prep(face_bgr: np.ndarray) -> np.ndarray:
        """与 DeepFace 官方 demography 流程一致：BGR → 224×224，不做均值归一化。"""
        if face_bgr is None or face_bgr.size == 0:
            raise AgeBackendError("空的人脸图像")
        if face_bgr.ndim == 2:
            face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

    def prepare_views(self, img_bgr: np.ndarray, box, landmarks=None,
                      views=DEFAULT_VIEWS) -> list[tuple[str, np.ndarray]]:
        """按视图定义准备输入图（取景外扩 45% 为主视图）。landmarks 不使用。"""
        out = []
        for spec in views:
            try:
                out.append((spec.name, apply_view(img_bgr, box, spec)))
            except Exception:  # noqa: BLE001
                continue
        return out

    def predict_batch(self, patches: list[np.ndarray]) -> tuple[list[float], np.ndarray | None]:
        if not patches:
            return [], None
        dist = self.predict_dist_batch(patches)
        ages = [float(age_from_dist(d)) for d in dist]
        return ages, dist

    # ------------------------------------------------------------------ #
    def predict_dist_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        """返回 (N, 101) 概率分布。"""
        if not faces:
            return np.zeros((0, AGE_CLASSES), dtype=np.float32)
        batch = np.stack([self._prep(f) for f in faces], axis=0)
        with self._lock:
            if self._mode == "dist":
                out = self._model.predict_on_batch(batch)
                out = np.asarray(out, dtype=np.float32)
                if out.ndim == 1:
                    out = out[None, :]
                if out.shape[1] != AGE_CLASSES:
                    raise AgeBackendError(f"模型输出维度异常: {out.shape}")
                return out
            # 退化路径：逐张调用 analyze（没有分布，只能给点估计）
            ages = []
            for f in faces:
                res = self._deepface.analyze(
                    f, actions=("age",), detector_backend="skip",
                    enforce_detection=False, silent=True,
                )
                if isinstance(res, list):
                    res = res[0]
                a = float(res["age"])
                d = np.zeros(AGE_CLASSES, dtype=np.float32)
                d[int(max(0, min(100, round(a))))] = 1.0
                ages.append(d)
            return np.stack(ages, axis=0)

    def predict_dist(self, face_bgr: np.ndarray) -> np.ndarray:
        return self.predict_dist_batch([face_bgr])[0]

    def predict(self, face_bgr: np.ndarray) -> float:
        """返回期望年龄（与 DeepFace 官方口径一致）。"""
        return float(age_from_dist(self.predict_dist(face_bgr)))

    # ------------------------------------------------------------------ #
    def predict_views(
        self, img_bgr: np.ndarray, box: tuple[int, int, int, int], views=DEFAULT_VIEWS
    ) -> tuple[list[float], np.ndarray]:
        """多视图预测：返回 (每个视图的期望年龄, (V,101) 分布)。兼容旧调用。"""
        patches = [p for _, p in self.prepare_views(img_bgr, box, views=views)]
        return self.predict_batch(patches)

    def describe(self) -> str:
        import tensorflow as tf

        dev = "CPU"
        try:
            if tf.config.list_physical_devices("GPU"):
                dev = "GPU"
        except Exception:  # noqa: BLE001
            pass
        extra = "" if self._mode == "dist" else "（兼容模式）"
        return f"{self.name} · TensorFlow {tf.__version__} · {dev}{extra}"


def age_from_dist(dist: np.ndarray) -> float:
    """101 类分布 → 期望年龄（DeepFace 官方 find_apparent_age 的等价实现）。"""
    d = np.asarray(dist, dtype=np.float64).ravel()
    s = d.sum()
    if s <= 0:
        return float(np.argmax(d))
    return float((d / s * _AGE_AXIS).sum())


def dist_stats(dist: np.ndarray) -> tuple[float, float]:
    """返回 (期望, 标准差)。"""
    d = np.asarray(dist, dtype=np.float64).ravel()
    s = d.sum()
    if s <= 0:
        return float(np.argmax(d)), 0.0
    p = d / s
    mean = float((p * _AGE_AXIS).sum())
    var = float((p * (_AGE_AXIS - mean) ** 2).sum())
    return mean, float(np.sqrt(max(0.0, var)))


# ---------------------------------------------------------------------- #
_backend: DeepFaceAgeBackend | None = None
_backend_lock = threading.Lock()
_backend_error: str | None = None

# ViT 后端单独一个单例（torch/transformers 体积大，按需加载）
_vit_backend = None
_vit_lock = threading.Lock()
_vit_error: str | None = None


def get_backend(choice: str = "deepface"):
    """进程级单例：模型只加载一次。choice="vit" 时返回 ViT 后端。"""
    if choice == "vit":
        return get_vit_backend()
    global _backend, _backend_error
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is None:
            try:
                _backend = DeepFaceAgeBackend()
                _backend_error = None
            except Exception as e:  # noqa: BLE001
                _backend_error = str(e)
                raise
    return _backend


def get_vit_backend():
    """ViT 年龄分类器单例（nateraw/vit-age-classifier，FairFace 微调）。"""
    global _vit_backend, _vit_error
    if _vit_backend is not None:
        return _vit_backend
    with _vit_lock:
        if _vit_backend is None:
            try:
                from .vit_backend import ViTAgeBackend

                _vit_backend = ViTAgeBackend()
                _vit_error = None
            except Exception as e:  # noqa: BLE001
                _vit_error = str(e)
                raise
    return _vit_backend


def backend_status(choice: str = "deepface") -> tuple[bool, str]:
    if choice == "vit":
        if _vit_backend is not None:
            return True, _vit_backend.describe()
        if _vit_error:
            return False, _vit_error
        return False, "ViT 年龄模型尚未加载"
    if _backend is not None:
        return True, _backend.describe()
    if _backend_error:
        return False, _backend_error
    return False, "年龄模型尚未加载"
