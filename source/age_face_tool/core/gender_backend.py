"""性别分类后端（DeepFace 的 VGGFace 性别模型，2 类：Woman / Man）。

只做"男/女"两类，用于在复查时和 json 里的性别做对比，**不作为性别结论**：
训练数据以成人为主，本数据集主体是儿童，准确率会明显下降，
实测数字见 reports/real_data_eval.md 的性别一节。

取景与年龄模型一致：人脸框外扩 45%；输入按 DeepFace 官方流程缩放到 224×224。
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

SIZE = (224, 224)


class GenderBackend:
    name = "DeepFace-Gender(VGGFace)"
    MALE, FEMALE = 1, 2   # 与 json 里的编码对齐：1=男 2=女

    def __init__(self) -> None:
        from deepface.models.demography.tf.Gender import GenderClient

        self.model = GenderClient()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def predict_patches(self, patches: list[np.ndarray]) -> tuple[float, float] | None:
        """返回 (p_female, p_male)，多视图取均值。patches 为 BGR。"""
        if not patches:
            return None
        acc = np.zeros(2, dtype=np.float64)
        n = 0
        with self._lock:
            for p in patches:
                if p.size == 0:
                    continue
                small = cv2.resize(p, SIZE, interpolation=cv2.INTER_AREA)
                out = self.model.predict(small)
                acc += np.asarray(out, dtype=np.float64).ravel()[:2]
                n += 1
        if n == 0:
            return None
        acc /= n
        # GENDER_LABELS = ['Woman', 'Man']
        return float(acc[0]), float(acc[1])

    def describe(self) -> str:
        import tensorflow as tf

        return f"{self.name} · TensorFlow {tf.__version__} · CPU"
