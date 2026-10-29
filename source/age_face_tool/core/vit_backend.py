"""第三个候选年龄模型：ViT 年龄分类器（HuggingFace）。

- 模型：`nateraw/vit-age-classifier`（ViT-base 在 **FairFace** 上微调，
  与 UTKFace 不同源，交叉评估不会被"训练数据污染"影响结论）。
- 输出年龄类别概率，可折算成期望年龄与年龄分布（同样支持基于分布的区间）。
- 只依赖 torch + transformers（CPU）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .age_backend import AgeBackendError

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_ID = "nateraw/vit-age-classifier"


class ViTAgeBackend:
    name = "ViT-Age(FairFace, 类别概率)"
    has_distribution = True

    def __init__(self, margin: float = 0.45) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            # 已下载的模型先直接读本地缓存，避免每次启动都等待远程仓库探测。
            # 缓存缺失时才沿用首次联网下载流程，不修改用户的全局离线设置。
            try:
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
                self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID, local_files_only=True)
            except OSError:
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
                self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        except Exception as e:  # noqa: BLE001
            raise AgeBackendError(
                "无法加载 ViT 年龄模型（nateraw/vit-age-classifier，约 330MB）。\n"
                "首次使用需联网从 HuggingFace 下载一次，之后完全离线可用；\n"
                "若已下载过，请检查 %USERPROFILE%\\.cache\\huggingface 是否完好，\n"
                "或设置 HF_HUB_OFFLINE=1 强制使用本地缓存。\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e
        self.model.eval()
        self.margin = margin
        self._lock = threading.Lock()
        # 把类别标签换算成"年龄区间中点"，用于求期望年龄
        self._ages = np.array([self._mid(a) for a in self.model.config.id2label.values()], dtype=np.float64)

    @staticmethod
    def _mid(label: str) -> float:
        """把 '20-29' / '70+' / '0-2' 这类标签转成代表年龄。"""
        s = "".join(c for c in str(label) if (c.isdigit() or c == "-"))
        if "-" in s:
            lo, hi = s.split("-", 1)
            try:
                return (float(lo) + float(hi)) / 2.0 + 0.5
            except ValueError:
                return 40.0
        if "+" in str(label) or s.isdigit():
            try:
                return float(s) + 15.0
            except ValueError:
                return 40.0
        return 40.0

    # ------------------------------------------------------------------ #
    def prepare_views(self, img_bgr: np.ndarray, box, landmarks=None) -> list[tuple[str, np.ndarray]]:
        from .detector import FaceBox, expand_box

        h, w = img_bgr.shape[:2]
        x, y, bw, bh = box
        x1, y1, cw, ch = expand_box(FaceBox(x, y, bw, bh), self.margin, w, h)
        patch = img_bgr[y1 : y1 + ch, x1 : x1 + cw]
        if patch.size == 0:
            patch = img_bgr[y : y + bh, x : x + bw]
        return [("标准取景", np.ascontiguousarray(patch)),
                ("水平翻转", np.ascontiguousarray(cv2.flip(patch, 1)))]

    # ------------------------------------------------------------------ #
    def _forward_probs(self, patches: list[np.ndarray]) -> np.ndarray:
        """原始类别概率矩阵 [n, n_classes]。"""
        import torch

        rgbs = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) if p.shape[2] == 3 else p for p in patches]
        inputs = self.processor(images=rgbs, return_tensors="pt")
        with self._lock, torch.inference_mode():
            logits = self.model(**inputs).logits
            return torch.softmax(logits, dim=-1).numpy()

    def predict_probs(self, patches: list[np.ndarray]):
        """返回 (ages, probs, labels)：概率原样给出，供界面展示"模型倾向"。

        与 predict_batch 的区别是这里不做 101 类铺开，保留 FairFace 的 9 个粗类别
        （0-2 / 3-9 / 10-19 / 20-29 / 30-39 / 40-49 / 50-59 / 60-69 / 70+）。
        """
        if not patches:
            return [], np.zeros((0, 0), dtype=np.float32), []
        probs = self._forward_probs(patches)
        ages = [float(np.dot(probs[i], self._ages)) for i in range(probs.shape[0])]
        labels = [str(self.model.config.id2label[i]) for i in range(probs.shape[1])]
        return ages, probs, labels

    def predict_probs_with_features(self, patches: list[np.ndarray]):
        """同一次前向输出原年龄概率及 CLS 特征，供参考图辅助比较。"""
        import torch

        if not patches:
            return [], np.zeros((0, 0), dtype=np.float32), [], np.zeros((0, 0), dtype=np.float32)
        rgbs = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) if p.shape[2] == 3 else p for p in patches]
        inputs = self.processor(images=rgbs, return_tensors="pt")
        with self._lock, torch.inference_mode():
            features = self.model.vit(**inputs).last_hidden_state[:, 0, :]
            logits = self.model.classifier(features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            vectors = features.cpu().numpy()
        ages = [float(np.dot(row, self._ages)) for row in probs]
        labels = [str(self.model.config.id2label[i]) for i in range(probs.shape[1])]
        return ages, probs, labels, vectors

    def predict_batch(self, patches: list[np.ndarray]) -> tuple[list[float], np.ndarray | None]:
        if not patches:
            return [], None
        probs = self._forward_probs(patches)
        ages = [float(np.dot(probs[i], self._ages)) for i in range(probs.shape[0])]
        # 折算成 0~100 的 101 类"伪分布"：在每个年龄区间内均匀铺开，
        # 这样后续区间代码可以复用同一套分位数逻辑（不会伪造精度）。
        dists = np.zeros((probs.shape[0], 101), dtype=np.float32)
        classes = self.model.config.id2label
        for i in range(probs.shape[0]):
            for cls, p in enumerate(probs[i]):
                a = self._ages[cls]
                lo = max(0, int(a - 7))
                hi = min(100, int(a + 8))
                if hi <= lo:
                    hi = min(100, lo + 1)
                dists[i, lo:hi] += float(p) / (hi - lo)
            if dists[i].sum() <= 0:
                dists[i, int(round(np.clip(ages[i], 0, 100)))] = 1.0
        return ages, dists

    def describe(self) -> str:
        import torch

        dev = "GPU" if torch.cuda.is_available() else "CPU"
        return f"{self.name} · torch {torch.__version__} · {dev}"
