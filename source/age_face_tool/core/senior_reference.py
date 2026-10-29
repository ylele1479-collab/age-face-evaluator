"""用户提供的 60/70/80 示意图，只对老年段作特征与模型联合提示。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .imageio import imread_bgr


REFERENCE_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "senior_age_reference.png"


class SeniorAgeReference:
    ages = np.array([60.0, 70.0, 80.0], dtype=np.float64)

    def __init__(self, backend) -> None:
        image = imread_bgr(str(REFERENCE_IMAGE))
        if image is None:
            raise RuntimeError("年龄参考图无法读取")
        height, width = image.shape[:2]
        # 仅提取用户图中的三张主肖像，去掉标题、文字和局部示意拼图。
        rectangles = ((8, 98, 134, 253), (264, 98, 393, 253), (512, 98, 641, 253))
        patches = []
        for left, top, right, bottom in rectangles:
            crop = image[round(top * height / 599):round(bottom * height / 599),
                         round(left * width / 759):round(right * width / 759)]
            if crop.size == 0:
                raise RuntimeError("年龄参考图取景失败")
            patches.extend([np.ascontiguousarray(crop), np.ascontiguousarray(cv2.flip(crop, 1))])
        _, _, _, features = backend.predict_probs_with_features(patches)
        anchors = np.asarray(features).reshape(3, 2, -1).mean(axis=1)
        self.anchors = anchors / np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12)

    def describe(self, model_age: float, features: np.ndarray) -> str:
        if model_age < 50:
            return "60/70/80 岁参考图仅用于老年段；当前沿用原模型估计。"
        vector = np.asarray(features).mean(axis=0)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self.anchors @ vector
        if not np.isfinite(similarities).all():
            return "参考特征无法比较；沿用原模型估计。"
        order = np.argsort(similarities)[::-1]
        best = int(order[0])
        if similarities[best] < 0.5 or similarities[best] - similarities[order[1]] < 0.02:
            return "参考人像特征相近，无法区分档位；沿用原模型估计。"
        # 示例相似度不是年龄概率。原模型占主要权重，示意图仅补充档位提示。
        prior = np.exp(-0.5 * ((self.ages - model_age) / 12.0) ** 2)
        prior /= prior.sum()
        visual = np.exp((similarities - similarities.max()) / 0.07)
        visual /= visual.sum()
        joint = 0.75 * prior + 0.25 * visual
        combined = int(self.ages[int(np.argmax(joint))])
        visual_age = int(self.ages[best])
        text = f"参考辅助：结合原模型 {model_age:.1f} 岁，更接近 {combined} 岁左右参考档。"
        if abs(model_age - visual_age) > 12:
            text += f" 人像特征更接近 {visual_age} 岁示意，存在分歧，建议复核。"
        else:
            text += f" 人像特征与 {visual_age} 岁示意接近。"
        return text
