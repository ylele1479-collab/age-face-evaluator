"""单张图片的处理流水线：读取 → 检测 → 年龄推断（主模型/对比模型）→ 区间生成。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .age_backend import DEFAULT_VIEWS, DeepFaceAgeBackend, age_from_dist, dist_stats, get_backend
from .detector import FaceBox, FaceDetector, suppress_annotations
from .imageio import imread_bgr
from .interval import Calibration, build_interval

ST_OK = "成功"
ST_NO_FACE = "未检测到人脸"
ST_READ_FAIL = "读取失败"
ST_PARTIAL = "部分人脸失败"
ST_FAIL = "处理失败"

MODEL_DEEPFACE = "deepface"
MODEL_VIT = "vit"
ALL_MODELS = (MODEL_DEEPFACE, MODEL_VIT)
MODEL_LABELS = {
    MODEL_DEEPFACE: "DeepFace 表观年龄模型（101 类概率分布）",
    MODEL_VIT: "ViT 年龄分类器（FairFace 微调，类别概率）",
}


def other_model(choice: str) -> str:
    """对比模型 = 另一个模型。"""
    return MODEL_VIT if choice == MODEL_DEEPFACE else MODEL_DEEPFACE


@dataclass
class FaceResult:
    index: int
    box: tuple[int, int, int, int]
    det_score: float = 0.0
    age: float | None = None                 # 主模型预测年龄
    views: list[float] = field(default_factory=list)
    sigma: float | None = None
    dist_sigma: float | None = None
    lo: int = 0
    hi: int = 0
    itype: str = ""
    half: float = 0.0
    status: str = ST_OK
    error: str = ""
    # 第二个模型（对比用）
    age2: float | None = None
    views2: list[float] = field(default_factory=list)
    model2: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "box": list(self.box),
            "det_score": round(float(self.det_score), 4),
            "age": None if self.age is None else round(float(self.age), 2),
            "views": [round(float(v), 2) for v in self.views],
            "sigma": None if self.sigma is None else round(float(self.sigma), 3),
            "dist_sigma": None if self.dist_sigma is None else round(float(self.dist_sigma), 3),
            "lo": self.lo,
            "hi": self.hi,
            "itype": self.itype,
            "half": round(float(self.half), 2),
            "status": self.status,
            "error": self.error,
            "age2": None if self.age2 is None else round(float(self.age2), 2),
            "views2": [round(float(v), 2) for v in self.views2],
            "model2": self.model2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FaceResult":
        return cls(
            index=int(d["index"]),
            box=tuple(int(v) for v in d["box"]),  # type: ignore[arg-type]
            det_score=float(d.get("det_score", 0)),
            age=d.get("age"),
            views=list(d.get("views", [])),
            sigma=d.get("sigma"),
            dist_sigma=d.get("dist_sigma"),
            lo=int(d.get("lo", 0)),
            hi=int(d.get("hi", 0)),
            itype=d.get("itype", ""),
            half=float(d.get("half", 0)),
            status=d.get("status", ST_OK),
            error=d.get("error", ""),
            age2=d.get("age2"),
            views2=list(d.get("views2", [])),
            model2=d.get("model2", ""),
        )


@dataclass
class ImageResult:
    path: str
    rel: str
    width: int = 0
    height: int = 0
    faces: list[FaceResult] = field(default_factory=list)
    status: str = ST_OK
    error: str = ""
    elapsed: float = 0.0
    warning: str = ""
    mtime: float = 0.0
    size: int = 0

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "rel": self.rel,
            "width": self.width,
            "height": self.height,
            "faces": [f.to_dict() for f in self.faces],
            "status": self.status,
            "error": self.error,
            "elapsed": round(self.elapsed, 3),
            "warning": self.warning,
            "mtime": self.mtime,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageResult":
        return cls(
            path=d["path"],
            rel=d.get("rel", Path(d["path"]).name),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            faces=[FaceResult.from_dict(f) for f in d.get("faces", [])],
            status=d.get("status", ST_OK),
            error=d.get("error", ""),
            elapsed=float(d.get("elapsed", 0)),
            warning=d.get("warning", ""),
            mtime=float(d.get("mtime", 0)),
            size=int(d.get("size", 0)),
        )


@dataclass
class PipelineConfig:
    mode: str = "base"
    half_width: int = 4
    use_tta: bool = False
    calib: Calibration | None = None
    model: str = MODEL_DEEPFACE
    compare_with: str | None = None

    def needs_tta(self) -> bool:
        return bool(self.use_tta)

    @property
    def primary_label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)


def _estimate(img: np.ndarray, box: FaceBox, backend, use_tta: bool):
    """返回 (每个视图的年龄, 分布或 None)。"""
    if isinstance(backend, DeepFaceAgeBackend):
        views = DEFAULT_VIEWS if use_tta else (DEFAULT_VIEWS[0],)
        prepared = backend.prepare_views(img, box.as_tuple(), views=views)
    else:
        prepared = backend.prepare_views(img, box.as_tuple(), box.landmarks)
    patches = [p for _, p in prepared]
    if not patches:
        raise RuntimeError("模型输入为空")
    return backend.predict_batch(patches)


class Pipeline:
    def __init__(
        self,
        detector: FaceDetector,
        backend,
        config: PipelineConfig,
        second_backend=None,
    ) -> None:
        self.detector = detector
        self.backend = backend
        self.second = second_backend
        self.config = config

    # ------------------------------------------------------------------ #
    def process(self, path: str | Path) -> ImageResult:
        p = Path(path)
        t0 = time.perf_counter()
        try:
            stat = p.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = 0.0, 0
        res = ImageResult(path=str(p), rel=p.name, mtime=mtime, size=size)

        try:
            img = imread_bgr(p)
        except Exception as e:  # noqa: BLE001
            res.status = ST_READ_FAIL
            res.error = f"图片无法解码: {type(e).__name__}: {e}"
            res.elapsed = time.perf_counter() - t0
            return res

        res.height, res.width = img.shape[0], img.shape[1]
        try:
            cleaned = suppress_annotations(img)
            boxes = self.detector.detect(img, cleaned=cleaned)
            if cleaned is not None and len(self.detector.detect(cleaned)) == len(boxes):
                img = cleaned
        except Exception as e:  # noqa: BLE001
            res.status = ST_FAIL
            res.error = f"人脸检测异常: {type(e).__name__}: {e}"
            res.elapsed = time.perf_counter() - t0
            return res

        if not boxes:
            res.status = ST_NO_FACE
            res.elapsed = time.perf_counter() - t0
            return res

        failed = 0
        for i, box in enumerate(boxes, start=1):
            fr = self._process_face(img, box, i)
            if fr.status != ST_OK:
                failed += 1
            res.faces.append(fr)
        if failed:
            res.status = ST_PARTIAL if failed < len(boxes) else ST_FAIL
        res.elapsed = time.perf_counter() - t0
        return res

    # ------------------------------------------------------------------ #
    def _process_face(self, img: np.ndarray, box: FaceBox, index: int) -> FaceResult:
        cfg = self.config
        fr = FaceResult(index=index, box=box.as_tuple(), det_score=box.score)

        # ---- 主模型 ----
        try:
            ages, dists = _estimate(img, box, self.backend, cfg.needs_tta())
            if not ages:
                raise RuntimeError("模型未返回预测结果")
            fr.views = ages
            primary = dists[0] if dists is not None else None
            fr.age = float(ages[0])
            fr.sigma = float(np.std(ages, ddof=0)) if len(ages) > 1 else None
            if primary is not None:
                fr.dist_sigma = float(dist_stats(primary)[1])
        except Exception as e:  # noqa: BLE001
            fr.status = ST_FAIL
            fr.error = f"年龄推断失败: {type(e).__name__}: {e}"
            return fr
        if fr.age is None or not math.isfinite(fr.age):
            fr.status = ST_FAIL
            fr.error = "年龄推断结果无效"
            return fr

        # ---- 第二个模型（对比）----
        if self.second is not None:
            try:
                ages2, _ = _estimate(img, box, self.second, use_tta=False)
                if ages2:
                    fr.age2 = float(ages2[0])
                    fr.views2 = ages2
                    fr.model2 = getattr(self.second, "name", "second")
            except Exception as e:  # noqa: BLE001
                fr.error = f"对比模型失败: {type(e).__name__}: {e}"

        itv, warn = build_interval(fr.age, fr.sigma, primary, cfg.mode, cfg.half_width, cfg.calib)
        fr.lo, fr.hi, fr.itype, fr.half = itv.lo, itv.hi, itv.itype, itv.half_width
        if warn:
            fr.error = f"{fr.error} | {warn}" if fr.error else warn
        return fr


def load_backend(choice: str):
    """按名称加载后端（进程级单例，重复调用不重复加载权重）。"""
    if choice == MODEL_VIT:
        from .age_backend import get_vit_backend

        return get_vit_backend()
    return get_backend()
