"""年龄区间生成：基础模式（人工参考范围）与进阶模式（基于模型分布 + 保形预测校准）。

重要说明
--------
* 基础模式：区间 = 预测值 ± 用户指定宽度。这是**人为设定的参考范围**，
  没有统计意义上的覆盖率保证，界面与导出中均标注为"人工参考范围"。
* 进阶模式：模型本身输出 0~100 岁共 101 个年龄的概率分布。
  - "dist" 策略：取分布的中心分位区间 [F⁻¹(0.5-q), F⁻¹(0.5+q)]，
    q 由**带真实年龄标签的验证集**通过分裂保形预测标定：
        s_i = |0.5 - F_i(y_i)|，q = Quantile(s, ⌈(n+1)(1-α)⌉/n)
    在可交换性假设下该区间边际覆盖率 ≥ 1-α；实际覆盖率一律在独立测试集上实测。
  - "scaled"/"fixed" 策略作为对照（按多视图离散度或固定残差）。
* 所有覆盖率数字都必须来自实测，不允许凭空给出。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import AGE_MAX, AGE_MIN, INTERVAL_TYPE_BASE, INTERVAL_TYPE_CALIB

MODE_BASE = "base"
MODE_CALIBRATED = "calibrated"

# 年龄段划分（用于条件化/Mondrian 保形预测：不同年龄段误差差别很大，
# 用同一个 q 会把模型准的年龄段区间也撑得很宽）
AGE_BANDS: list[tuple[int, int]] = [
    (0, 4), (5, 9), (10, 15), (16, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 120),
]

# 区间标定实际使用的分组（比 AGE_BANDS 粗：每个段要有足够样本才能标定）。
# 10-20 合并是因为青少年样本在任一子段都不足 20；该结构随校准一起写进
# calibration.json，推理时用同一套分段，保证校准与推理一致。
GROUP_BANDS: list[tuple[int, int]] = [
    (0, 4), (5, 9), (10, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 120),
]
STRATEGY_GROUPED = "grouped"   # 按年龄段分别标定 q（Mondrian 保形预测）

STRATEGY_DIST = "dist"        # 模型概率分布的中心分位区间（推荐）
STRATEGY_SCALED = "scaled"    # 按多视图离散度归一化
STRATEGY_FIXED = "fixed"      # 固定半径

AGE_CLASSES = 101
_AGE_AXIS = np.arange(AGE_CLASSES, dtype=np.float64)


@dataclass
class Calibration:
    alpha: float = 0.10
    strategy: str = STRATEGY_DIST
    groups: list[dict] = field(default_factory=list)
    q: float = 0.35
    sigma_floor: float = 1.0
    n_calib: int = 0
    n_test: int = 0
    model: str = ""
    detector: str = "YuNet 2023mar"
    dataset: str = "UTKFace"
    created: str = ""
    metrics: dict = field(default_factory=dict)
    notes: str = ""

    # ---------------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["created"] = data["created"] or datetime.now().isoformat(timespec="seconds")
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def coverage_target(self) -> float:
        return 1.0 - self.alpha

    def band_index(self, age: float) -> int:
        for i, (lo, hi) in enumerate(AGE_BANDS):
            if lo <= age <= hi:
                return i
        return len(AGE_BANDS) - 1

    def _group_index(self, age: float) -> int:
        """按 calibration.json 里实际保存的分组边界定位（推理与校准一致）。"""
        if self.groups:
            for i, g in enumerate(self.groups):
                if float(g.get("lo", -1)) <= age <= float(g.get("hi", 120)):
                    return i
            return len(self.groups) - 1
        return _band_index(age, GROUP_BANDS)

    def band_q(self, age: float) -> float:
        """该年龄段标定出的半径；未分年龄段标定时退回全局 q。"""
        if not self.groups:
            return float(self.q)
        idx = self._group_index(age)
        if 0 <= idx < len(self.groups) and self.groups[idx].get("q") is not None:
            return float(self.groups[idx]["q"])
        return float(self.q)

    def describe(self) -> str:
        st = {STRATEGY_DIST: "模型分布分位", STRATEGY_SCALED: "多视图自适应",
              STRATEGY_FIXED: "固定残差", STRATEGY_GROUPED: "分年龄段残差"}.get(
            self.strategy, self.strategy
        )
        if self.strategy == STRATEGY_DIST:
            qtxt = f"中心分位 q={self.q:.3f}"
        elif self.strategy == STRATEGY_GROUPED and self.groups:
            qs = [g["q"] for g in self.groups if g.get("q") is not None]
            qtxt = f"分年龄段半径 {min(qs):.1f}~{max(qs):.1f} 岁（{len(qs)} 段）"
        else:
            qtxt = f"半径 q={self.q:.2f} 岁"
        return (
            f"目标覆盖率 {self.coverage_target:.0%} · 策略={st} · {qtxt} · "
            f"校准样本 {self.n_calib} · {self.dataset}"
        )


@dataclass
class AgeInterval:
    lo: int
    hi: int
    itype: str
    half_width: float

    @property
    def width(self) -> int:
        return self.hi - self.lo


def _clip(v: float) -> int:
    return int(max(AGE_MIN, min(AGE_MAX, round(v))))


def dist_cdf(dist: np.ndarray) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64).ravel()
    s = d.sum()
    if s <= 0:
        d = np.zeros(AGE_CLASSES)
        d[int(np.argmax(dist))] = 1.0
    else:
        d = d / s
    return np.cumsum(d)


def dist_quantile(dist: np.ndarray, p: float) -> int:
    """离散分布的分位数（返回年龄整数）。"""
    cdf = dist_cdf(dist)
    idx = int(np.searchsorted(cdf, min(max(p, 0.0), 1.0), side="left"))
    return int(min(AGE_MAX, max(AGE_MIN, idx)))


def cdf_at(dist: np.ndarray, age: float) -> float:
    cdf = dist_cdf(dist)
    a = int(max(0, min(AGE_CLASSES - 1, round(age))))
    return float(cdf[a])


def make_base_interval(age: float, half_width: int) -> AgeInterval:
    hw = max(0, int(half_width))
    return AgeInterval(lo=_clip(age - hw), hi=_clip(age + hw), itype=INTERVAL_TYPE_BASE, half_width=float(hw))


def make_calibrated_interval(
    age: float, sigma: float | None, dist: np.ndarray | None, calib: Calibration
) -> AgeInterval:
    if calib.strategy == STRATEGY_DIST:
        if dist is None:
            r = float(calib.q) * 20.0
            return AgeInterval(lo=_clip(age - r), hi=_clip(age + r), itype=INTERVAL_TYPE_CALIB, half_width=r)
        q = float(calib.q)
        q = min(0.5, max(0.01, q))
        lo = dist_quantile(dist, 0.5 - q)
        hi = dist_quantile(dist, 0.5 + q)
        if hi <= lo:
            hi = min(AGE_MAX, lo + 1)
        return AgeInterval(lo=lo, hi=hi, itype=INTERVAL_TYPE_CALIB, half_width=max(1.0, (hi - lo) / 2.0))
    if calib.strategy == STRATEGY_SCALED:
        s = float(sigma) if sigma is not None and math.isfinite(sigma) else 0.0
        r = float(calib.q) * (s + float(calib.sigma_floor))
    else:
        # fixed / grouped：半径按"预测年龄所在年龄段"取，误差大的年龄段区间更宽
        r = calib.band_q(age)
    r = max(1.0, r)
    return AgeInterval(lo=_clip(age - r), hi=_clip(age + r), itype=INTERVAL_TYPE_CALIB, half_width=r)


def build_interval(
    age: float,
    sigma: float | None,
    dist: np.ndarray | None,
    mode: str,
    half_width: int,
    calib: Calibration | None,
) -> tuple[AgeInterval, str | None]:
    """返回 (区间, 警告信息)。校准文件缺失时自动回退到基础模式。"""
    if mode == MODE_CALIBRATED:
        if calib is None:
            return make_base_interval(age, half_width), "未找到校准文件，已回退到人工参考范围"
        return make_calibrated_interval(age, sigma, dist, calib), None
    return make_base_interval(age, half_width), None


# ---------------------------------------------------------------------- #
def conformal_quantile(scores: list[float] | np.ndarray, alpha: float) -> float:
    """分裂保形预测的分位数（含 (n+1)/n 有限样本修正）。"""
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = arr.size
    if n == 0:
        raise ValueError("校准集为空")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    idx = min(n - 1, max(0, int(math.ceil(level * n)) - 1))
    return float(arr[idx])


def evaluate(
    records: list[dict],
    alpha: float,
    half_width: int,
    q_fixed: float,
    q_scaled: float,
    q_dist: float,
    sigma_floor: float = 1.0,
    group_qs: dict[int, float] | None = None,
    group_bands: list[tuple[int, int]] | None = None,
) -> dict:
    """在带标签的数据上实测各区间策略的覆盖率 / 平均宽度，并给出 MAE。

    records: [{"y": 真实年龄, "pred": 期望预测, "sigma": 多视图标准差, "dist": (101,) 概率}]
    group_bands: 分年龄段策略使用的分段（默认 GROUP_BANDS，须与标定时一致）。
    """
    if not records:
        return {}
    n = len(records)
    err = [abs(r["pred"] - r["y"]) for r in records]

    def _stats(getter) -> dict:
        cov = 0
        widths = []
        for r in records:
            lo, hi = getter(r)
            widths.append(hi - lo)
            if lo <= r["y"] <= hi:
                cov += 1
        return {"coverage": round(cov / n, 4), "mean_width": round(float(np.mean(widths)), 2)}

    def base(r):
        return (_clip(r["pred"] - half_width), _clip(r["pred"] + half_width))

    def conf_dist(r):
        q = min(0.5, max(0.01, q_dist))
        lo = dist_quantile(r["dist"], 0.5 - q)
        hi = dist_quantile(r["dist"], 0.5 + q)
        return (lo, max(hi, lo + 1))

    def conf_fixed(r):
        return (_clip(r["pred"] - q_fixed), _clip(r["pred"] + q_fixed))

    def conf_scaled(r):
        rad = q_scaled * (float(r.get("sigma") or 0.0) + sigma_floor)
        return (_clip(r["pred"] - rad), _clip(r["pred"] + rad))

    gq = group_qs or {}
    gb = group_bands or GROUP_BANDS

    def conf_grouped(r):
        idx = _band_index(float(r["pred"]), gb)
        rad = gq.get(idx, q_fixed)
        return (_clip(r["pred"] - rad), _clip(r["pred"] + rad))

    return {
        "n": n,
        "mae": round(float(np.mean(err)), 3),
        "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 3),
        "median_ae": round(float(np.median(err)), 3),
        "base": {"label": f"人工参考范围 ±{half_width}", **_stats(base)},
        "conformal_dist": {"label": f"校准区间(模型分布, q={q_dist:.3f})", **_stats(conf_dist)},
        "conformal_fixed": {"label": f"校准区间(固定残差, q={q_fixed:.2f}岁)", **_stats(conf_fixed)},
        "conformal_scaled": {"label": f"校准区间(多视图自适应, q={q_scaled:.2f})", **_stats(conf_scaled)},
        "conformal_grouped": {"label": "校准区间(分年龄段)", **_stats(conf_grouped)},
        "alpha": alpha,
        "target_coverage": round(1 - alpha, 4),
    }


def _band_index(age: float, bands: list[tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(bands):
        if lo <= age <= hi:
            return i
    return len(bands) - 1


def band_of(age: float) -> int:
    return _band_index(age, AGE_BANDS)


def per_band_report(records: list[dict]) -> list[dict]:
    """按**真实年龄**分段统计 MAE / 中位误差 / 样本数。

    分段依据是 y（真实年龄）而不是 pred：评估模型在某个真实年龄段的表现时，
    只有按真实年龄分组才无偏（按预测年龄分组会把误差大的样本挪到别的段）。
    """
    out = []
    for i, (lo, hi) in enumerate(AGE_BANDS):
        sel = [r for r in records if band_of(float(r["y"])) == i]
        if not sel:
            continue
        err = [abs(r["pred"] - r["y"]) for r in sel]
        out.append(
            {
                "band": f"{lo}-{hi}" if hi < 120 else f"{lo}+",
                "n": len(sel),
                "mae": round(float(np.mean(err)), 2),
                "median_ae": round(float(np.median(err)), 2),
                "mae_over5": round(float(np.mean([e for e in err if e > 5])), 2) if any(e > 5 for e in err) else 0.0,
            }
        )
    return out
