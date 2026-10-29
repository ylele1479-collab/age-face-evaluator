"""复查辅助工具的数据层：扫描用户拖进来的文件/文件夹，读取同名 json 的候选年龄，
再用人脸模型给出"模型估计年龄 + 类别倾向"。

**只读**：本模块只读图片与 json，不写任何文件（日志/导出另由界面处理）。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

GENDER_TEXT = {1: "男", 2: "女"}

# 文件名形如 451200900000007201439-451226200609060034-20160602.jpg
# 第二段是 18 位身份证号，第 17 位奇数为男、偶数为女
_ID_RE = re.compile(r"^(\d{15,25})-(\d{17}[\dXx])-(\d{8})\.", re.I)


def valid_age(value) -> bool:
    """与旧版一致：年龄为非负、有限的数字，不能是布尔值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def parse_box(value) -> tuple[float, float, float, float] | None:
    """无效框交给人脸检测器处理，不影响其他图片的导入或更新。"""
    if not isinstance(value, dict) or not value:
        return None
    try:
        box = tuple(float(value.get(k, 0)) for k in ("left", "top", "width", "height"))
        return box if all(math.isfinite(v) for v in box) else None
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass
class ReviewItem:
    uid: int
    image_path: str
    json_path: str | None = None
    face_index: int = 0
    box: tuple[float, float, float, float] | None = None
    # json 里的候选
    ori: float | None = None
    ts: float | None = None
    cake: float | None = None
    cake_gender: int | None = None
    anno: float | None = None
    anno_gender: int | None = None
    anno_gender_source: str = ""
    # 运行期结果
    pred: float | None = None
    probs: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    status: str = "待计算"          # 待计算 / 完成 / 无人脸 / 失败
    error: str = ""
    seen: bool = False
    pick: int | None = None         # 用户按 1/2/3 标记的候选（仅内存，不落盘）
    box_signature: str = "null"   # 推理使用的原 JSON 框，区别于 detector 补出的运行期框。
    reference_hint: str = ""

    @property
    def fill_gender(self) -> int:
        """沿用已有数据；均缺失时采用用户设定的默认男，不作图像判断。"""
        for value in (self.anno_gender, self.cake_gender):
            if value in (1, 2):
                return int(value)
        return 1

    @property
    def fill_gender_source(self) -> str:
        if self.anno_gender in (1, 2):
            if self.anno_gender_source == "user_default_male" and self.anno_gender == 1:
                return "用户设定默认男（未经图像判断）"
            if self.anno_gender_source == "cake_default":
                return "已接受 Cake 候选"
            return "已有标注"
        if self.cake_gender in (1, 2):
            return "已有 Cake 候选"
        return "用户设定默认男（未经图像判断）"

    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        return Path(self.image_path).name

    @property
    def candidates(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for label, v in (("原始", self.ori), ("TS", self.ts), ("Cake", self.cake)):
            if valid_age(v):
                out.append((label, float(v)))
        return out

    def nearest(self) -> tuple[str, float, float] | None:
        """返回离模型估计最近的候选 (名称, 值, 差值)。"""
        if self.pred is None or not self.candidates:
            return None
        label, value = min(self.candidates, key=lambda kv: abs(kv[1] - self.pred))
        return label, value, abs(value - self.pred)

    # ---------------------------------------------------------------- #
    @property
    def saved(self) -> bool:
        """json 里已经有 anno_age，即原工具里已保存（原工具显示为绿色）。"""
        return self.anno is not None

    @property
    def sex_conflict(self) -> bool:
        """有效的已标注性别与 Cake 性别不一致，仅作已有数据对照。"""
        return (self.anno_gender in (1, 2) and self.cake_gender in (1, 2)
                and self.anno_gender != self.cake_gender)

    @property
    def id_gender(self) -> int | None:
        """身份证号第 17 位推出的性别（1 男 / 2 女）。

        仅用于已有字段对照，不能验证照片中人物的性别。
        身份证的出生日期**不能**用来推照片年龄（文件名里的日期不是拍照日期）。
        """
        m = _ID_RE.match(Path(self.image_path).name)
        if not m:
            return None
        try:
            d = int(m.group(2)[16])
        except (IndexError, ValueError, TypeError):
            return None
        return 1 if d % 2 == 1 else 2

    @property
    def gender_conflict(self) -> bool:
        """文件名身份证性别位与有效的 Cake/标注性别不同，不作图像判断。"""
        g = self.id_gender
        if g is None:
            return False
        for v in (self.cake_gender, self.anno_gender):
            if v in (1, 2) and v != g:
                return True
        return False

    @property
    def data_gender_conflict(self) -> bool:
        """任意已有性别字段冲突；未发现冲突不代表已验证正确。"""
        return self.sex_conflict or self.gender_conflict

    @property
    def gender_review_reasons(self) -> list[str]:
        """已有数据冲突、缺失标注或默认来源；单独缺少候选字段不算异常。"""
        reasons: list[str] = []
        if self.sex_conflict:
            reasons.append("标注性别与 Cake 候选不一致")
        if self.gender_conflict:
            reasons.append("文件名身份证性别与已有字段不一致")
        if self.anno_gender not in (1, 2):
            reasons.append("标注性别缺失或无有效值")
        if self.anno_gender_source == "user_default_male" and self.anno_gender == 1:
            reasons.append("标注性别来自用户设定默认男，待人工确认")
        elif (self.anno_gender_source == "cake_default" and self.anno_gender in (1, 2)
              and (self.cake_gender not in (1, 2) or self.anno_gender == self.cake_gender)):
            reasons.append("标注性别沿用 Cake 候选，待人工确认")
        return reasons

    @property
    def gender_needs_review(self) -> bool:
        """合并数据冲突、缺失标注和仍沿用默认值的复核入口。"""
        return bool(self.gender_review_reasons)

    @property
    def baseline(self) -> tuple[float, str] | None:
        """沿用旧版年龄优先级：已标注、Cake、原始、TS。"""
        for value, label in ((self.anno, "已标注"), (self.cake, "Cake"),
                             (self.ori, "原始"), (self.ts, "TS")):
            if valid_age(value):
                return float(value), label
        return None

    @property
    def err(self) -> float | None:
        """模型预测 − 基准值（带符号，正数=模型估大）。"""
        b = self.baseline
        if self.pred is None or b is None:
            return None
        return float(self.pred) - b[0]

    @property
    def err_i(self) -> int | None:
        """误差取整（界面上的分级与显示都用这个，避免 +2.4 显示 +2 却判成黄）。"""
        e = self.err
        return None if e is None else int(round(e))

    def issues(self) -> list[str]:
        """需要人再确认的点。只列事实，不假装能判对错。"""
        out: list[str] = []
        if self.sex_conflict:
            out.append("已标注性别与 Cake 不一致")
        if self.gender_conflict:
            out.append(f"身份证性别({GENDER_TEXT.get(self.id_gender, '?')})与 json 不一致")
        e = self.err_i
        if e is not None and abs(e) > 10:
            out.append(f"模型与基准差 {abs(e)} 岁（>10）")
        return out

    def bucket_probs(self) -> tuple[float, float]:
        """(十几岁 10-19 的概率, 二十几岁 20-29 的概率)。"""
        if not self.probs:
            return 0.0, 0.0
        p_teen = p_twenty = 0.0
        for lab, p in zip(self.labels, self.probs):
            if lab == "10-19":
                p_teen = float(p)
            elif lab == "20-29":
                p_twenty = float(p)
        return p_teen, p_twenty

    def verdict(self) -> tuple[str, str]:
        """给界面用的一句话结论与等级：('ok'|'warn'|'info', 文本)。

        口径来自本机在用户真实标注数据上的实测（1575 条）：
        - 总体 MAE 2.3 岁，误差 ≤5 岁占 90%；
        - 但 15~25 岁段系统性低估，20 岁分界几乎无分辨力（平衡准确 55~59%）。
        因此这里绝不输出"模型倾向十几/二十几"这类结论。
        """
        if self.status != "完成" or self.pred is None:
            return "info", self.status if self.status != "失败" else f"失败：{self.error}"
        if 15 <= self.pred <= 25:
            return "warn", "模型落在 15~25 岁段：这一段它系统性偏低、几乎分不开十几与二十几，请以你自己判断为准"
        near = self.nearest()
        if near and near[2] >= 8:
            return "warn", f"模型与三个候选都差 {near[2]:.0f} 岁以上，建议复核"
        return "info", "模型估计仅供参考（本机实测平均误差约 2.3 岁）"


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def collect_images(paths: list[str], recursive: bool = True) -> list[str]:
    """把用户给的文件/文件夹展开成图片路径列表（去重、保持稳定顺序）。"""
    found: list[str] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        found.append(str(p))

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    add(f)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            add(p)
    return found


def build_items(images: list[str], start_uid: int = 0) -> list[ReviewItem]:
    """每张图 → 若干人脸条目；同名 json 存在时带出候选年龄与检测框。"""
    items: list[ReviewItem] = []
    uid = start_uid
    for img in images:
        jp = Path(str(img) + ".json")
        obj = _read_json(jp) if jp.exists() else None
        results = (obj or {}).get("result") or []
        if not isinstance(results, list):
            results = []
        if results:
            for i, r in enumerate(results):
                r = r if isinstance(r, dict) else {}
                box = parse_box(r.get("box"))
                items.append(ReviewItem(
                    uid=uid, image_path=img, json_path=str(jp), face_index=i, box=box,
                    ori=r.get("ori_age"), ts=r.get("ts_age"), cake=r.get("cake_age"),
                    cake_gender=r.get("cake_gender"),
                    anno=r.get("anno_age"), anno_gender=r.get("anno_gender"),
                    anno_gender_source=r.get("anno_gender_source", ""),
                    box_signature=json.dumps(r.get("box"), sort_keys=True, ensure_ascii=False),
                ))
                uid += 1
        else:
            # 路径即使暂不存在也要监听，旧版之后新建标注时才能刷新。
            items.append(ReviewItem(uid=uid, image_path=img, json_path=str(jp), face_index=0))
            uid += 1
    return items
