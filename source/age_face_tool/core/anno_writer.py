"""把"默认值"写回标注 json（用户点「一键保存」时用）。

规则（与用户确认过）：
- **只处理还没有 anno 的条目**（界面上的黑色行），已有 anno 的一律不碰；
- 年龄采用 Cake；缺失性别采用已有 Cake 候选或用户设定默认男，记录来源；
- 写之前把每个原文件**完整备份**到 `%LOCALAPPDATA%\\AgeFaceTool\\anno_backup\\<时间戳>\\`；
- 保持原文件的 2 空格缩进 + **CRLF** 换行（实测原文件就是 \\r\\n），不重排字段顺序；
- 先写临时文件再原子替换，写完重新解析一遍做校验。
"""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
from pathlib import Path

from .review_data import ReviewItem, valid_age


def backup_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "AgeFaceTool" / "anno_backup"


def _dumps_like_original(obj: dict) -> str:
    """按原文件风格序列化：2 空格缩进 + CRLF + 结尾换行。"""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return text.replace("\n", "\r\n") + "\r\n"


def _fill_missing_gender(record: dict) -> bool:
    """仅填缺失项，返回是否使用用户设定默认男；不从图像推断性别。"""
    gender = record.get("cake_gender")
    use_default = gender not in (1, 2)
    record["anno_gender"] = 1 if use_default else int(gender)
    record["anno_gender_source"] = "user_default_male" if use_default else "cake_default"
    return use_default


def is_gender_save_candidate(item: ReviewItem) -> bool:
    """预期标注路径也可能尚未创建，不能把它当成可写的标注。"""
    return bool(item.json_path and item.anno_gender not in (1, 2)
                and Path(item.json_path).is_file())


def save_defaults(items: list[ReviewItem], progress_cb=None, *, max_error: int | None = None,
                  gender_only: bool = False) -> dict:
    """把未保存条目的 anno 写成 Cake 默认值。

    返回 {"written": n, "skipped": n, "errors": [(path, msg)], "backup_dir": str}
    """
    if gender_only:
        todo = [it for it in items if is_gender_save_candidate(it)]
    else:
        todo = [it for it in items
                if it.anno is None and it.json_path and valid_age(it.cake)]
    result = {"written": 0, "skipped": len(items) - len(todo), "errors": [],
              "backup_dir": "", "mapping": {}, "written_hashes": {}, "default_genders": 0}
    if not todo:
        return result

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = backup_root()
    root.mkdir(parents=True, exist_ok=True)
    bdir = Path(tempfile.mkdtemp(prefix=stamp + "_", dir=root))
    result["backup_dir"] = str(bdir)

    # 同一张图可能有多个 face，按 json 分组，一个文件只读一次写一次
    by_file: dict[str, list[ReviewItem]] = {}
    for it in todo:
        by_file.setdefault(it.json_path, []).append(it)

    for i, (jp, its) in enumerate(by_file.items()):
        p = Path(jp)
        tmp = None
        try:
            raw = p.read_bytes()
            obj = json.loads(raw.decode("utf-8"))
            res = obj.get("result") or []
            changed = 0
            default_genders = 0
            for it in its:
                if it.face_index >= len(res):
                    continue
                r = res[it.face_index]
                if json.dumps(r.get("box"), sort_keys=True, ensure_ascii=False) != it.box_signature:
                    continue
                if gender_only:
                    if r.get("anno_gender") in (1, 2):
                        continue
                    default_genders += int(_fill_missing_gender(r))
                    changed += 1
                    continue
                if r.get("anno_age") is not None or not valid_age(r.get("cake_age")):
                    continue
                if max_error is not None:
                    # 用写入时的 Cake 值复核误差；同一文件只修改符合范围的脸。
                    # 若旧版在计算后改了框或候选值，跳过，避免沿用失效的预测。
                    if (it.status != "完成" or it.pred is None
                            or r.get("cake_gender") != it.cake_gender
                            or r.get("cake_age") != it.cake
                            or abs(int(round(float(it.pred) - float(r["cake_age"])))) > max_error):
                        continue
                r["anno_age"] = r["cake_age"]
                if r.get("anno_gender") not in (1, 2):
                    default_genders += int(_fill_missing_gender(r))
                changed += 1
            if not changed:
                continue
            # 备份（保留目录名，避免不同目录同名文件冲突）
            path_tag = hashlib.sha256(str(p.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
            bname = f"{path_tag}__{p.name}"
            # 备份实际用于本次计算的原始字节，而不是再次读取正在被编辑的文件。
            (bdir / bname).write_bytes(raw)
            with tempfile.NamedTemporaryFile(mode="wb", prefix=p.name + ".", suffix=".tmp",
                                             dir=p.parent, delete=False) as output:
                tmp = Path(output.name)
                written = _dumps_like_original(obj).encode("utf-8")
                output.write(written)
            json.loads(tmp.read_text(encoding="utf-8"))   # 校验能解析
            if p.read_bytes() != raw:
                # 已有其他保存时不覆盖；也不把该文件加入本次可撤销清单。
                raise RuntimeError("文件在保存过程中已被其他程序修改，请刷新后再试")
            os.replace(tmp, p)                            # 原子替换
            tmp = None
            result["mapping"][bname] = str(p)
            result["written_hashes"][str(p)] = hashlib.sha256(written).hexdigest()
            result["written"] += changed
            result["default_genders"] += default_genders
        except Exception as e:  # noqa: BLE001
            result["errors"].append((jp, f"{type(e).__name__}: {e}"))
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        if progress_cb and i % 20 == 0:
            progress_cb(i + 1, len(by_file))
    return result


def restore_from_map(backup_dir: str, mapping: dict[str, str], *, expected_hashes=None) -> dict:
    """按 {备份文件名: 原始绝对路径} 恢复。"""
    out = {"restored": 0, "restored_paths": [], "errors": []}
    bdir = Path(backup_dir)
    for bname, orig in mapping.items():
        src = bdir / bname
        tmp = None
        try:
            if not src.is_file():
                out["errors"].append((orig, "备份缺失"))
                continue
            destination = Path(orig)
            current = destination.read_bytes()
            expected = (expected_hashes or {}).get(orig)
            if expected is None or hashlib.sha256(current).hexdigest() != expected:
                raise RuntimeError("保存后文件已被修改，未覆盖后续标注")
            with tempfile.NamedTemporaryFile(mode="wb", prefix=destination.name + ".undo.",
                                             suffix=".tmp", dir=destination.parent, delete=False) as output:
                tmp = Path(output.name)
                output.write(src.read_bytes())
            if destination.read_bytes() != current:
                raise RuntimeError("撤销期间文件发生变化，未覆盖")
            os.replace(tmp, destination)
            tmp = None
            out["restored"] += 1
            out["restored_paths"].append(orig)
        except Exception as e:  # noqa: BLE001
            out["errors"].append((orig, f"{type(e).__name__}: {e}"))
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    return out
