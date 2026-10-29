"""Single-target navigation through the legacy tool's own search and list."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JumpResult:
    ok: bool
    message: str


def unique_search_query(paths, target):
    """Find a short, unique substring using the entire confirmed directory."""
    paths = tuple(paths)
    if target not in paths:
        return None
    folded = [str(path).replace('\\', '/').casefold() for path in paths]
    name = Path(target).stem
    tokens = re.findall(r'[0-9A-Za-z]+', name)
    candidates = []
    for size in (6, 8, 10, 12):
        for token in tokens:
            if len(token) >= size:
                candidates.extend((token[-size:], token[:size]))
    candidates.append(Path(target).name)
    for query in dict.fromkeys(candidates):
        needle = query.casefold()
        if query and sum(needle in path for path in folded) == 1:
            return query
    # Computing a common Windows root for thousands of paths is comparatively
    # costly. It is needed only for duplicate basenames, not the normal route.
    try:
        root = os.path.commonpath([os.path.dirname(path) for path in paths])
        query = os.path.relpath(target, root).replace('\\', '/')
        needle = query.casefold()
        if query and sum(needle in path for path in folded) == 1:
            return query
    except ValueError:
        pass
    return None


def search_query_matches(snapshot, query):
    """Read back only OCR words drawn inside the confirmed search entry."""
    rect, entry = snapshot.get('rect', ()), snapshot.get('entry_rect', ())
    if len(rect) != 4 or len(entry) != 4:
        return False
    x, y, width, height = entry
    words = []
    for word in snapshot.get('words', ()):
        center_x = rect[0] + float(word['x']) + float(word['width']) / 2
        center_y = rect[1] + float(word['y']) + float(word['height']) / 2
        if x <= center_x < x + width and y <= center_y < y + height:
            words.append(word)
    text = ''.join(str(word.get('text', '')) for word in sorted(words, key=lambda w: w['x']))
    normalize = lambda value: ''.join(c for c in value.casefold() if c.isalnum())
    actual, expected = normalize(text), normalize(query)
    if expected.isdecimal():
        actual = actual.translate(str.maketrans({'o': '0', 'i': '1', 'l': '1', 'g': '9'}))
    return actual == expected


def perform_direct_jump(query, target, sender, read_position, *, cancelled=lambda: False,
                        on_sent=lambda: None, wait=time.sleep, now=time.monotonic):
    """One selection, verified arrival, then restore the full legacy list."""
    result = JumpResult(False, '直接定位已取消')
    if cancelled():
        return result
    try:
        if not sender.prepare():
            return JumpResult(False, '未确认旧版搜索入口，未执行直接定位')
        if cancelled() or not sender.type_query(query):
            result = JumpResult(False, '搜索输入未完成，未打开图片')
        else:
            query_deadline = now() + 1.5
            matched = False
            while not cancelled() and now() < query_deadline:
                if sender.query_matches(query):
                    matched = True
                    break
                wait(.02)
            if not matched or cancelled():
                result = JumpResult(False, '未核对到目标搜索内容，未打开图片')
            elif not sender.select_target(on_sent):
                result = JumpResult(False, '旧版未接收目标选择')
            else:
                deadline, consecutive = now() + 4, 0
                while not cancelled() and now() < deadline:
                    position = read_position()
                    if position is not None and position.image_path == target:
                        consecutive += 1
                        if position.counter_verified or consecutive >= 2:
                            result = JumpResult(True, '已直接定位并核对目标图片')
                            break
                    else:
                        consecutive = 0
                    wait(.02)
                if not result.ok:
                    result = JumpResult(False, '旧版尚未确认目标图片，已停止定位')
    except Exception as exc:
        result = JumpResult(False, f'直接定位失败：{exc}')
    finally:
        if sender.dirty:
            try:
                if not sender.clear_query():
                    result = JumpResult(False, '定位搜索尚未清空；再次选择目标可恢复')
            except Exception as exc:
                result = JumpResult(False, f'恢复旧版图片列表失败：{exc}')
    if result.ok and not cancelled():
        # Clearing a filter must not change the selected image.
        position = read_position()
        if position is None or position.image_path != target:
            return JumpResult(False, '清空搜索后目标尚未确认，未报告同步成功')
    return result if not cancelled() else JumpResult(False, '直接定位已取消')


class NativeJumpSender:
    """Known-layout controls; all Windows input stays inside the legacy UI."""
    def __init__(self, hwnd, reader, cancelled, wait):
        self.hwnd, self.reader = hwnd, reader
        self.cancelled, self.wait = cancelled, wait
        self.snapshot = None
        self.identity = None
        self.dirty = False

    def prepare(self):
        from .legacy_buttons import native_search_snapshot
        from .win_hook import window_identity
        self.snapshot = native_search_snapshot(self.hwnd)
        self.identity = window_identity(self.hwnd)
        return self.snapshot is not None

    def type_query(self, query):
        from .legacy_buttons import click_search_control, type_search_query
        if not click_search_control(self.hwnd, self.snapshot, 'entry', cancelled=self.cancelled):
            return False
        # Track the change before typing so partial input is also cleaned.
        self.dirty = True
        return type_search_query(self.hwnd, self.snapshot, query,
                                 cancelled=self.cancelled, wait=self.wait)

    def query_matches(self, query):
        from .legacy_buttons import current_rectangle
        if current_rectangle(self.hwnd) != tuple(self.snapshot['rect']):
            return False
        result = self.reader.search_words(self.hwnd)
        result['entry_rect'] = self.snapshot['entry_rect']
        return search_query_matches(result, query)

    def select_target(self, on_sent):
        from .legacy_buttons import click_search_control
        return click_search_control(self.hwnd, self.snapshot, 'list',
                                    cancelled=self.cancelled, on_sent=on_sent)

    def clear_query(self):
        from .legacy_buttons import click_search_control, native_search_snapshot
        from .win_hook import window_identity
        if self.identity is None or window_identity(self.hwnd) != self.identity:
            return False
        fresh = native_search_snapshot(self.hwnd)
        if fresh is None:
            return False
        self.snapshot = fresh
        result = click_search_control(self.hwnd, self.snapshot, 'clear', cancelled=lambda: False)
        if result:
            deadline = time.monotonic() + .5
            while time.monotonic() < deadline:
                if self.query_matches(''):
                    self.dirty = False
                    return True
                self.wait(.01)
        return False
