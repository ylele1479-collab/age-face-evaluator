"""核对旧版导航控件后发送系统点击，避免 A/D 落进输入框。"""
from __future__ import annotations

import ctypes
import hashlib
import re
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

from .win_hook import is_foreground, kernel32, user32, window_identity
from .sync_log import sync_log


SYNC_REVISION = "20261028.3"


user32.WindowFromPoint.argtypes = (wintypes.POINT,)
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetAncestor.restype = wintypes.HWND
user32.GetParent.argtypes = (wintypes.HWND,)
user32.GetParent.restype = wintypes.HWND
user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetWindow.restype = wintypes.HWND
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.IsWindowEnabled.argtypes = (wintypes.HWND,)
user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.GetCursorPos.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.SetThreadDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD,
                                               wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

# 静态核对的 EXE：_build 内 Panedwindow 的左、中、右三个 Frame；
# 左 Frame 顶部 navigation Frame 仅含上一张/下一张两个 ttk.Button。
_KNOWN_EXE_SHA256 = "db9033909cf85aec6698465219752ab162ccd8b6849800eae0fa96cf081d0a52"
_version_cache: dict[tuple, bool] = {}
_mouse_release_fault = False


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _InputData(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("data", _InputData)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT


@contextmanager
def _physical_coordinates():
    previous = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    if not previous:
        raise RuntimeError("无法确认旧版导航的物理坐标，未发送输入")
    try:
        yield
    finally:
        user32.SetThreadDpiAwarenessContext(previous)


def _mouse_event(flags: int, x: int = 0, y: int = 0) -> _Input:
    event = _Input()
    event.type = 0  # INPUT_MOUSE
    event.data.mi = _MouseInput(x, y, 0, flags, 0, 0xA6E6)
    return event


@_physical_coordinates()
def current_rectangle(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("无法确认旧版窗口位置")
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(80)
    user32.GetClassNameW(hwnd, buf, len(buf))
    return buf.value


def _known_executable(hwnd: int) -> bool:
    identity = window_identity(hwnd)
    if identity is None:
        return False
    process = kernel32.OpenProcess(0x1000, False, identity[0])
    if not process:
        return False
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(process, 0, buf, ctypes.byref(size)):
            return False
    finally:
        kernel32.CloseHandle(process)
    try:
        path = Path(buf.value)
        info = path.stat()
        key = (identity, str(path).casefold(), info.st_size, info.st_mtime_ns)
        if key not in _version_cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            _version_cache[key] = digest.hexdigest() == _KNOWN_EXE_SHA256
        return _version_cache[key]
    except OSError:
        return False


def _children(hwnd: int) -> list[int]:
    children = []
    seen = set()
    child = user32.GetWindow(hwnd, 5)  # GW_CHILD
    while child and int(child) not in seen:
        seen.add(int(child))
        if user32.IsWindowVisible(child):
            children.append(int(child))
        child = user32.GetWindow(child, 2)  # GW_HWNDNEXT
    return children


def _known_layout_pair(hwnd: int) -> dict[int, tuple[int, int, int]]:
    """从已核对版本的三栏布局找到左栏顶部双按钮，不靠文字坐标命中。"""
    outer = current_rectangle(hwnd)
    pending = [hwnd]
    visited = set()
    pairs = {}
    layout = []
    while pending and len(visited) < 4096:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        children = _children(parent)
        pending.extend(children)
        if len(children) != 3 or any(_class_name(child) != "TkChild" for child in children):
            continue
        area = current_rectangle(parent)
        if area[2] < outer[2] * 0.6 or area[3] < outer[3] * 0.5:
            continue
        panes = sorted(((current_rectangle(child), child) for child in children), key=lambda pair: pair[0][0])
        if (any(rect[3] < area[3] * 0.7 for rect, _ in panes)
                or max(rect[1] for rect, _ in panes) - min(rect[1] for rect, _ in panes) > area[3] * 0.08):
            continue
        left_rect, left = panes[0]
        if not 0 < left_rect[2] < area[2] * 0.45:
            continue
        containers = [(current_rectangle(child), child) for child in _children(left)]
        if not containers:
            continue
        # 已核对的 _build 按 pack 顺序先放导航，再放搜索、图片列表。
        # 搜索栏也可能只有两个可见 HWND，不能仅凭“双控件”识别导航。
        top = min(rect[1] for rect, _ in containers)
        topmost = [(rect, child) for rect, child in containers if rect[1] == top]
        layout.append((left, [(child, rect) for rect, child in containers]))
        if len(topmost) != 1:
            continue
        for navigation_rect, navigation in topmost:
            buttons = _children(navigation)
            layout.append((navigation, navigation_rect,
                           [(button, current_rectangle(button)) for button in buttons]))
            if len(buttons) != 2:
                continue
            if any(_class_name(button) != "TkChild" or _children(button)
                   or not user32.IsWindowEnabled(button) for button in buttons):
                continue
            ordered = sorted(((current_rectangle(button), button) for button in buttons), key=lambda pair: pair[0][0])
            a, b = ordered[0][0], ordered[1][0]
            if (min(a[2], b[2], a[3], b[3]) <= 0
                    or abs(a[2] - b[2]) > max(4, min(a[2], b[2]) * 0.2)
                    or abs(a[1] - b[1]) > max(8, min(a[3], b[3]) * 0.3)
                    or max(a[3], b[3]) > left_rect[3] * 0.16
                    or a[1] - left_rect[1] > left_rect[3] * 0.18
                    or a[0] + a[2] > b[0] + 2):
                continue
            pair = {direction: (button, rect[0] + rect[2] // 2 - outer[0],
                                          rect[1] + rect[3] // 2 - outer[1])
                    for direction, (rect, button) in zip((0x41, 0x44), ordered)}
            pairs[tuple(pair[direction][0] for direction in (0x41, 0x44))] = pair
    if len(pairs) != 1:
        sync_log("navigation_layout_unconfirmed", candidates=len(pairs),
                 windows=len(visited), navigation_children=layout)
        raise RuntimeError("未能确认该版本的左栏导航控件，未发送导航")
    return next(iter(pairs.values()))


@_physical_coordinates()
def native_navigation_snapshot(hwnd: int) -> dict | None:
    if not _known_executable(hwnd):
        return None
    rect = current_rectangle(hwnd)
    pair = _known_layout_pair(hwnd)
    if current_rectangle(hwnd) != rect:
        raise RuntimeError("旧版窗口正在移动或缩放，请重试同步")
    sync_log("navigation_controls_confirmed", method="native_layout",
             previous=pair[0x41][0], following=pair[0x44][0])
    return {"rect": rect, "native_pair": pair}


def _direction(text: str) -> int | None:
    value = re.sub(r"\s", "", text)
    if "上一张" in value or "上ー张" in value or re.fullmatch(r"[\[({（【]?[Aa][\])}）】]?", value):
        return 0x41
    if "下一张" in value or "下ー张" in value or re.fullmatch(r"[\[({（【]?[Dd][\])}）】]?", value):
        return 0x44
    return None


def _confirmed_pair(hwnd: int, snapshot: dict) -> dict[int, tuple[int, int, int]]:
    """文字必须落在同一父容器的两个独立可见叶控件上。"""
    rect = tuple(snapshot.get("rect", []))
    if len(rect) != 4 or current_rectangle(hwnd) != rect:
        raise RuntimeError("旧版布局已改变，请重试同步")
    if "native_pair" in snapshot:
        pair = _known_layout_pair(hwnd)
        if pair != snapshot["native_pair"]:
            raise RuntimeError("旧版导航布局已变化，请重试同步")
        return pair
    points = {0x41: [], 0x44: []}
    for word in snapshot.get("words", []):
        direction = _direction(str(word.get("text", "")))
        if direction is None:
            continue
        x = int(float(word["x"]) + float(word["width"]) / 2)
        y = int(float(word["y"]) + float(word["height"]) / 2)
        handle = int(user32.WindowFromPoint(wintypes.POINT(rect[0] + x, rect[1] + y)) or 0)
        if (not handle or int(user32.GetAncestor(handle, 2) or 0) != hwnd
                or not user32.IsWindowVisible(handle) or not user32.IsWindowEnabled(handle)
                or _class_name(handle) != "TkChild" or user32.GetWindow(handle, 5)):
            continue
        points[direction].append((handle, x, y))
    pairs = []
    for previous in points[0x41]:
        for following in points[0x44]:
            parent = user32.GetParent(previous[0])
            if (previous[0] == following[0] or not parent
                    or parent != user32.GetParent(following[0])
                    or previous[1] >= following[1] or abs(previous[2] - following[2]) > 18):
                continue
            children = []
            child = user32.GetWindow(parent, 5)
            while child:
                if user32.IsWindowVisible(child):
                    children.append(int(child))
                child = user32.GetWindow(child, 2)
            if set(children) == {previous[0], following[0]}:
                pairs.append((previous, following))
    unique = {(a[0], b[0]): (a, b) for a, b in pairs}
    if len(unique) != 1:
        raise RuntimeError("未能唯一确认旧版“上一张/下一张”按钮，未发送导航")
    previous, following = next(iter(unique.values()))
    return {0x41: previous, 0x44: following}


@_physical_coordinates()
def send_navigation_button(hwnd: int, vk: int, snapshot: dict, *, cancelled, on_sent) -> bool:
    pair = _confirmed_pair(hwnd, snapshot)
    handles = tuple(pair[direction][0] for direction in (0x41, 0x44))
    if snapshot.setdefault("button_handles", handles) != handles:
        raise RuntimeError("旧版导航控件已重建，请重新打开助手后同步")
    button, x, y = pair[vk]
    return _send_control_click(hwnd, button,
        (snapshot["rect"][0] + x, snapshot["rect"][1] + y), tuple(snapshot["rect"]),
        cancelled=cancelled, on_sent=on_sent)


@_physical_coordinates()
def _send_control_click(hwnd, button, point, rect, *, cancelled, on_sent=lambda: None):
    global _mouse_release_fault
    if _mouse_release_fault:
        raise RuntimeError("鼠标释放未确认，自动导航已停用；请手动点击并松开左键后重新打开助手")
    if cancelled() or not is_foreground(hwnd):
        return False
    if any(user32.GetAsyncKeyState(key) & 0x8000
           for key in (0x01, 0x02, 0x04, 0x10, 0x11, 0x12, 0x5B, 0x5C)):
        return False
    identity = window_identity(hwnd)
    point = wintypes.POINT(*point)
    if int(user32.WindowFromPoint(point) or 0) != button:
        raise RuntimeError("旧版导航按钮被遮挡，未发送输入")
    cursor = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(cursor)):
        return False
    if (cancelled() or not is_foreground(hwnd) or identity != window_identity(hwnd)
            or current_rectangle(hwnd) != rect):
        return False
    left, top, width, height = (user32.GetSystemMetrics(index) for index in (76, 77, 78, 79))
    if width <= 1 or height <= 1:
        return False

    def normalized(p):
        if not left <= p.x < left + width or not top <= p.y < top + height:
            raise RuntimeError("旧版按钮或鼠标在可用桌面之外，未发送输入")
        return round((p.x - left) * 65535 / (width - 1)), round((p.y - top) * 65535 / (height - 1))

    target_x, target_y = normalized(point)
    restore_x, restore_y = normalized(cursor)
    move_flags = 0x0001 | 0x8000 | 0x4000  # MOVE | ABSOLUTE | VIRTUALDESK
    events = (_Input * 4)(_mouse_event(move_flags, target_x, target_y),
                          _mouse_event(0x0002), _mouse_event(0x0004),
                          _mouse_event(move_flags, restore_x, restore_y))
    current = wintypes.POINT()
    if (not user32.GetCursorPos(ctypes.byref(current)) or (current.x, current.y) != (cursor.x, cursor.y)
            or cancelled() or not is_foreground(hwnd) or identity != window_identity(hwnd)
            or current_rectangle(hwnd) != rect
            or int(user32.WindowFromPoint(point) or 0) != button
            or any(user32.GetAsyncKeyState(key) & 0x8000
                   for key in (0x01, 0x02, 0x04, 0x10, 0x11, 0x12, 0x5B, 0x5C))):
        return False
    # TkWinGetModifierState 依赖 GetKeyState；PostMessage 不会产生真实按键状态。
    # 系统输入按顺序完成移动、按下、释放、还原，不向画布或编辑框发送操作。
    sent = int(user32.SendInput(len(events), events, ctypes.sizeof(_Input)))
    released = True
    if sent == 2:
        # 紧急释放优先于记录在途、日志或鼠标移动，并核对系统确实接收释放。
        release = (_Input * 1)(_mouse_event(0x0004))
        released = int(user32.SendInput(1, release, ctypes.sizeof(_Input))) == 1
        if not released:
            _mouse_release_fault = True
    if sent >= 2:
        on_sent()
    if sent != len(events) and released:
        current = wintypes.POINT()
        if (sent and user32.GetCursorPos(ctypes.byref(current))
                and abs(current.x - point.x) <= 1 and abs(current.y - point.y) <= 1):
            restore = (_Input * 1)(_mouse_event(move_flags, restore_x, restore_y))
            user32.SendInput(1, restore, ctypes.sizeof(_Input))
    sync_log("navigation_input", method="SendInput", hwnd=hwnd, button=button,
             count=sent, released=released)
    if not released:
        raise RuntimeError("系统未确认鼠标释放，自动导航已停用；请手动点击并松开左键后重新打开助手")
    if sent != len(events):
        raise RuntimeError(f"系统未完整接收旧版导航输入（{sent}/{len(events)}）")
    return True


@_physical_coordinates()
def native_search_snapshot(hwnd: int) -> dict | None:
    """Identify the known EXE's left search frame and image list structurally."""
    if not _known_executable(hwnd):
        return None
    rect = current_rectangle(hwnd)
    navigation = _known_layout_pair(hwnd)
    nav_parent = user32.GetParent(navigation[0x41][0])
    left = user32.GetParent(nav_parent)
    left_rect = current_rectangle(left)
    nav_rect = current_rectangle(nav_parent)
    children = _children(left)
    lists = [child for child in children if not _children(child)
             and current_rectangle(child)[2] >= left_rect[2] * .65
             and current_rectangle(child)[3] >= left_rect[3] * .35
             and _class_name(child) == 'TkChild']
    if len(lists) != 1:
        raise RuntimeError('未能确认旧版图片列表，未执行直接定位')
    list_handle = lists[0]
    list_rect = current_rectangle(list_handle)
    candidates = []
    for child in children:
        child_rect = current_rectangle(child)
        if child == nav_parent or not nav_rect[1] + nav_rect[3] - 2 <= child_rect[1] < list_rect[1]:
            continue
        leaves = _children(child)
        if not 2 <= len(leaves) <= 3:
            continue
        if any(_children(leaf) or _class_name(leaf) != 'TkChild'
               or not user32.IsWindowEnabled(leaf) for leaf in leaves):
            continue
        ordered = sorted(leaves, key=lambda leaf: current_rectangle(leaf)[0])
        entry = max(ordered, key=lambda leaf: current_rectangle(leaf)[2])
        clear = ordered[-1]
        entry_rect, clear_rect = current_rectangle(entry), current_rectangle(clear)
        if (entry == clear or entry_rect[2] < left_rect[2] * .35
                or clear_rect[0] < entry_rect[0] + entry_rect[2]
                or abs(entry_rect[1] - clear_rect[1]) > max(entry_rect[3], clear_rect[3])
                or max(entry_rect[3], clear_rect[3]) > left_rect[3] * .15):
            continue
        candidates.append((entry, clear, entry_rect, clear_rect))
    if len(candidates) != 1:
        raise RuntimeError('未能唯一确认旧版搜索框，未执行直接定位')
    entry, clear, entry_rect, clear_rect = candidates[0]
    if current_rectangle(hwnd) != rect:
        raise RuntimeError('旧版窗口正在移动，已停止定位')
    return {'rect': rect, 'entry_rect': entry_rect,
            'controls': {'entry': entry, 'clear': clear, 'list': list_handle},
            'control_rects': {'entry': entry_rect, 'clear': clear_rect, 'list': list_rect}}


@_physical_coordinates()
def click_search_control(hwnd, snapshot, name, *, cancelled, on_sent=lambda: None):
    if cancelled() or not is_foreground(hwnd) or snapshot is None:
        return False
    current = native_search_snapshot(hwnd)
    if (current is None or current['rect'] != snapshot['rect']
            or current['controls'] != snapshot['controls']):
        raise RuntimeError('旧版搜索布局已变化，已停止定位')
    control = current['controls'][name]
    x, y, width, height = current['control_rects'][name]
    row_offset = max(6, round(10 * current['entry_rect'][3] / 24))
    point = (x + width // 2, y + min(height // 2, row_offset) if name == 'list' else y + height // 2)
    return _send_control_click(hwnd, control, point, tuple(current['rect']),
                               cancelled=cancelled, on_sent=on_sent)


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('flags', wintypes.DWORD),
                ('hwndActive', wintypes.HWND), ('hwndFocus', wintypes.HWND),
                ('hwndCapture', wintypes.HWND), ('hwndMenuOwner', wintypes.HWND),
                ('hwndMoveSize', wintypes.HWND), ('hwndCaret', wintypes.HWND),
                ('rcCaret', wintypes.RECT)]


user32.GetGUIThreadInfo.argtypes = (wintypes.DWORD, ctypes.POINTER(_GuiThreadInfo))
user32.GetGUIThreadInfo.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
user32.ClientToScreen.restype = wintypes.BOOL


@_physical_coordinates()
def _search_has_caret(hwnd, entry_rect):
    identity = window_identity(hwnd)
    if identity is None:
        return False
    info = _GuiThreadInfo()
    info.cbSize = ctypes.sizeof(info)
    if (not user32.GetGUIThreadInfo(identity[1], ctypes.byref(info)) or not info.hwndCaret
            or int(user32.GetAncestor(info.hwndCaret, 2) or 0) != hwnd):
        return False
    point = wintypes.POINT(info.rcCaret.left, info.rcCaret.top)
    if not user32.ClientToScreen(info.hwndCaret, ctypes.byref(point)):
        return False
    x, y, width, height = entry_rect
    return x <= point.x < x + width and y <= point.y < y + height


def _keyboard_event(vk, *, scan=0, flags=0):
    event = _Input()
    event.type = 1
    event.data.ki = _KeyboardInput(vk, scan, flags, 0, 0xA6E8)
    return event


def type_search_query(hwnd, snapshot, query, *, cancelled, wait):
    import time
    if not query or len(query) > 260:
        return False
    # Tk retains OS focus on the toplevel; caret position identifies the Entry.
    deadline = time.monotonic() + .3
    while not cancelled() and time.monotonic() < deadline:
        if is_foreground(hwnd) and _search_has_caret(hwnd, snapshot['entry_rect']):
            break
        wait(.01)
    else:
        return False
    if (cancelled() or not is_foreground(hwnd)
            or current_rectangle(hwnd) != tuple(snapshot['rect'])
            or any(user32.GetAsyncKeyState(key) & 0x8000
                   for key in (0x01, 0x02, 0x04, 0x10, 0x11, 0x12, 0x53, 0x5B, 0x5C))):
        return False
    events = [_keyboard_event(0x24), _keyboard_event(0x24, flags=2),
              _keyboard_event(0x10), _keyboard_event(0x23),
              _keyboard_event(0x23, flags=2), _keyboard_event(0x10, flags=2),
              _keyboard_event(0x08), _keyboard_event(0x08, flags=2)]
    encoded = query.encode('utf-16-le')
    for offset in range(0, len(encoded), 2):
        scan = int.from_bytes(encoded[offset:offset+2], 'little')
        events.extend((_keyboard_event(0, scan=scan, flags=4),
                       _keyboard_event(0, scan=scan, flags=6)))
    batch = (_Input * len(events))(*events)
    sent = int(user32.SendInput(len(events), batch, ctypes.sizeof(_Input)))
    if sent != len(events):
        release = (_Input * 1)(_keyboard_event(0x10, flags=2))
        user32.SendInput(1, release, ctypes.sizeof(_Input))
        raise RuntimeError('搜索输入未完整送达，已停止定位')
    return True
