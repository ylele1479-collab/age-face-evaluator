"""旧版键盘观察器；双向同步开启时由新版筛选列表决定 A/D 的目标。

用途：用户在既有的 face_attribute_anno.exe 里按 A/D 翻页、按 S 保存时，
让复查助手跟着切换图片。实现要点：

- WH_KEYBOARD_LL 低级键盘钩子，装在独立线程的消息循环里（不占 Qt 主线程）；
- 双向同步开启时，拦截旧版的物理 A/D，交给新版选出筛选列表的上下张并同步旧版；
- 同步用的模拟 A/D 仍照常发往旧版，S 保存和其他按键原样放行；
- 只有当前台窗口标题包含目标路径、且不是本程序自己的窗口时才上报。

任何异常都只让"同步"失效，不影响主界面。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from threading import Lock

from PySide6.QtCore import QThread, Signal

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
PM_NOREMOVE = 0x0000

VK_A, VK_D, VK_S = 0x41, 0x44, 0x53
NAV_KEY_TAG = 0xA6E5

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowTextW.restype = ctypes.c_int
kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT)
user32.PeekMessageW.restype = wintypes.BOOL


WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = (WNDENUMPROC, wintypes.LPARAM)
user32.IsWindowVisible.argtypes = (wintypes.HWND,)
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101


def is_legacy_window_title(title: str) -> bool:
    """旧版编辑后会加未保存标记；仍只接受它实际使用的两个标题。"""
    return "".join(title.split()) in (
        "年龄/性别属性标注", "年龄/性别属性标注*未保存")


def window_identity(hwnd: int) -> tuple[int, int] | None:
    """只读窗口进程/线程身份；销毁后旧按键不能带到新窗口。"""
    if not hwnd or not user32.IsWindow(hwnd):
        return None
    pid = wintypes.DWORD()
    tid = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not tid:
        if not user32.IsWindow(hwnd):
            return None
        raise OSError(ctypes.get_last_error(), "无法确认旧版窗口身份")
    return int(pid.value), int(tid)


def find_window_by_title(keywords: list[str]) -> int:
    """识别旧版实际标题；排除包含程序名的资源管理器等无关窗口。"""
    hits: list[int] = []
    standard = any(k in ("属性标注", "face_attribute") for k in keywords)

    def cb(hwnd, _lparam):  # noqa: ANN001
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value
        if standard and not is_legacy_window_title(title):
            return True
        if any(k in title for k in keywords) and "复查助手" not in title:
            hits.append(int(hwnd))
        return True

    try:
        user32.EnumWindows(WNDENUMPROC(cb), 0)
    except Exception:  # noqa: BLE001
        return 0
    return hits[0] if len(hits) == 1 else 0


gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetWindowDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.GetBitmapBits.argtypes = (wintypes.HBITMAP, wintypes.LONG, ctypes.c_void_p)
user32.PrintWindow.argtypes = (wintypes.HWND, wintypes.HDC, wintypes.UINT)
user32.GetWindowDC.argtypes = (wintypes.HWND,)
gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
gdi32.CreateCompatibleBitmap.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int)
gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
gdi32.DeleteDC.argtypes = (wintypes.HDC,)
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.IsHungAppWindow.argtypes = (wintypes.HWND,)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.ShowWindowAsync.argtypes = (wintypes.HWND, ctypes.c_int)
user32.keybd_event.argtypes = (wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t)


def capture_window_image(hwnd: int):
    """读取窗口渲染图；完整声明 64 位句柄，并始终还原/释放 GDI 对象。"""
    from PIL import Image

    if not hwnd or user32.IsIconic(hwnd) or user32.IsHungAppWindow(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 100 or height < 60 or width * height > 40_000_000:
        return None
    dc = mem = bitmap = old = None
    try:
        dc = user32.GetWindowDC(hwnd)
        mem = gdi32.CreateCompatibleDC(dc)
        bitmap = gdi32.CreateCompatibleBitmap(dc, width, height)
        if not dc or not mem or not bitmap:
            return None
        old = gdi32.SelectObject(mem, bitmap)
        if not user32.PrintWindow(hwnd, mem, 2):
            return None
        data = ctypes.create_string_buffer(width * height * 4)
        if gdi32.GetBitmapBits(bitmap, len(data), data) != len(data):
            return None
        return Image.frombytes("RGB", (width, height), data.raw, "raw", "BGRX")
    finally:
        if old and mem:
            gdi32.SelectObject(mem, old)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if mem:
            gdi32.DeleteDC(mem)
        if dc:
            user32.ReleaseDC(hwnd, dc)


def capture_photo_hash(hwnd: int) -> str:
    """截原工具窗口、只取照片区域，返回哈希。

    实测：整窗哈希会因动画/光标抖动而不稳定，照片区（x 17~46%、y 12~95%）稳定。
    用于"发完按键到底动没动"的验证。
    """
    import hashlib

    if not hwnd:
        return ""
    try:
        img = capture_window_image(hwnd)
        if img is None:
            return ""
        w, h = img.size
        sub = img.crop((int(w * .17), int(h * .12), int(w * .46), int(h * .95))).tobytes()
        return hashlib.md5(sub).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


def _force_foreground(hwnd: int) -> bool:
    """把窗口切到前台。Windows 有前台锁，用 AttachThreadInput 绕过。"""
    if user32.IsIconic(hwnd):
        user32.ShowWindowAsync(hwnd, 9)  # SW_RESTORE，仅在用户明确请求跳转时调用
    if is_foreground(hwnd):
        return True
    try:
        fg = user32.GetForegroundWindow()
        user32.SetForegroundWindow(hwnd)
        if int(user32.GetForegroundWindow() or 0) == int(hwnd):
            return True
        tid_fg = user32.GetWindowThreadProcessId(fg, None)
        tid_me = kernel32.GetCurrentThreadId()
        attached = tid_fg != tid_me and user32.AttachThreadInput(tid_fg, tid_me, True)
        try:
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(tid_fg, tid_me, False)
    except Exception:  # noqa: BLE001
        pass
    return int(user32.GetForegroundWindow() or 0) == int(hwnd)


def is_foreground(hwnd: int) -> bool:
    return bool(hwnd) and int(user32.GetForegroundWindow() or 0) == int(hwnd)


def send_nav_keys_verified(hwnd: int, vk: int, times: int,
                           gap: float = 0.035) -> tuple[bool, bool]:
    """发按键并验证是否真的翻页了。

    返回 (是否发出, 画面是否变化)。画面没变通常意味着原工具焦点在输入框里、
    或窗口没响应——这时再发多少次都没用，必须让人手动点一下图片区。
    """
    import time as _t

    before = capture_photo_hash(hwnd)
    ok = send_nav_keys(hwnd, vk, times, gap=gap)
    _t.sleep(0.25 + min(1.5, times * 0.02))
    after = capture_photo_hash(hwnd)
    moved = bool(before) and bool(after) and before != after
    return ok, moved


def send_nav_keys(hwnd: int, vk: int, times: int = 1, gap: float = 0.03,
                  cancelled=lambda: False, activate: bool = True, on_sent=None) -> bool:
    """让原工具翻 times 次页（vk=0x44 前进 / 0x41 后退）。

    实测结论（2026-10-23，用窗口照片区截图哈希验证）：
    - PostMessage 到窗口或焦点控件 **无效**，原工具不响应；
    - SetForegroundWindow + keybd_event **有效**（D 换图、A 能还原）。
    因此这里必须走真实按键，代价是会短暂抢焦点。
    """
    import time as _t

    if times <= 0:
        return True
    if cancelled() or not hwnd:
        return False
    if activate and not _force_foreground(hwnd):
        return False
    try:
        for _ in range(times):
            if cancelled() or int(user32.GetForegroundWindow() or 0) != int(hwnd):
                return False
            if any(user32.GetAsyncKeyState(key) & 0x8000
                   for key in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
                return False
            user32.keybd_event(vk, 0, 0, NAV_KEY_TAG)
            user32.keybd_event(vk, 0, 2, NAV_KEY_TAG)
            if on_sent is not None:
                on_sent()
            _t.sleep(gap)
        return True
    except Exception:  # noqa: BLE001
        return False


class KeyWatcher(QThread):
    """观察旧版按键；开启导航接管时，A/D 交由新版列表处理。"""

    key = Signal(str)
    local_nav = Signal(int)
    failed = Signal(str)

    def __init__(self, match_texts, parent=None) -> None:
        super().__init__(parent)
        if isinstance(match_texts, str):
            match_texts = [match_texts]
        self._match: list[str] = [m.strip().lower() for m in match_texts if m and m.strip()]
        self._own_hwnd = 0
        self._tid = 0
        self._state_lock = Lock()
        self._stop_requested = False
        self._hook = None
        self._proc_ref = HOOKPROC(self._proc)
        self._running = False
        self._local_navigation = False
        self.seen = 0  # 钩子回调收到过的按键总数（自检用）
        self.last_hwnd = 0  # 最近一次命中的原工具窗口句柄（反向同步要用）

    def set_own_hwnd(self, hwnd: int) -> None:
        self._own_hwnd = int(hwnd or 0)

    def set_match_text(self, texts) -> None:
        if isinstance(texts, str):
            texts = [texts]
        self._match = [m.strip().lower() for m in texts if m and m.strip()]

    def set_local_navigation(self, enabled: bool) -> None:
        """持续接管旧版物理 A/D；主动同步发出的带标记按键不受影响。"""
        self._local_navigation = bool(enabled)

    # ---------------------------------------------------------------- #
    def _proc(self, n_code, w_param, l_param):  # noqa: ANN001
        try:
            if n_code == 0 and w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                kb = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                self.seen += 1
                if (kb.flags & 0x10 and int(kb.dwExtraInfo or 0) == NAV_KEY_TAG
                        and kb.vkCode in (VK_A, VK_D)
                        and int(user32.GetForegroundWindow() or 0) == self._own_hwnd):
                    # 取消同步并归还焦点的瞬间，迟到的模拟键不能让新版再多翻一次。
                    return 1
                if not kb.flags & 0x10 and kb.vkCode in (VK_A, VK_D, VK_S) and self._match:
                    hwnd = user32.GetForegroundWindow()
                    if hwnd and hwnd != self._own_hwnd:
                        buf = ctypes.create_unicode_buffer(512)
                        user32.GetWindowTextW(hwnd, buf, 512)
                        title = buf.value.lower()
                        if "复查助手" not in title and any(m in title for m in self._match):
                            self.last_hwnd = int(hwnd)
                            if (self._local_navigation and kb.vkCode in (VK_A, VK_D)
                                    and is_legacy_window_title(title)
                                    and not any(user32.GetAsyncKeyState(key) & 0x8000
                                                for key in (0x10, 0x11, 0x12, 0x5B, 0x5C))):
                                self.local_nav.emit(-1 if kb.vkCode == VK_A else 1)
                                return 1
                            self.key.emit(chr(kb.vkCode).lower())
        except Exception:  # noqa: BLE001
            pass
        return user32.CallNextHookEx(None, n_code, w_param, l_param)

    # ---------------------------------------------------------------- #
    def run(self) -> None:  # noqa: C901
        # 创建线程消息队列后才发布 tid；stop 与发布过程用同一把锁协调。
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_NOREMOVE)
        with self._state_lock:
            if self._stop_requested:
                return
            self._tid = kernel32.GetCurrentThreadId()
        self._hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc_ref, kernel32.GetModuleHandleW(None), 0)
        if not self._hook:
            self.failed.emit(f"安装键盘钩子失败（错误码 {ctypes.get_last_error()}）")
            with self._state_lock:
                self._tid = 0
            return
        self._running = True
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        if self._hook:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
        self._running = False
        with self._state_lock:
            self._tid = 0

    def stop(self) -> bool:
        self._local_navigation = False
        with self._state_lock:
            self._stop_requested = True
            if self._tid:
                user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        return self.wait(2000)
