"""旧版窗口的只读位置识别与后台闭环导航；不访问旧进程内存或修改其文件。"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .win_hook import _force_foreground, find_window_by_title, is_foreground, window_identity
from .legacy_buttons import send_navigation_button
from .sync_log import sync_log


def _normalized(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


@dataclass(frozen=True)
class LegacyPosition:
    image_path: str
    index: int                 # 旧版显示的编号，从 1 开始（按图片，而非人脸）
    total: int
    counter_verified: bool = True  # 保存提示没有编号时，仅文件名经过确认。


class PositionCatalog:
    def __init__(self, paths):
        self.paths = tuple(dict.fromkeys(str(p) for p in paths))
        self.groups: dict[str, list[str]] = {}
        for path in self.paths:
            self.groups.setdefault(os.path.normcase(os.path.dirname(path)), []).append(path)
        for group in self.groups.values():
            group.sort(key=lambda p: Path(p).name)
        # 新版 load_folder 对递归得到的完整 Path 排序；保留旧版分目录顺序作候选。
        self._all_paths = sorted(self.paths, key=Path)
        self.names = {p: _normalized(Path(p).name) for p in self.paths}
        self._ocr_names = {p: self._ocr_name_key(Path(p).name) for p in self.paths}
        self._ocr_suffixes: dict[str, list[str]] = {}
        for path, name in self._ocr_names.items():
            if len(name) >= 11:
                self._ocr_suffixes.setdefault(name[-11:], []).append(path)
        self._paths_by_name: dict[str, list[str]] = {}
        self._groups_by_total: dict[int, list[list[str]]] = {}
        self._groups_by_path: dict[str, list[list[str]]] = {}
        self._last_verified_group: list[str] | None = None
        self._filename_cache: dict[str, str] = {}
        for path, name in self.names.items():
            self._paths_by_name.setdefault(name, []).append(path)
        sequences: set[tuple[str, ...]] = set()
        for group in [self._all_paths, *self.groups.values()]:
            sequence = tuple(group)
            if not sequence or sequence in sequences:
                continue
            sequences.add(sequence)
            self._groups_by_total.setdefault(len(group), []).append(group)
            for path in group:
                self._groups_by_path.setdefault(path, []).append(group)

    @staticmethod
    def _image_caption(text: str) -> str:
        # save 显示 doc.path.name，即 *.jpg.json；OCR 也可能丢掉 json 前的点。
        # 只接受明确的图片扩展名 + 末尾 JSON，不任意删除文件名里的 json 字样。
        return re.sub(
            r"((?:\.\s*)?(?:j\s*p\s*e?\s*g|p\s*n\s*g|w\s*e\s*b\s*p|b\s*m\s*p))"
            r"\s*(?:\.\s*)?j\s*s\s*o\s*n\s*$",
            lambda match: re.sub(r"\s+", "", match[1]), text.strip(), flags=re.IGNORECASE)

    @staticmethod
    def _ocr_name_key(text: str) -> str:
        # 只供计数已限定候选后的 OCR 对照；原始文件名精确匹配仍优先。
        return _normalized(text).translate(str.maketrans({
            "l": "1", "i": "1", "o": "0", "g": "9", "一": None, "ー": None}))

    def _unique_suffix_match(self, ocr_name: str, path: str, score: float) -> bool:
        """编号已限定候选后，处理文件名前部被 OCR 误读的情况。"""
        name = self._ocr_names[path]
        if score < 0.80 or len(name) < 11 or len(ocr_name) < 11:
            return False
        suffix = name[-11:]
        if not ocr_name.endswith(suffix) or self._ocr_suffixes.get(suffix) != [path]:
            return False
        # 后缀必须唯一，且全目录中不能存在得分更接近 OCR 的其他文件。
        return all(self._ratio_below(ocr_name, other_name, score - 0.035)
                   for other_path, other_name in self._ocr_names.items() if other_path != path)

    @staticmethod
    def _ratio_below(a: str, b: str, threshold: float) -> bool:
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        # 两种快速值都是真实相似度的上界；严格小于阈值才可提前拒绝。
        # 上界达到阈值时仍计算原始 ratio，保留文件名歧义判断的边界。
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            return True
        return matcher.ratio() < threshold

    def resolve(self, text: str) -> LegacyPosition | None:
        """只有编号、总数、文件名共同确定唯一序列时才信任计数。"""
        text = self._image_caption(text)
        # 不把 10g5 / 1742 的局部 5 / 1742 当作真实计数。
        match = re.match(r"\s*(\d{1,7})\s*[/／]\s*([0-9OoIl|gGqQ \t]{1,15})(?![\w])", text)
        if not match:
            return self._resolve_filename_only(text)
        total_text = "".join(match[2].split()).translate(str.maketrans({
            "O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
            "g": "9", "G": "9", "q": "9", "Q": "9"}))
        if not total_text.isdecimal():
            return self._resolve_filename_only(text)
        index, total = int(match[1]), int(total_text)
        if not 1 <= index <= total:
            return self._resolve_filename_only(text)
        name = _normalized(text[match.end():])
        if not name:
            return None
        # 总数必须对应实际完整目录，计数只在该目录中限定当前位置候选。
        groups = self._groups_by_total.get(total, [])
        if not groups:
            return None
        candidates = list(dict.fromkeys(group[index - 1] for group in groups))
        exact = self._paths_by_name.get(name, [])
        if exact:
            coherent = [path for path in candidates if path in exact]
            if len(coherent) != 1:
                # 清晰文件名与编号矛盾时，不能退回“仅文件名”把第6张误报成第5张。
                return None
            path = coherent[0]
        else:
            ocr_name = self._ocr_name_key(text[match.end():])
            scores = sorted(((SequenceMatcher(None, ocr_name, self._ocr_names[p], autojunk=False).ratio(), p)
                             for p in candidates), reverse=True)
            if not scores:
                return None
            best_score, path = scores[0]
            if best_score < 0.92 and not self._unique_suffix_match(ocr_name, path, best_score):
                return None
            # 比较同一编号在不同目录序列中的歧义；相邻编号不应推翻清楚的计数。
            if len(scores) > 1 and best_score - scores[1][0] < 0.035:
                return None
        position = LegacyPosition(path, index, total)
        group = self.group_for(position)
        if group is None:
            return None
        self._last_verified_group = group
        return position

    def _resolve_filename_only(self, text: str) -> LegacyPosition | None:
        # 旧版保存会将底部编号替换为“已保存全部 N 张人脸：文件名”。
        # 不依赖中文提示能否被 OCR 识别，但文件名必须在完整目录中唯一。
        text = self._image_caption(text)
        name = _normalized(text)
        # 空白或纯标点的 OCR 结果不可能匹配普通文件名；避免逐文件做模糊匹配。
        # 保留规范化后为空的特殊文件名原有处理逻辑。
        if not name and "" not in self._paths_by_name:
            return None
        cached_path = self._filename_cache.get(name)
        matches = ([cached_path] if cached_path is not None else
                   [path for path, filename in self.names.items() if name.endswith(filename)])
        if len(matches) > 1:
            return None
        if matches:
            path = matches[0]
        else:
            # 保存提示中的冒号可能也被 OCR 丢失。按文件名内的稳定片段对齐尾部，
            # 再计算完整文件名相似度，不能把中文提示或人脸数当成文件名的一部分。
            parts = re.split(r"[:：]", text)
            caption = _normalized(parts[-1]) if len(parts) > 1 else name
            scores = []
            for candidate_path, filename in self.names.items():
                matcher = SequenceMatcher(None, caption, filename, autojunk=False)
                block = matcher.find_longest_match()
                if block.size == 0:
                    continue
                start = max(0, block.a - block.b)
                tail = caption[start:]
                # 仍要求尾部与整个文件名接近；仅有一个相同片段不能确认位置。
                tail_matcher = SequenceMatcher(None, tail, filename, autojunk=False)
                # 最佳候选至少为 0.92；低于 0.92 - 0.035 的候选也不可能
                # 成为影响歧义判断的第二名，只用上界排除这些候选。
                if (tail_matcher.real_quick_ratio() < 0.92 - 0.035
                        or tail_matcher.quick_ratio() < 0.92 - 0.035):
                    continue
                score = tail_matcher.ratio()
                scores.append((score, candidate_path))
            scores.sort(reverse=True)
            if not scores or scores[0][0] < 0.92:
                return None
            if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.035:
                return None
            path = scores[0][1]
        # Cache only the unique filename, never its position: a later verified
        # subdirectory can give the same file a different index and total.
        if len(self._filename_cache) >= 128:
            self._filename_cache.clear()
        self._filename_cache[name] = path
        groups = self._groups_by_path.get(path, [])
        group = self._last_verified_group
        if group is None or group not in groups:
            # 多个可能的目录序列时，仍可展示和识别文件名，但此处的序号
            # 只供显示，group_for 不允许使用它发送导航输入。
            group = groups[0] if len(groups) == 1 else self._all_paths
        # 没有可靠计数时仅确认文件名；序号来自此前已核对序列或完整 Path 排序。
        return LegacyPosition(path, group.index(path) + 1, len(group), counter_verified=False)

    def group_for(self, position: LegacyPosition) -> list[str] | None:
        """恢复该位置所属的完整序列，禁止把递归目录误当成单个父目录。"""
        if not 1 <= position.index <= position.total:
            return None
        if not position.counter_verified:
            possible = self._groups_by_path.get(position.image_path, [])
            if self._last_verified_group is not None and self._last_verified_group in possible:
                group = self._last_verified_group
            else:
                # 唯一候选也可能只是新版导入的部分图片；没有旧版编号和总数
                # 核对过完整顺序时，不能据此发送翻页输入。
                return None
            return (group if len(group) == position.total
                    and group[position.index - 1] == position.image_path else None)
        matches = [group for group in self._groups_by_total.get(position.total, [])
                   if group[position.index - 1] == position.image_path]
        return matches[0] if len(matches) == 1 else None

    def target_index(self, target: str, position: LegacyPosition) -> int | None:
        group = self.group_for(position)
        return group.index(target) + 1 if group is not None and target in group else None


# 使用系统自带 Windows OCR；常驻进程，图片只走管道，不写截图到用户目录。
_OCR_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
public static class SyncWindowCapture {
    [StructLayout(LayoutKind.Sequential)] struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out Rect r);
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] static extern bool IsHungAppWindow(IntPtr h);
    [DllImport("user32.dll")] static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
    public static int Left, Top, Width, Height, AreaX, AreaY;
    public static byte[] Capture(long handle, bool navigation) {
        SetThreadDpiAwarenessContext(new IntPtr(-4));
        IntPtr h = new IntPtr(handle);
        Rect r;
        if (handle == 0) throw new Exception("Legacy window not found");
        if (IsIconic(h)) throw new Exception("Legacy window minimized: " + handle);
        if (IsHungAppWindow(h)) throw new Exception("Legacy window not responding: " + handle);
        if (!GetWindowRect(h, out r)) throw new Exception("Invalid legacy window: " + handle);
        int w = r.Right-r.Left, height = r.Bottom-r.Top;
        Left = r.Left; Top = r.Top; Width = w; Height = height;
        if (w < 100 || height < 60 || (long)w*height > 40000000)
            throw new Exception("Invalid legacy window dimensions");
        using (Bitmap image = new Bitmap(w, height, PixelFormat.Format32bppArgb)) {
            using (Graphics g = Graphics.FromImage(image)) {
                IntPtr dc = g.GetHdc();
                try {
                    if (!PrintWindow(h, dc, 2)) throw new Exception("Cannot capture legacy window");
                } finally { g.ReleaseHdc(dc); }
            }
            Rectangle area = navigation
                ? new Rectangle(0, 0, Math.Min(w, Math.Max(360, w/2)), Math.Min(height, Math.Max(180, height/3)))
                : new Rectangle(8, height-42, w-16, 36);
            AreaX = area.X; AreaY = area.Y;
            using (Bitmap footer = image.Clone(area, PixelFormat.Format24bppRgb))
            using (Bitmap scaled = new Bitmap(footer.Width*2, footer.Height*2)) {
                using (Graphics g = Graphics.FromImage(scaled)) {
                    g.Clear(Color.White);
                    g.DrawImage(footer, 0, 0, scaled.Width, scaled.Height);
                }
                using (MemoryStream output = new MemoryStream()) {
                    scaled.Save(output, ImageFormat.Png);
                    return output.ToArray();
                }
            }
        }
    }
}
'@
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($Operation, $ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    $task.Result
}
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('en-US'))
if ($null -eq $engine) { $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
if ($null -eq $engine) { throw 'Windows OCR language is unavailable' }
$navEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $navEngine) { $navEngine = $engine }
[Console]::WriteLine('{"ready":true}')
while ($null -ne ($line = [Console]::ReadLine())) {
    $stream = $null; $writer = $null; $bitmap = $null
    try {
        $navigation = $line.StartsWith('nav:')
        $handleText = if ($navigation) { $line.Substring(4) } else { $line }
        $stream = [Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
        $writer = [Windows.Storage.Streams.DataWriter]::new($stream)
        $captureWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $imageBytes = [SyncWindowCapture]::Capture([long]$handleText, $navigation)
        $captureWatch.Stop()
        $decodeWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $writer.WriteBytes($imageBytes)
        $null = Await ($writer.StoreAsync()) ([uint32])
        $null = $writer.DetachStream()
        $stream.Seek(0)
        $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $decodeWatch.Stop()
        $activeEngine = if ($navigation) { $navEngine } else { $engine }
        $recognizeWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $result = Await ($activeEngine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $recognizeWatch.Stop()
        $timings = @{capture_ms=$captureWatch.Elapsed.TotalMilliseconds;
                     decode_ms=$decodeWatch.Elapsed.TotalMilliseconds;
                     recognize_ms=$recognizeWatch.Elapsed.TotalMilliseconds}
        if ($navigation) {
            $words = @(
                foreach ($ocrLine in $result.Lines) {
                    foreach ($word in $ocrLine.Words) {
                        $box = $word.BoundingRect
                        @{text=$word.Text; x=($box.X/2 + [SyncWindowCapture]::AreaX);
                          y=($box.Y/2 + [SyncWindowCapture]::AreaY);
                          width=($box.Width/2); height=($box.Height/2)}
                    }
                }
            )
            $rect = @([SyncWindowCapture]::Left, [SyncWindowCapture]::Top,
                      [SyncWindowCapture]::Width, [SyncWindowCapture]::Height)
            [Console]::WriteLine((@{words=$words; rect=$rect; timings=$timings} | ConvertTo-Json -Depth 4 -Compress))
        } else {
            [Console]::WriteLine((@{text=$result.Text; timings=$timings} | ConvertTo-Json -Depth 4 -Compress))
        }
    } catch {
        [Console]::WriteLine((@{error=$_.Exception.Message} | ConvertTo-Json -Compress))
    } finally {
        if ($null -ne $bitmap) { $bitmap.Dispose() }
        if ($null -ne $writer) { $writer.Dispose() }
        if ($null -ne $stream) { $stream.Dispose() }
    }
}
'''


class WindowsOcrReader:
    def __init__(self):
        self.process = None
        self.lines = queue.Queue()
        self.last_error = ""
        self._aborted = threading.Event()
        self._navigation_cache = None
        self.last_timings = {}

    def _start(self):
        if self._aborted.is_set():
            raise RuntimeError("位置读取已停止")
        if self.process is not None and self.process.poll() is None:
            return
        started = time.monotonic()
        self.close()
        lines = self.lines = queue.Queue()
        encoded = base64.b64encode(_OCR_SCRIPT.encode("utf-16-le")).decode("ascii")
        powershell = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                  "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        process = self.process = subprocess.Popen(
            [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        if self._aborted.is_set():
            process.kill()

        def collect():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        threading.Thread(target=collect, daemon=True).start()
        # 首次加载 WinRT 和编译捕获代码可能超过普通读数的三秒时限。
        response = self._response(timeout=15)
        if not response.get("ready"):
            raise RuntimeError("Windows OCR 启动失败")
        sync_log("ocr_service_ready", startup_ms=round((time.monotonic() - started) * 1000, 3))

    def _response(self, timeout=3):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError("Windows OCR 读取超时") from None
        if line is None:
            self.close()
            raise RuntimeError("Windows OCR 不可用，请检查系统 OCR 语言包")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def read_text(self, hwnd: int) -> str:
        started = time.monotonic()
        self.last_timings = {}
        self._start()
        ready = time.monotonic()
        # 截图和 OCR 都在子进程中；PrintWindow 卡住时也能超时/主动终止。
        self.process.stdin.write(str(int(hwnd)) + "\n")
        self.process.stdin.flush()
        result = self._response()
        self.last_timings = {**result.get("timings", {}),
                             "startup_ms": round((ready - started) * 1000, 3),
                             "request_ms": round((time.monotonic() - ready) * 1000, 3)}
        return result.get("text", "")

    def navigation(self, hwnd: int) -> dict:
        from .legacy_buttons import current_rectangle, native_navigation_snapshot

        native = native_navigation_snapshot(hwnd)
        if native is not None:
            return native
        identity = (hwnd, window_identity(hwnd), current_rectangle(hwnd))
        if self._navigation_cache is not None and self._navigation_cache[0] == identity:
            return self._navigation_cache[1]
        self._start()
        self.process.stdin.write(f"nav:{int(hwnd)}\n")
        self.process.stdin.flush()
        result = self._response(timeout=5)
        if tuple(result.get("rect", [])) != identity[2]:
            raise RuntimeError("旧版窗口正在移动或缩放，请重试同步")
        self._navigation_cache = (identity, result)
        return result

    def search_words(self, hwnd: int) -> dict:
        """Read the search field using the existing small navigation OCR crop."""
        self._start()
        self.process.stdin.write(f"nav:{int(hwnd)}\n")
        self.process.stdin.flush()
        return self._response(timeout=3)

    def close(self):
        self._navigation_cache = None
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()

    def abort(self):
        """可从界面线程调用：中断管道等待，回收仍在后台线程完成。"""
        self._aborted.set()
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    message: str


def confirm_navigation(debt, position, hwnd: int) -> bool:
    """仅确认已发出的那一步；不可靠读数会打断连续文件名确认。"""
    if debt is None:
        return False
    if (position is None or debt["hwnd"] != hwnd
            or debt["expected"] != position.image_path):
        debt["confirmed"] = 0
        return False
    debt["confirmed"] += 1
    return position.counter_verified or debt["confirmed"] >= 2


def inflight_displaced(debt, position, hwnd: int, catalog) -> bool:
    """识别旧版已远离未确认位置；只作提示，不释放在途输入。"""
    if (debt is None or position is None or not position.counter_verified
            or debt["hwnd"] != hwnd):
        return False
    group = catalog.group_for(position)
    expected_index = debt["expected_index"]
    if group is None:
        return False
    if not 1 <= expected_index <= len(group) or group[expected_index - 1] != debt["expected"]:
        return True  # 旧版目录序列已变；提示用户恢复，仍不释放原输入。
    return abs(position.index - expected_index) > 1


def navigate_verified(catalog, target, read_position, send_steps, cancelled=lambda: False,
                      observed=lambda p: None, wait=time.sleep, jump_to=None) -> SyncResult:
    """最多保留一个未确认导航动作；连续两次读到目标才报告完成。"""
    deadline = time.monotonic() + 60
    confirmed = unreadable = 0
    distance_budget_set = False
    pending_index = None
    pending_since = 0.0
    pre_send_failures = 0
    while time.monotonic() < deadline:
        if cancelled():
            return SyncResult(False, "同步已取消")
        position = read_position()
        if cancelled():
            return SyncResult(False, "同步已取消")
        if position is None:
            confirmed = 0
            unreadable += 1
            # 连按 S、D 时旧版还可能在保存和重绘；短暂读不到不能立刻丢掉目标。
            if unreadable >= 6:
                return SyncResult(False, "未确认旧版位置：请加载同一完整目录，并让旧版底部编号和文件名可见")
            wait(0.12)
            continue
        unreadable = 0
        observed(position)
        target_index = catalog.target_index(target, position)
        if target_index is None:
            if (not position.counter_verified and position.image_path == target
                    and pending_index is None):
                confirmed += 1
                if confirmed >= 2:
                    return SyncResult(True, "已核对旧版当前文件名与目标一致")
                wait(0.06)
                continue
            if not position.counter_verified and catalog.group_for(position) is None:
                return SyncResult(False, "旧版目录顺序尚未确认，未发送导航；请让旧版显示底部编号后重试")
            return SyncResult(False, "两边不是同一完整图片目录，未发送按键")
        delta = target_index - position.index
        if abs(delta) > 1 and pending_index is None and jump_to is not None:
            if cancelled():
                return SyncResult(False, "同步已取消")
            direct_result = jump_to(target, position)
            if direct_result is not None:
                return direct_result
        if not distance_budget_set:
            # 筛选项可能相距数百张，不能用小跳转的固定时限把正常长跳转截断。
            deadline = max(deadline, time.monotonic() + 15 + abs(delta) * 2.0)
            distance_budget_set = True
        if pending_index is not None:
            if position.index == pending_index:
                pending_index = None
            else:
                if time.monotonic() - pending_since >= 10:
                    return SyncResult(False, "旧版尚未确认上一步，已停止补发；请按同步提示手动对齐后重试")
                wait(0.06)
                continue
        if delta == 0:
            confirmed += 1
            if confirmed >= 2:
                if not position.counter_verified:
                    return SyncResult(True, "已核对旧版当前文件名与目标一致")
                return SyncResult(True, f"已核对旧版第 {position.index}/{position.total} 张与文件名")
            wait(0.06)
            continue
        confirmed = 0
        if cancelled():
            return SyncResult(False, "同步已取消")
        batch, gap = 1, 0.015
        vk = 0x44 if delta > 0 else 0x41
        sync_log("send_batch", target=target, index=position.index,
                 destination=target_index, count=batch, vk=vk,
                 counter_verified=position.counter_verified)
        if not send_steps(vk, batch, gap):
            # send_steps 返回 False 只表示尚未发送输入。短暂忙碌时重新读位置，
            # 不沿用旧编号；已经注入的动作必须等确认，不能在此重复发送。
            pre_send_failures += 1
            if cancelled():
                return SyncResult(False, "同步已取消")
            if pre_send_failures >= 6:
                return SyncResult(False, "焦点、拖动或按钮状态已改变，已停止发送导航")
            sync_log("navigation_precondition_retry", attempt=pre_send_failures)
            wait(0.12)
            continue
        pre_send_failures = 0
        pending_index = position.index + (batch if delta > 0 else -batch)
        pending_since = time.monotonic()
        wait(0.04)
    return SyncResult(False, "同步超时，已停止；再次点击目标可重新核对位置")


class LegacySyncWorker(QThread):
    position = Signal(object)
    completed = Signal(int, bool, str)
    warning = Signal(str)

    def __init__(self, paths, parent=None, *, initial_inflight=None):
        super().__init__(parent)
        self._catalog = PositionCatalog(paths)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._generation = 0
        self._request = None
        self._request_at = None
        self._reader = None
        self._jump_queries = {}
        self._jump_cleanup = None
        # 同一助手窗口内停用/重启同步时保留已发出但尚未确认的输入。
        self._inflight = initial_inflight

    def set_paths(self, paths):
        paths = tuple(dict.fromkeys(str(p) for p in paths))
        with self._lock:
            if paths == self._catalog.paths:
                return False
        catalog = PositionCatalog(paths)
        with self._lock:
            self._catalog = catalog
            self._jump_queries.clear()
            self._generation += 1
            self._request = None
            self._request_at = None
        self.wake()
        return True

    def request_target(self, path):
        with self._lock:
            if self._request and self._request[1] == path:
                return self._request[0]
            self._generation += 1
            self._request = (self._generation, path)
            self._request_at = time.monotonic()
            generation = self._generation
        self.wake()
        sync_log("request", generation=generation, target=path)
        return generation

    def cancel(self):
        with self._lock:
            self._generation += 1
            self._request = None
            self._request_at = None
            generation = self._generation
        self.wake()
        sync_log("cancel", generation=generation)

    def navigation_pending(self) -> bool:
        return self._inflight is not None

    def wake(self):
        self._wake.set()

    def _cancelled(self, generation):
        with self._lock:
            return self._stop_event.is_set() or generation != self._generation

    def run(self):
        reader = self._reader = WindowsOcrReader()
        last_warning = ""
        try:
            if not self._stop_event.is_set():
                try:
                    # 导入后在本线程预热 WinRT/PowerShell，避免首次选图才冷启动。
                    # _reader 已发布，stop() 可随时 abort 这段启动等待。
                    reader._start()
                except Exception as exc:
                    # 预热失败不终止同步；后续读数沿用原有的按需启动重试。
                    reader.last_error = str(exc)
                    reader.close()
                    if not self._stop_event.is_set():
                        sync_log("ocr_warmup_error", error=repr(exc))
            while not self._stop_event.is_set():
                self._wake.clear()
                with self._lock:
                    catalog, request = self._catalog, self._request
                    request_at = self._request_at
                hwnd = find_window_by_title(["属性标注", "face_attribute"])
                if not is_foreground(hwnd):
                    confirm_navigation(self._inflight, None, hwnd)
                last_position = None

                def read():
                    nonlocal last_position
                    if self._stop_event.is_set() or not is_foreground(hwnd):
                        confirm_navigation(self._inflight, None, hwnd)
                        return None
                    read_started = time.monotonic()
                    try:
                        text = reader.read_text(hwnd)
                    except Exception as exc:
                        # 捕获/重绘的瞬时错误和 OCR 冷启动超时应进入有上限的回读，
                        # 不应直接丢弃用户刚选择的目标。
                        reader.last_error = str(exc)
                        sync_log("read_error", hwnd=hwnd, error=str(exc),
                                 elapsed_ms=round((time.monotonic() - read_started) * 1000, 3))
                        confirm_navigation(self._inflight, None, hwnd)
                        return None
                    # Tk 的 PrintWindow 在遮挡时可能带入其他窗口像素，只接受前台稳定读数。
                    resolve_started = time.monotonic()
                    position = catalog.resolve(text) if is_foreground(hwnd) else None
                    if request:
                        sync_log("sync_read_timing", generation=request[0],
                                 read_ms=round((resolve_started - read_started) * 1000, 3),
                                 resolve_ms=round((time.monotonic() - resolve_started) * 1000, 3),
                                 recognized=position is not None,
                                 ocr=getattr(reader, "last_timings", {}))
                    debt = self._inflight
                    if confirm_navigation(debt, position, hwnd):
                        self._inflight = None
                        sync_log("inflight_confirmed", image=position.image_path)
                    elif (debt is not None and position is not None
                          and debt["hwnd"] == hwnd and debt["expected"] == position.image_path):
                        # 保存状态缺少编号时需连续两次文件名确认；中途失败要重新计数。
                        position = None
                    if position is not None:
                        last_position = position
                    reader.last_error = "" if position is not None else "底部编号或文件名未确认"
                    if request:
                        sync_log("read_position", generation=request[0], text=text,
                                 position=repr(position))
                    return position

                def send_one(vk, count, gap):
                    if count != 1 or last_position is None or self._inflight is not None:
                        return False
                    group = catalog.group_for(last_position)
                    if group is None:
                        return False
                    expected = last_position.index + (1 if vk == 0x44 else -1)
                    if not 1 <= expected <= len(group):
                        return False
                    identity = window_identity(hwnd)
                    if identity is None:
                        return False

                    def mark_sent():
                        self._inflight = {"hwnd": hwnd, "identity": identity,
                                          "expected": group[expected - 1],
                                          "expected_index": expected, "confirmed": 0,
                                          "sent_at": time.monotonic()}

                    navigation_started = time.monotonic()
                    snapshot = reader.navigation(hwnd)
                    input_started = time.monotonic()
                    try:
                        sent = send_navigation_button(hwnd, vk, snapshot,
                            cancelled=lambda: self._cancelled(generation), on_sent=mark_sent)
                        sync_log("sync_send_timing", generation=generation,
                                 controls_ms=round((input_started - navigation_started) * 1000, 3),
                                 button_send_ms=round((time.monotonic() - input_started) * 1000, 3), sent=sent)
                        return sent
                    except Exception:
                        # 不让一次识别失败的缓存永久挡住后续重试。
                        reader._navigation_cache = None
                        raise

                def jump_to(target, position):
                    if not hasattr(reader, 'search_words'):
                        return None
                    from .legacy_jump import NativeJumpSender, perform_direct_jump, unique_search_query
                    group = catalog.group_for(position)
                    if group is None:
                        return SyncResult(False, '旧版完整目录未确认，未执行直接定位')
                    key = (id(group), target)
                    if key not in self._jump_queries:
                        if len(self._jump_queries) >= 256:
                            self._jump_queries.clear()
                        self._jump_queries[key] = unique_search_query(group, target)
                    query = self._jump_queries[key]
                    if not query:
                        return SyncResult(False, '无法唯一定位该文件，未打开其他图片')
                    identity = window_identity(hwnd)
                    if identity is None:
                        return SyncResult(False, '旧版窗口已关闭')
                    target_index = group.index(target) + 1
                    sender = NativeJumpSender(hwnd, reader,
                        lambda: self._cancelled(generation), self._stop_event.wait)
                    self._jump_cleanup = sender

                    def mark_jump_sent():
                        self._inflight = {'hwnd': hwnd, 'identity': identity, 'expected': target,
                            'expected_index': target_index, 'confirmed': 0,
                            'sent_at': time.monotonic()}

                    started = time.monotonic()
                    result = perform_direct_jump(query, target, sender, read,
                        cancelled=lambda: self._cancelled(generation), on_sent=mark_jump_sent,
                        wait=self._stop_event.wait)
                    if not sender.dirty:
                        self._jump_cleanup = None
                    sync_log('direct_jump', generation=generation, ok=result.ok,
                        distance=abs(target_index-position.index), query_length=len(query),
                        elapsed_ms=round((time.monotonic()-started)*1000, 3))
                    return SyncResult(result.ok, result.message)

                try:
                    debt = self._inflight
                    if debt is not None and window_identity(debt["hwnd"]) != debt["identity"]:
                        # 原窗口已销毁或句柄被其他进程/线程复用，旧消息不会作用于新窗口。
                        self._inflight = None
                        sync_log("inflight_window_gone", hwnd=debt["hwnd"])
                    if request:
                        generation, target = request
                        request_started = time.monotonic()
                        displaced = False
                        reader.last_error = ""
                        if self._cancelled(generation):
                            continue
                        if not hwnd:
                            result = SyncResult(False, "未找到唯一的「年龄 / 性别属性标注」窗口")
                        elif not _force_foreground(hwnd):
                            result = SyncResult(False, "无法激活旧版，未发送按键；请再次点击目标图片")
                        else:
                            if self._jump_cleanup is not None and self._jump_cleanup.dirty:
                                if not self._jump_cleanup.clear_query():
                                    raise RuntimeError('上次定位搜索尚未清空，已停止新的定位')
                                self._jump_cleanup = None
                            sync_log("navigation_start", generation=generation,
                                     target=target, hwnd=hwnd)
                            self._stop_event.wait(0.08)
                            settle_deadline = time.monotonic() + 10
                            displaced_key = None
                            displaced_reads = 0
                            displaced_since = 0.0
                            while (self._inflight is not None and time.monotonic() < settle_deadline
                                   and not self._cancelled(generation) and is_foreground(hwnd)):
                                position = read()
                                debt = self._inflight
                                group = (catalog.group_for(position) if position is not None
                                         and position.counter_verified else None)
                                same_sequence = (debt is not None and group is not None
                                    and debt["hwnd"] == hwnd
                                    and window_identity(hwnd) == debt["identity"]
                                    and 1 <= debt["expected_index"] <= len(group)
                                    and group[debt["expected_index"] - 1] == debt["expected"])
                                if same_sequence and position.image_path != debt["expected"]:
                                    key = (position.image_path, position.index, position.total)
                                    if displaced_key == key:
                                        displaced_reads += 1
                                    else:
                                        displaced_key, displaced_reads = key, 1
                                        displaced_since = time.monotonic()
                                    displaced = inflight_displaced(debt, position, hwnd, catalog)
                                    now = time.monotonic()
                                    if (displaced_reads >= 3 and now - displaced_since >= .6
                                            and now - debt.get("sent_at", request_started) >= 1.0):
                                        # A fresh user target can recover after manual navigation
                                        # or a consumed click with a missed acknowledgement.
                                        # Re-read counter + filename in the same sequence over
                                        # a settling interval; never replay the orphaned step.
                                        self._inflight = None
                                        sync_log("inflight_rebased", expected_index=debt["expected_index"],
                                                 current_index=position.index, reads=displaced_reads)
                                        break
                                else:
                                    displaced_key, displaced_reads = None, 0
                                self._stop_event.wait(0.06)
                            if self._inflight is not None:
                                result = SyncResult(False, "上一次导航仍未确认，已停止发送")
                            else:
                                result = navigate_verified(
                                    catalog, target, read, send_one,
                                    cancelled=lambda: self._cancelled(generation),
                                    observed=lambda p: (self.position.emit(p)
                                        if not self._cancelled(generation) else None),
                                    wait=self._stop_event.wait, jump_to=jump_to)
                        if not result.ok and self._inflight is not None:
                            if displaced:
                                result = SyncResult(False,
                                    "旧版的位置或目录已改变，之前的一步仍未确认。"
                                    "请先保存并关闭旧版，重新打开后再点“重试同步”。")
                            else:
                                result = SyncResult(False,
                                    "上一步尚未确认。请先保存并关闭旧版，"
                                    "重新打开后再点“重试同步”。")
                        elif not result.ok and reader.last_error:
                            result = SyncResult(False, f"{result.message}（{reader.last_error}）")
                        request_finished = time.monotonic()
                        queued_at = request_at if request_at is not None else request_started
                        sync_log("sync_request_timing", generation=generation, ok=result.ok,
                                 cancelled=self._cancelled(generation),
                                 queue_ms=round((request_started - queued_at) * 1000, 3),
                                 processing_ms=round((request_finished - request_started) * 1000, 3),
                                 total_ms=round((request_finished - queued_at) * 1000, 3))
                        if not self._cancelled(generation):
                            with self._lock:
                                if self._request == request:
                                    self._request = None
                                    self._request_at = None
                            sync_log("completed", generation=generation, ok=result.ok,
                                     message=result.message)
                            self.completed.emit(generation, result.ok, result.message)
                    elif is_foreground(hwnd):
                        position = read()
                        if position is not None:
                            self.position.emit(position)
                            last_warning = ""
                        elif last_warning != "unconfirmed":
                            self.warning.emit("旧版位置未确认：请加载同一完整目录，并保持底部编号和文件名可见")
                            last_warning = "unconfirmed"
                except Exception as exc:
                    message = f"位置读取失败：{exc}"
                    sync_log("worker_error", request=request, error=repr(exc))
                    if request and not self._cancelled(request[0]):
                        with self._lock:
                            if self._request == request:
                                self._request = None
                        self.completed.emit(request[0], False, message)
                    elif message != last_warning and not self._stop_event.is_set():
                        self.warning.emit(message)
                    last_warning = message
                self._wake.wait(0.35 if hwnd else 0.8)
        finally:
            reader.close()

    def stop(self):
        self._stop_event.set()
        self.cancel()
        reader = self._reader
        if reader is not None and hasattr(reader, "abort"):
            reader.abort()
        return self.wait(1500)
