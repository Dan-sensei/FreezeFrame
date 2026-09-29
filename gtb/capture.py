"""Capture daemon.

Watches for the Ninja Ripper rip hotkey. On press it immediately screenshots the
game window (the reference image the Blender scene is matched against), then
waits for Ninja Ripper to finish writing the new rip folder and files both into
captures/<game>_<timestamp>/.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab
from pynput import keyboard

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Physical pixels, so window rects match what ImageGrab sees on scaled displays.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    user32.SetProcessDPIAware()


def foreground_window_info():
    hwnd = user32.GetForegroundWindow()
    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    bbox = (origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom)

    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title, 512)

    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    exe = ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
    if handle:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            exe = buf.value
        kernel32.CloseHandle(handle)
    return {"hwnd": hwnd, "bbox": bbox, "title": title.value, "exe": exe}


def dir_snapshot(path: Path):
    """(file count, total bytes, newest mtime) for change detection."""
    count = total = 0
    newest = 0.0
    for f in path.rglob("*"):
        if f.is_file():
            st = f.stat()
            count += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return count, total, newest


def recent_nr_files(root: Path, since: float, newest_dirs=6):
    """.nr files modified after `since` inside the newest per-process folders.

    Ninja Ripper makes one folder per hooked process (<date>_<exe>_<pid>) when
    the process starts and writes each rip somewhere inside it, so we watch for
    new mesh files rather than new folders."""
    if not root.exists():
        return []
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_ctime)[-newest_dirs:]
    out = []
    for d in dirs:
        for f in d.rglob("*.nr"):
            try:
                st = f.stat()  # on Windows st_ctime is the creation time
                if max(st.st_mtime, st.st_ctime) >= since:
                    out.append(f)
            except OSError:
                pass
    return out


def wait_for_new_rip(root: Path, since: float, settle: float, timeout: float, start_timeout: float):
    """Returns (rip_dir, [new .nr files]) once writing stops, or (None, [])."""
    start = time.time()
    files = []
    while time.time() - start < start_timeout:
        files = recent_nr_files(root, since)
        if files:
            break
        time.sleep(1.0)
    if not files:
        return None, []
    rip_dir = Path(os.path.commonpath([str(f.parent) for f in files]))
    last, stable_since = None, time.time()
    while time.time() - start < timeout:
        snap = dir_snapshot(rip_dir)
        if snap != last:
            last, stable_since = snap, time.time()
        elif time.time() - stable_since >= settle:
            break
        time.sleep(1.0)
    return rip_dir, sorted(recent_nr_files(root, since))


class CaptureDaemon:
    def __init__(self, cfg, on_capture=None):
        self.cfg = cfg
        self.on_capture = on_capture
        self.ripper_root = Path(cfg["ripper_output_dir"])
        self.captures = Path(cfg["captures_dir"])
        self.busy = threading.Lock()
        key_name = cfg["rip_hotkey"].lower()
        self.hotkey = getattr(keyboard.Key, key_name, None) or keyboard.KeyCode.from_char(key_name)

    def _on_press(self, key):
        if key == self.hotkey and not self.busy.locked():
            # Grab the frame right now, before the game moves on; the rest can wait.
            pressed = time.time() - 1.0
            info = foreground_window_info()
            shot = ImageGrab.grab(bbox=info["bbox"], all_screens=True)
            threading.Thread(target=self._finish, args=(info, shot, pressed), daemon=True).start()

    def _finish(self, info, shot, pressed):
        with self.busy:
            game = Path(info["exe"]).stem or "game"
            name = f"{game}_{datetime.now():%Y%m%d_%H%M%S}"
            out = self.captures / name
            out.mkdir(parents=True)
            shot.save(out / "screenshot.png")
            print(f"[capture] screenshot {shot.size[0]}x{shot.size[1]} from '{info['title']}' -> {out}")
            print(f"[capture] waiting for Ninja Ripper to write into {self.ripper_root} ...")

            rip, nr_files = wait_for_new_rip(self.ripper_root, pressed, self.cfg["rip_settle_seconds"],
                                             self.cfg["rip_timeout_seconds"], self.cfg["rip_start_timeout_seconds"])
            meta = {
                "game_exe": info["exe"],
                "window_title": info["title"],
                "resolution": list(shot.size),
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "rip_dir": str(rip) if rip else None,
                # Only these files belong to this capture; the folder may hold earlier rips too.
                "rip_files": [str(f.relative_to(rip)) for f in nr_files] if rip else [],
            }
            (out / "capture.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if rip is None:
                print("[capture] no rip folder appeared - Ninja Ripper doesn't seem to be hooked into the game.")
                print("[capture]   Its output folder is " + str(self.ripper_root) + ". For Steam games, fully exit Steam")
                print("[capture]   (tray icon -> Exit), start steam.exe from Ninja Ripper, then launch the game.")
                return
            n, size, _ = dir_snapshot(rip)
            print(f"[capture] rip complete: {rip} ({len(nr_files)} meshes, {n} files, {size / 1e6:.1f} MB)")
            if self.on_capture:
                self.on_capture(out)

    def run(self):
        src = self.cfg.get("_ripper_output_dir_source", "config.json")
        print(f"[capture] watching for '{self.cfg['rip_hotkey']}'. Ctrl+C to stop.")
        print(f"[capture] Ninja Ripper output: {self.ripper_root} ({src})")
        if not self.ripper_root.exists():
            print("[capture] WARNING: that folder doesn't exist yet - set ripper_output_dir in config.json")
        with keyboard.Listener(on_press=self._on_press) as listener:
            listener.join()
