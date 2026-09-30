"""Loads config.json from the project root, filling in defaults."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    # Folder Ninja Ripper writes its frame rips into (one subfolder per rip).
    # "auto" reads it from Ninja Ripper's own settings.
    "ripper_output_dir": "auto",
    # Key that triggers a frame rip in Ninja Ripper (its default is PrintScreen).
    # We only *observe* it (never swallow it) so the screenshot is the same frame.
    # Names follow pynput: print_screen, f10, insert, ... or a single character.
    "rip_hotkey": "print_screen",
    "blender_exe": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    # Relative paths are relative to the project folder, so the folder can be renamed or moved.
    "captures_dir": "captures",
    # Seconds the rip folder must stay unchanged before we treat it as finished.
    "rip_settle_seconds": 4.0,
    # Give up if Ninja Ripper hasn't started a rip folder this long after the key.
    "rip_start_timeout_seconds": 45.0,
    # Give up waiting for the rip to finish writing after this many seconds.
    "rip_timeout_seconds": 180.0,
    # Automatically process + build the .blend after each capture.
    "auto_build": True,
    # Open the finished scene in the Blender GUI after building.
    "auto_open": False,
}


def load():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    else:
        CONFIG_PATH.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    cfg["captures_dir"] = str(ROOT / cfg["captures_dir"])     # an absolute path stays as it is
    Path(cfg["captures_dir"]).mkdir(parents=True, exist_ok=True)
    if cfg["ripper_output_dir"] == "auto":
        from gtb.nr import default_output_dir
        found = default_output_dir()
        cfg["ripper_output_dir"] = found or str(Path.home() / "Documents" / "NinjaRipper")
        cfg["_ripper_output_dir_source"] = "Ninja Ripper settings" if found else "fallback (set it in config.json)"
    return cfg
