"""End-to-end check without a game: synthetic rip -> process -> build -> calibrate.

  python tests/selftest.py

Asserts the camera solve recovers the synthetic camera (fov 45, 16:9) and that
the junk draws (UI quad, shadow pass, depth prepass) are filtered out.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "captures" / "_selftest"

shutil.rmtree(OUT, ignore_errors=True)
subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_rip.py"), str(OUT)], check=True)
subprocess.run([sys.executable, str(ROOT / "gtb.py"), "all", str(OUT / "capture")], check=True)

m = json.loads((OUT / "capture" / "manifest.json").read_text())
cam, stats = m["camera"], m["stats"]
checks = {
    "fov solved": cam["fov_source"] == "solved",
    "fov_y == 45": abs(cam["fov_y_deg"] - 45.0) < 0.05,
    "aspect == 16:9": abs(cam["aspect_solved"] - 16 / 9) < 0.005,
    # Shadow pass, UI quad and depth prepass: 3 junk draws, whichever filter catches them first
    # (they are untextured, so the "untextured" rule usually wins).
    "3 junk draws removed": sum(stats["skipped"].values()) == 3,
    "6 meshes kept": stats["kept"] == 6,
    "scene.blend written": (OUT / "capture" / "scene.blend").exists(),
}
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")
print(f"comparison sheet: {OUT / 'capture' / 'comparison.png'}")
sys.exit(0 if all(checks.values()) else 1)
