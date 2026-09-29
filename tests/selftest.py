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

# Unreal export (no Unreal needed): plan, glb geometry and the Blender-baked colour LUT.
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from gtb import config  # noqa: E402
from gtb.scene_common import load_look  # noqa: E402
from ue import export, pipeline  # noqa: E402

cap = OUT / "capture"
plan = export.export(cap)
glb = next(iter(plan["glb_meshes"]))
data = (cap / "unreal" / "meshes" / glb).read_bytes()
first = next(a for a in plan["actors"])
M = np.array(cam["matrix_world"])
fwd = export.dir_to_ue(-M[:3, 2] / np.linalg.norm(M[:3, 2]))
lut = pipeline.bake_lut(config.load(), cap, load_look(cap / "look.json"))
import cv2  # noqa: E402
L = cv2.imread(str(lut), cv2.IMREAD_UNCHANGED)
n = L.shape[0]
grey = [L[i, i * n + i, 1] for i in range(n)]   # r = g = b along the diagonal
imp = subprocess.run([sys.executable, str(ROOT / "gtb.py"), "import", str(Path(m["textures"]["t0000"]["source"]).parent),
                      "--name", "_selftest/imported"], capture_output=True, text=True)
imp_meta = json.loads((OUT / "imported" / "capture.json").read_text()) if imp.returncode == 0 else {}
# Texture roles: a nearly flat normal packed in A+G (roughness/mask in R/B) is not
# an albedo; and the shader slot consensus overrides a lone contrary classification.
from gtb import textures  # noqa: E402
rng = np.random.default_rng(3)
flat_ag = np.stack([rng.random((256, 256)), 0.5 + 0.01 * rng.standard_normal((256, 256)),
                    0.3 * rng.random((256, 256)), 0.5 + 0.005 * rng.standard_normal((256, 256))], -1).astype(np.float32)
slot_tex = {f"t{i}": {"role": "albedo", "details": {}} for i in range(4)}
slot_tex.update({f"n{i}": {"role": "normal", "details": {"channels": "AG"}} for i in range(4)})
slot_tex["n3"] = {"role": "albedo", "details": {}}                      # a normal taken for an albedo
textures.slot_consensus([("ps", {"0": f"t{i}", "1": f"n{i}"}) for i in range(4)], slot_tex)
checks.update({
    "textures: flat A+G normal is a normal": textures.classify(flat_ag, None) == ("normal", {"channels": "AG"}),
    "textures: slot consensus fixes a lone albedo in a normal slot": slot_tex["n3"]["role"] == "normal"
    and slot_tex["n3"]["details"].get("channels") == "AG" and slot_tex["t3"]["role"] == "albedo",
})
checks.update({
    "import: capture.json from the rip": imp_meta.get("game_exe") == "Frostpunk.exe"
    and imp_meta.get("resolution") == list(m["resolution"]),
    "unreal plan: 6 actors": len(plan["actors"]) == 6,
    "unreal plan: fov 45": abs(plan["camera"]["fov_y_deg"] - 45.0) < 0.05,
    "unreal plan: camera forward (x, -y, z)": np.allclose(plan["camera"]["forward"], fwd, atol=1e-4),
    "unreal glb valid": data[:4] == b"glTF" and len(plan["glb_meshes"][glb]) == 6,
    "unreal LUT 64^3, grey ramp rises": L.shape[:2] == (64, 64 * 64) and all(np.diff(grey) >= 0) and grey[-1] > grey[0],
})
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")
print(f"comparison sheet: {OUT / 'capture' / 'comparison.png'}")
sys.exit(0 if all(checks.values()) else 1)
