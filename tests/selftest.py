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
from gtb.scene_common import packed_channels, assign_slots, below_scene, decal_volume, overlay_layer  # noqa: E402
flat_r = {"details": {"mean_rgb": [0.0, 0.51, 0.0]}}                  # Frostpunk banner normal map t0006
# A unit box with split normals: 24 vertices, 8 distinct corners, 12 triangles.
_c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
box_p = np.concatenate([_c[[0, 1, 3, 2]], _c[[4, 5, 7, 6]], _c[[0, 1, 5, 4]], _c[[2, 3, 7, 6]], _c[[0, 2, 6, 4]], _c[[1, 3, 7, 5]]])
box_t = np.array([[f * 4, f * 4 + 1, f * 4 + 2] for f in range(6)] + [[f * 4, f * 4 + 2, f * 4 + 3] for f in range(6)])
# A colour atlas bound with three different albedos (Frostpunk t0033) is a detail layer;
# on a draw whose other colour is a grey bark map, the bark becomes the albedo.
layer = {"a0": {"role": "albedo", "size": [2048, 2048], "file": "a0.png"},
         "a1": {"role": "albedo", "size": [2048, 2048], "file": "a1.png"},
         "a2": {"role": "albedo", "size": [2048, 2048], "file": "a2.png"},
         "atlas": {"role": "albedo", "size": [4096, 4096], "file": "atlas.png"},
         "bark": {"role": "gray", "size": [1024, 1024], "file": "bark.png",
                  "details": {"mean_rgb": [0.65, 0.65, 0.65], "border_mean": 0.64}}}
textures.detail_layers([("ps", {"0": f"a{i}", "4": "atlas"}) for i in range(3)] + [("pt", {"0": "bark", "2": "atlas"})], layer)
bark_slots = assign_slots({"0": "bark", "2": "atlas"}, layer, {})
rock_slots = assign_slots({"0": "a0", "4": "atlas"}, layer, {})
checks.update({
    "materials: an all-zero roughness channel is not a mirror": packed_channels({"R": "Roughness"}, flat_r) == {}
    and packed_channels({"R": "Roughness"}, {"details": {"mean_rgb": [0.7, 0.5, 0.1]}}) == {"R": "Roughness"},
    "textures: flat A+G normal is a normal": textures.classify(flat_ag, None) == ("normal", {"channels": "AG"}),
    "textures: slot consensus fixes a lone albedo in a normal slot": slot_tex["n3"]["role"] == "normal"
    and slot_tex["n3"]["details"].get("channels") == "AG" and slot_tex["t3"]["role"] == "albedo",
    "scene: a surface whose only texture is a mask is an overlay (hidden)":
    overlay_layer({"mask": {}}) and not overlay_layer({"mask": {}, "normal": {}}) and not overlay_layer({})
    and not overlay_layer({"albedo": {}}),
    "scene: a box with no colour texture is a decal volume, a textured crate isn't":
    decal_volume(box_p, box_t, {"gray", "normal", "shared"}) and not decal_volume(box_p, box_t, {"albedo", "normal"})
    and not decal_volume(box_p[:6], box_t[:4], {"gray"}),
    "scene: geometry far below the scene is another pass's (hidden), a valley isn't":
    below_scene([-150, -140, -131, -122, -118, -95, -900, -1500]) == [6, 7]
    and below_scene([-150, -140, -131, -122, -118, -95, -230]) == [],
    "textures: an atlas bound with several albedos is a detail layer, not the base colour":
    layer["atlas"]["role"] == "detail" and all(layer[f"a{i}"]["role"] == "albedo" for i in range(3))
    and rock_slots["albedo"]["file"] == "a0.png" and bark_slots["albedo"]["file"] == "bark.png",
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
