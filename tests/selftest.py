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
# A particle layer process.py dropped as snowflakes (faint flipbook) is mist when its cards
# are mist-sized (Frostpunk mesh_5273: 21 m cards, mean alpha 0.009); snowflake sheets
# (mean alpha 0.0001) and small steam puffs stay hidden.
def _cards(width, n=5):
    q = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]) * width
    pos = np.concatenate([q + [30.0 * k, 0, 0] for k in range(n)]).astype(np.float32)
    tris = np.concatenate([[[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]] for k in range(n)])
    return {"positions": pos, "indices": tris.ravel(), "uv0": np.zeros((len(pos), 2), np.float32)}
_prof = {"sprite_layouts": ["POSITION0,COLOR0,TEXCOORD0"]}
_fx = {"category": "effect", "layout_pre": ["POSITION0:0x3", "COLOR0:0x4", "TEXCOORD0:0x4"], "textures": {"0": "fb"}}
_fb = lambda a: {"fb": {"role": "gray", "details": {"alpha_mean": a, "alpha_std": 0.06}}}
mist_ok = (export.faint_mist(_fx, _cards(21.0), _fb(0.009), _prof, np.eye(3)) or [{}])[0].get("kind") == "mist"
mist_ok &= export.faint_mist(_fx, _cards(21.0), _fb(0.0001), _prof, np.eye(3)) is None      # snowflake sheet
mist_ok &= export.faint_mist(_fx, _cards(7.0), _fb(0.009), _prof, np.eye(3)) is None        # building steam
mist_ok &= export.faint_mist(dict(_fx, category="surface"), _cards(21.0), _fb(0.009), _prof, np.eye(3)) is None
# Known textures: a rip that holds a texture from a lower mip level down (streaming) still
# matches it by the mip levels it shares; a hash that points to two roles decides nothing.
import struct  # noqa: E402
def _dds(path, levels):
    h, w = levels[0].shape[:2]
    head = struct.pack("<4s7I44x", b"DDS ", 124, 0x2100F, h, w, w * 4, 0, len(levels))
    pf = struct.pack("<2I4s5I", 32, 0x41, bytes(4), 32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000)
    Path(path).write_bytes(head + pf + struct.pack("<5I", 0x401008, 0, 0, 0, 0) + b"".join(l.tobytes() for l in levels))
_r = np.random.default_rng(7)
_lv = [_r.integers(0, 255, (s, s, 4), dtype=np.uint8) for s in (64, 32, 16, 8, 4, 2, 1)]
_dds(OUT / "full.dds", _lv)
_dds(OUT / "streamed.dds", _lv[1:])                                  # the same texture, top level not loaded
_dds(OUT / "flat.dds", [np.full((s, s, 4), 128, np.uint8) for s in (32, 16)])
h_full, h_low = textures.mip_hashes(OUT / "full.dds"), textures.mip_hashes(OUT / "streamed.dds")
_known = {h: {"ref": "ref/t0001", "role": "normal", "channels": "AG", "hashes": h_full} for h in h_full}
_ents = {"a": {"role": "albedo", "details": {}, "hashes": h_low}, "b": {"role": "albedo", "details": {}, "hashes": ["x"]}}
_hit = textures.apply_known(_ents, _known)
_amb = {"a": {"role": "albedo", "details": {}, "hashes": h_low}}
textures.apply_known(_amb, {**_known, h_low[1]: {"ref": "ref/t0002", "role": "gray", "hashes": [h_low[1]]}})
known_ok = (len(h_full) == 3 and h_low == h_full[1:] and textures.mip_hashes(OUT / "flat.dds") == []
            and _ents["a"]["role"] == "normal" and _ents["a"]["details"]["channels"] == "AG" and "a" in _hit
            and _ents["a"]["known"]["ref"] == "ref/t0001" and _ents["b"]["role"] == "albedo" and "known" not in _ents["b"]
            and _amb["a"]["role"] == "albedo")
# The database helpers: add by DDS file or by hashes, get by either; the same texture added
# from another rip updates its entry (and keeps where it was first checked) instead of a second one.
_kdb = OUT / "known.json"
_kdb.unlink(missing_ok=True)
textures.known_add(_kdb, OUT / "full.dds", "normal", channels="AG", what="a note", ref="first/t0001")
_again = textures.known_add(_kdb, h_low, "normal", channels="AG", ref="second/t0009")
helpers_ok = (textures.known_get(_kdb, OUT / "streamed.dds")["ref"] == "first/t0001" and _again["what"] == "a note"
              and textures.known_get(_kdb, ["nope"]) is None and len(json.loads(_kdb.read_text())["textures"]) == 1)
from gtb.audit import audit  # noqa: E402
audit_dir, audit_rows = audit(cap, plan=plan)
checks.update({
    "textures: a known texture is matched by its mip hashes, also when streamed at a lower level": known_ok,
    "textures: known_add / known_get keep one entry per texture": helpers_ok,
    "audit: writes the material sheets and the report": len(audit_rows) > 0 and (audit_dir / "report.md").exists()
    and (audit_dir / "materials_01.png").exists(),
})
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
    "unreal: a faint particle layer with mist-sized cards is mist, snowflakes and steam aren't": mist_ok,
})
# People: a synthetic person on the Frostpunk rig (tubes along the bones, blended at the
# joints), posed by known bone rotations and drawn through linear blend skinning at
# 1.3x, mirrored to D3D like a rip's bind pose. The solver recovers its bones and scale,
# the walk keeps a planted foot in place, and a loop closes on itself.
from gtb import characters as ch  # noqa: E402
from ue import walkers as walkers_mod  # noqa: E402
_rig = ch.rig_from_profile(json.loads((ROOT / "profiles" / "frostpunk.json").read_text(encoding="utf-8")))
_jt = {1: (0, 1.0, 0), 2: (0, 1.1, 0), 3: (0, 1.3, 0), 4: (0, 1.5, 0), 5: (0, 1.6, 0), 6: (0.05, 1.45, 0),
       7: (0.2, 1.45, 0), 8: (0.32, 1.2, 0), 9: (0.42, 1.0, 0), 11: (0.46, 0.92, 0), 21: (0.1, 0.95, 0),
       22: (0.1, 0.5, 0), 23: (0.1, 0.09, 0)}
for _a, _b in _rig.spec["mirror"]:
    _have, _miss = (_a, _b) if _a in _jt else (_b, _a)
    _jt[_miss] = (-_jt[_have][0],) + tuple(_jt[_have][1:])
_jt = {b: np.array(v, float) for b, v in _jt.items()}
_tip = {5: np.array([0, 1.75, 0]), 11: _jt[11] + [0.03, -0.08, 0], 17: _jt[17] + [-0.03, -0.08, 0],
        20: np.array([-0.1, 0.0, -0.18]), 23: np.array([0.1, 0.0, -0.18])}
_pts, _idx, _w = [], [], []
for _b in _rig.order:
    _kids = [k for k, q in _rig.parent.items() if q == _b]
    _end = _tip.get(_b, np.mean([_jt[k] for k in _kids], 0) if _kids else _jt[_b] + [0, -0.1, 0])
    _ax = _end - _jt[_b]
    if np.linalg.norm(_ax) < 0.05:                  # the pelvis: its children average out to itself
        _ax = np.array([0.0, 0.1, 0.0])
    _u = np.cross(_ax, [0.3, 0.2, 0.9]); _u /= np.linalg.norm(_u); _v = np.cross(_ax / np.linalg.norm(_ax), _u)
    for _s in np.linspace(0, 1, 6):
        for _k in range(6):
            _pts.append(_jt[_b] + _ax * _s + 0.03 * (np.cos(_k) * _u + np.sin(_k) * _v))
            _p = _rig.parent.get(_b, _b)
            _blend = 0.5 if _s == 0 and _p != _b else 0.0
            _idx.append([_b, _p, 0, 0]); _w.append([1 - _blend, _blend, 0, 0])
_pts, _idx, _w = np.array(_pts), np.array(_idx), np.array(_w)
_rng = np.random.default_rng(5)
_pose = {b: _rng.normal(0, 0.25, 3) for b in _rig.order}
_pose[1] = np.array([0.1, 0.4, 0.0])
_fake = ch.Person.__new__(ch.Person)
_fake.rig, _fake.nb, _fake.joints = _rig, 24, {b: _jt[b] * 1.3 for b in _rig.order}
_R, _t = _fake.fk(_pose, ch.frame(np.array([1.0, 0, 0]), _rig), np.array([5.0, 2.0, 1.3]))
_P = ch.lbs(_pts * 1.3, _idx, _w, _R, _t)
_person = ch.Person("synthetic", {"skin_bind": (_pts * [1, 1, -1]).astype(np.float32), "skin_index": _idx,
                                  "skin_weight": _w.astype(np.float32), "positions": _P}, _rig)
_Rb, _tb, _stride, _hip, _ = walkers_mod.bake_cycle(_person, ch.default_walk(_rig), frames=48)
_slip = []
for _k in range(48):
    _a, _b2 = _person.skin(_Rb[_k], _tb[_k]), _person.skin(_Rb[(_k + 1) % 48], _tb[(_k + 1) % 48])
    _a[:, 0] += _stride * _k / 48
    _b2[:, 0] += _stride * (_k + 1) / 48
    _on = (_a[:, 2] < 0.01) & (_b2[:, 2] < 0.01)
    if _on.any():
        _slip.append(np.linalg.norm((_b2[_on] - _a[_on])[:, :2], axis=1).min())
_lp, _seg = walkers_mod.loop_points(np.zeros(3), np.array([1.0, 0, 0]), 5.0, 3.0, np.array([0, 1.0, 0]))
_xy, _yaw, _L = walkers_mod.resample(_lp, _seg, 64)
checks.update({
    "people: bones, joints and scale solved from a skinned mesh": _person.res.max() < 1e-3
    and abs(_person.scale - 1.3) < 1e-3 and _person.upright()
    and max(np.linalg.norm(_person.joints[b] - _jt[b] * 1.3) for b in (19, 22, 8, 14)) < 0.01,
    "people: the walk keeps a planted foot in place": _stride > 0.5 and np.median(_slip) < 0.002,
    "people: a walking loop closes (2 lanes, 2 half turns)": abs(_L - (16 + np.pi)) < 0.05
    and abs(abs(_yaw[-1] - _yaw[0]) - 2 * np.pi) < 1e-6,
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
