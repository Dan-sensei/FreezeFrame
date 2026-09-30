"""gtb.py unreal commands: capture -> Unreal level -> Movie Render Queue -> comparison sheets.

  unreal <capture>           export plan + glb, bake the colour LUT, build the level, render, compare
  unreal-look <capture>      re-apply look.json only (lights, fog, LUT, materials), render, compare
  unreal-render <capture>    render stills (game camera + close-ups); --anim for the Level Sequence
  unreal-calibrate <capture> match exposure to the screenshot (look.unreal.exposure_offset)
  unreal-open <capture>      open the project at the capture's level

Unreal is found automatically (newest C:/Program Files/Epic Games/UE_5.x) or set
with "unreal_editor" in config.json; the project defaults to unreal_project/.
"""
import glob
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "ue" / "project_template"
EDITOR_SCRIPT = ROOT / "ue" / "editor" / "gtb_ue.py"

from gtb.scene_common import load_look  # noqa: E402
from ue import export as ue_export  # noqa: E402
from ue import look as ue_look_mod  # noqa: E402


# --------------------------------------------------------------------------- setup

def find_unreal(cfg, gui=False):
    exe = cfg.get("unreal_editor")
    if not exe:
        cands = []
        roots = glob.glob(r"C:\Program Files\Epic Games\UE_5.*")
        dat = Path(r"C:\ProgramData\Epic\UnrealEngineLauncher\LauncherInstalled.dat")
        if dat.exists():
            for item in json.loads(dat.read_text(encoding="utf-8")).get("InstallationList", []):
                if re.fullmatch(r"UE_5\.\d+", item.get("AppName", "")):
                    roots.append(item["InstallLocation"])
        for r in set(roots):
            p = Path(r) / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
            m = re.search(r"UE_5\.(\d+)", r)
            if p.exists() and m:
                cands.append((int(m.group(1)), str(p)))
        if not cands:
            sys.exit("Unreal Engine 5 not found: set unreal_editor in config.json to UnrealEditor-Cmd.exe")
        exe = max(cands)[1]
    exe = Path(exe)
    return exe.with_name("UnrealEditor.exe") if gui else exe


def write_cube_dds(path, size=4, value=1.0):
    """A constant-colour float cubemap (the flat sky light's source)."""
    flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000
    head = struct.pack("<4s7I44x", b"DDS ", 124, flags, size, size, size * 16, 0, 1)
    pf = struct.pack("<II4s5I", 32, 0x4, struct.pack("<I", 116), 0, 0, 0, 0, 0)   # D3DFMT_A32B32G32R32F
    caps = struct.pack("<5I", 0x1000 | 0x8, 0x200 | 0xFC00, 0, 0, 0)               # cubemap, all faces
    data = np.full((6, size, size, 4), value, np.float32)
    data[..., 3] = 1.0
    Path(path).write_bytes(head + pf + caps + data.tobytes())


def ensure_project(cfg):
    proj = Path(cfg.get("unreal_project") or ROOT / "unreal_project" / "FreezeFrame.uproject")
    d = proj.parent
    old = d / "GameToBlender.uproject"          # the project's name before 2026-09-30
    if old.exists() and not proj.exists():
        if editor_has_project_open(old):
            proj = old                          # renamed once no editor has it open
        else:
            old.rename(proj)
    for src in TEMPLATE.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(TEMPLATE)
        dst = proj if src.suffix == ".uproject" else d / rel
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    s = d / "GTBSource"
    s.mkdir(exist_ok=True)
    shared = {"default_color": s / "default_color.png", "default_linear": s / "default_linear.png",
              "default_lut": s / "default_lut.png", "white_cube": s / "white_cube.dds"}
    if not shared["default_color"].exists():
        cv2.imwrite(str(shared["default_color"]), np.full((4, 4, 4), 255, np.uint8))
        cv2.imwrite(str(shared["default_linear"]), np.tile(np.array([255, 128, 128, 255], np.uint8), (4, 4, 1)))
        cv2.imwrite(str(shared["default_lut"]), np.full((4, 16, 3), 32768, np.uint16))
        write_cube_dds(shared["white_cube"])
    return proj, {k: str(v) for k, v in shared.items()}


# --------------------------------------------------------------------------- steps

def bake_lut(cfg, cap, look):
    out = cap / "unreal" / "lut.png"
    key_file = cap / "unreal" / "lut.key"
    key = ue_look_mod.lut_key(look)
    if out.exists() and key_file.exists() and key_file.read_text() == key:
        return out
    cmd = [cfg["blender_exe"], "-b", "--factory-startup", "--python", str(ROOT / "ue" / "blender_lut.py"), "--",
           str(cap / "look.json"), str(out), str(ue_look_mod.LUT_SIZE), str(ue_look_mod.LUT_LO),
           str(ue_look_mod.LUT_HI)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists() or "Traceback" in r.stdout + r.stderr:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("LUT bake failed")
    key_file.write_text(key)
    print(f"[ue] colour LUT baked by Blender ({look['view_transform']}, {look['look']}, exposure {look['exposure']:+.2f})")
    return out


def write_look(cfg, cap):
    look = load_look(cap / "look.json")
    plan = json.loads((cap / "unreal" / "plan.json").read_text(encoding="utf-8"))
    lut = bake_lut(cfg, cap, look)
    ul = ue_look_mod.ue_look(look, plan, lut.resolve())
    (cap / "unreal" / "look_ue.json").write_text(json.dumps(ul, indent=1), encoding="utf-8")
    return look, ul


def editor_has_project_open(proj):
    """Unreal editors (not our own commandlets) that have this project loaded."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'UnrealEditor%'\" | "
          "ForEach-Object { $_.CommandLine }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    want = str(Path(proj).resolve()).replace("\\", "/").lower()
    return [line for line in r.stdout.splitlines()
            if want in line.replace("\\", "/").lower() and "-run=pythonscript" not in line.lower()]


def run_editor(cfg, cap, action):
    proj, shared = ensure_project(cfg)
    if editor_has_project_open(proj):
        sys.exit(f"[ue] the Unreal editor has {proj.name} open; close it first (this step saves the level "
                 f"and assets the editor has loaded), then run the command again")
    ue = find_unreal(cfg)
    udir = cap / "unreal"
    log = udir / f"{action}.log"
    job = {"action": action, "plan": str((udir / "plan.json").resolve()),
           "look": str((udir / "look_ue.json").resolve()), "shared": shared, "log": str(log.resolve())}
    job_file = udir / f"job_{action}.json"
    job_file.write_text(json.dumps(job, indent=1), encoding="utf-8")
    if log.exists():
        log.unlink()
    cmd = [str(ue), str(proj), "-run=pythonscript", f"-script={EDITOR_SCRIPT.as_posix()} {job_file.resolve().as_posix()}",
           "-unattended", "-nosplash", "-nullrhi", "-stdout", "-FullStdOutLogOutput"]
    print(f"[ue] {action}: {ue.parent.parent.parent.parent.name} {proj.name} (log: {log})")
    code, _ = _run_streaming(cmd, udir / f"{action}_engine.log", log)
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    if "FAILED" in text or " done (" not in text:
        sys.exit(f"[ue] {action} failed (exit {code}); see {log} and {udir / (action + '_engine.log')}")


def _run_streaming(cmd, engine_log, progress_log=None, patterns=()):
    """Run Unreal, echo our progress log (and matching engine lines) as they come."""
    shown = 0
    with open(engine_log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        pos = 0
        while True:
            done = p.poll() is not None
            if progress_log and Path(progress_log).exists():
                lines = Path(progress_log).read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[shown:]:
                    print("  " + line)
                shown = len(lines)
            if patterns:
                with open(engine_log, encoding="utf-8", errors="replace") as r:
                    r.seek(pos)
                    chunk = r.read()
                    pos = r.tell()
                for line in chunk.splitlines():
                    if any(k in line for k in patterns):
                        print("  " + line.split("]", 2)[-1].strip()[:200])
            if done:
                return p.returncode, shown
            time.sleep(2)


def render(cfg, cap, kind="stills"):
    proj, _ = ensure_project(cfg)
    ue = find_unreal(cfg)
    name = cap.name
    base = f"/Game/GTB/{name}"
    tag = "Stills" if kind == "stills" else "Anim"
    out_dir = cap / "unreal" / "renders" / kind
    if out_dir.exists():
        for f in out_dir.glob("*.png"):
            f.unlink()
    cmd = [str(ue), str(proj), f"{base}/{name}", "-game",
           f"-LevelSequence={base}/Render/LS_{tag}.LS_{tag}",
           f"-MoviePipelineConfig={base}/Render/MRQ_{tag}.MRQ_{tag}",
           "-windowed", "-resx=1280", "-resy=720", "-RenderOffscreen", "-unattended", "-nosplash",
           "-NoLoadingScreen", "-log", "-stdout", "-FullStdOutLogOutput"]
    print(f"[ue] rendering {kind} with Movie Render Queue (first run compiles shaders; can take a while)")
    t0 = time.time()
    code, _ = _run_streaming(cmd, cap / "unreal" / f"render_{kind}_engine.log",
                             patterns=("LogMovieRenderPipeline: Display: [", "Shaders left to compile",
                                       "Fatal error", "LogMovieRenderPipeline: Error"))
    files = sorted(out_dir.glob("*.png"))
    print(f"[ue] {len(files)} frame(s) in {time.time() - t0:.0f}s -> {out_dir}")
    if not files:
        sys.exit(f"[ue] render produced nothing (exit {code}); see {cap / 'unreal' / f'render_{kind}_engine.log'}")
    return files


# --------------------------------------------------------------------------- comparison

def blender_reference(cfg, cap, look):
    """Blender renders of the same look (game camera + close-ups) for the parity sheet.
    Reads scene.blend without saving it."""
    bdir = (cap / "unreal" / "blender").resolve()
    bdir.mkdir(parents=True, exist_ok=True)
    key = json.dumps({k: v for k, v in look.items() if k != "unreal"}, sort_keys=True)   # Blender ignores "unreal"
    if (bdir / "look.key").exists() and (bdir / "look.key").read_text() == key and (bdir / "game.png").exists():
        return bdir
    if not (cap / "scene.blend").exists():
        return None
    print("[ue] rendering the Blender reference for the same look")
    for script, args in (("render.py", [str(cap), str(bdir / "game.png")]),
                         ("closeups.py", [str(cap), str(bdir / "closeup")])):
        cmd = [cfg["blender_exe"], "-b", "--factory-startup", str(cap / "scene.blend"), "--python",
               str(ROOT / "blender" / script), "--", *args]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            return None
    (bdir / "look.key").write_text(key)
    return bdir


def _label(img, text):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)
    return img


def _fit(path, w, h):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR) if path and Path(path).exists() else None
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def compare_all(cfg, cap, look, files):
    from gtb.compare import compare
    udir = cap / "unreal"
    stills = {f.stem.split(".")[0]: f for f in files}
    game = stills.get("GameCamera")
    metrics = compare(cap / "screenshot.png", game, udir / "comparison.png",
                      ignore_mask=look.get("compare_ignore"), label="UNREAL")
    bdir = blender_reference(cfg, cap, look)
    if bdir:
        m_b = compare(bdir / "game.png", game, udir / "parity_diff.png", label="UNREAL")
        metrics["vs_blender"] = {"exposure_offset_stops": m_b["exposure_offset_stops"],
                                 "edge_alignment_within_3px": m_b["edge_alignment_within_3px"],
                                 "region_lum_ratio_unreal_over_blender": m_b["region_lum_ratio_render_over_game"]}
    # Parity sheet: rows = views, columns = game/Blender/Unreal.
    h0 = cv2.imread(str(game)).shape[0]
    w0 = cv2.imread(str(game)).shape[1]
    w, h = w0 // 2, h0 // 2
    rows = [np.concatenate([_label(_fit(cap / "screenshot.png", w, h), "GAME"),
                            _label(_fit(bdir / "game.png" if bdir else None, w, h), "BLENDER"),
                            _label(_fit(game, w, h), "UNREAL")], 1)]
    for i, name in enumerate(("Closeup35", "Closeup15", "CloseupLow")):
        if name in stills:
            rows.append(np.concatenate([_label(np.zeros((h, w, 3), np.uint8), name),
                                        _label(_fit(bdir / f"closeup_{i}.png" if bdir else None, w, h), "BLENDER"),
                                        _label(_fit(stills[name], w, h), "UNREAL")], 1))
    cv2.imwrite(str(udir / "parity.png"), np.concatenate(rows, 0))
    (udir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in ("exposure_offset_stops", "edge_alignment_within_3px")}, indent=1))
    if "vs_blender" in metrics:
        print("[ue] vs Blender:", json.dumps(metrics["vs_blender"]))
    print(f"[ue] comparison -> {udir / 'comparison.png'}\n[ue] game | Blender | Unreal -> {udir / 'parity.png'}")
    return metrics


# --------------------------------------------------------------------------- commands

def smoke_choice_notice(cap, plan=None):
    """The generator smoke has two versions; whoever runs this (often Claude) must ask the user
    which one they want, not pick one (docs/GENERATOR_SMOKE.md)."""
    if plan is None:
        path = cap / "unreal" / "plan.json"
        plan = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not (plan.get("plume") or {}).get("plumes"):
        return
    n = cap.name
    for line in ("GENERATOR SMOKE - ask the user which version they want (don't choose for them):",
                 "  a) sprite column (in the level now, SmokeColumn00): darker flipbook puffs, lit by the fire, "
                 "flames at the base",
                 "  b) Niagara Fluids plume (the look the original user approved): light, billowing smoke rising "
                 "out of the furnace",
                 f"     with the editor open (python gtb.py unreal-open {n}): python -m ue.fluid_smoke {n}",
                 f"  switch any time, instantly: python -m ue.fluid_smoke {n} --show fluid|sprites"
                 "   (docs/GENERATOR_SMOKE.md)"):
        print("[ue] " + line)


def cmd_unreal(cfg, cap, do_render=True):
    plan = ue_export.export(cap)
    look, _ = write_look(cfg, cap)
    run_editor(cfg, cap, "build")
    if do_render:
        compare_all(cfg, cap, look, render(cfg, cap))
    smoke_choice_notice(cap, plan)


def cmd_look(cfg, cap, do_render=True):
    look, _ = write_look(cfg, cap)
    run_editor(cfg, cap, "look")
    if do_render:
        return compare_all(cfg, cap, look, render(cfg, cap))


def cmd_render(cfg, cap, anim=False):
    look = load_look(cap / "look.json")
    files = render(cfg, cap, "anim" if anim else "stills")
    if not anim:
        compare_all(cfg, cap, look, files)


def cmd_calibrate(cfg, cap, rounds=4):
    """Like gtb.py calibrate, but moves look.unreal.exposure_offset (Blender ignores it)."""
    path = cap / "look.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw.get("calibrate") is False:
        print("[ue] calibrate: skipped (look.json has calibrate: false)")
        return cmd_look(cfg, cap)
    tried, prev = [], None
    for i in range(rounds):
        m = cmd_look(cfg, cap)
        u = raw.setdefault("unreal", {})
        exp, off = u.get("exposure_offset", 0.0), m["exposure_offset_stops"]
        tried.append((abs(off), exp))
        print(f"[ue] calibrate {i + 1}: exposure_offset {exp:+.2f}, off by {off:+.2f} stops")
        if abs(off) < 0.08:
            break
        slope = 1.0
        if prev and abs(exp - prev[0]) > 0.05:
            s_ = (off - prev[1]) / (exp - prev[0])
            slope = s_ if 0.2 < s_ < 1.5 else 1.0
        prev = (exp, off)
        u["exposure_offset"] = round(exp + max(-1.0, min(1.0, -off / slope)), 2)
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    raw["unreal"]["exposure_offset"] = min(tried)[1]
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"[ue] calibrate: using exposure_offset {min(tried)[1]:+.2f}")
    cmd_look(cfg, cap)


def cmd_open(cfg, cap):
    proj, _ = ensure_project(cfg)
    subprocess.Popen([str(find_unreal(cfg, gui=True)), str(proj), f"/Game/GTB/{cap.name}/{cap.name}"])
    smoke_choice_notice(cap)
