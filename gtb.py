"""FreezeFrame command line.

  python gtb.py watch                 run the capture daemon (hotkey -> screenshot + rip)
  python gtb.py import <rip folder> [--name N]  make a capture from an existing Ninja Ripper rip
  python gtb.py process <capture>     parse the rip into manifest.json + meshes + textures
  python gtb.py textures <capture>    re-run the texture role rules on a processed capture (no rip needed)
  python gtb.py audit <capture>       material sheets: textures, roles (known or guessed), game screenshot crops
  python gtb.py known <capture> [texture id] [--role R] [--what "note"] [--merge other.json]
                                      the known-texture database: list the capture's guessed textures, show
                                      one texture's entry, add/update it with a checked role (normal:AG, albedo, ...),
                                      or merge a copy of the database that someone sent
  python gtb.py build <capture>       build scene.blend (runs Blender headless)
  python gtb.py render <capture>      re-apply look.json, render preview, write comparison.png
  python gtb.py calibrate <capture>   auto-match exposure to the screenshot (part of `all`)
  python gtb.py closeups <capture>    render close-up checks (camera dollied in)
  python gtb.py look <capture> <name> apply a look preset (profiles/looks/), e.g. frostpunk_day
  python gtb.py open <capture>        open scene.blend in Blender
  python gtb.py all <capture>         process + build + calibrate

Blender is needed only for build, render, calibrate, closeups, open and all. The Unreal
path is `process` then `unreal`, without Blender.

  python gtb.py unreal <capture>           build an Unreal level (UE 5.x) and render/compare it
  python gtb.py unreal-look <capture>      re-apply look.json to the Unreal level, render, compare
  python gtb.py unreal-render <capture>    render stills (--anim: the Level Sequence)
  python gtb.py unreal-calibrate <capture> match Unreal's exposure to the screenshot
  python gtb.py unreal-open <capture>      open the level in the Unreal editor

<capture> may be a folder name under captures/ or "latest".
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from gtb import config

ROOT = Path(__file__).resolve().parent
BLENDER_DIR = ROOT / "blender"


def resolve_capture(cfg, name):
    base = Path(cfg["captures_dir"])
    if name == "latest":
        caps = sorted((p for p in base.iterdir() if (p / "capture.json").exists()
                       and json.loads((p / "capture.json").read_text()).get("rip_dir")),
                      key=lambda p: p.stat().st_mtime)
        if not caps:
            sys.exit("no captures yet")
        return caps[-1]
    p = Path(name)
    return p if p.is_absolute() or p.exists() else base / name


def has_blender(cfg):
    return Path(cfg["blender_exe"]).exists()


def need_blender(cfg):
    if not has_blender(cfg):
        sys.exit(f"Blender isn't installed ({cfg['blender_exe']}; set blender_exe in config.json). "
                 "The Unreal path doesn't need it: python gtb.py process <capture>, then python gtb.py unreal <capture>.")


def run_blender(cfg, args):
    need_blender(cfg)
    cmd = [cfg["blender_exe"], "-b", "--factory-startup", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        if line.startswith(("[gtb]", "Error", "Traceback")) or "Error:" in line:
            print(line)
    if r.returncode != 0 or "Traceback" in r.stdout + r.stderr:
        print(r.stderr[-3000:])
        sys.exit("blender step failed")


def cmd_process(cfg, cap):
    from gtb import process
    process.process(cap)


def cmd_build(cfg, cap):
    run_blender(cfg, ["--python", str(BLENDER_DIR / "build.py"), "--", str(cap)])


def cmd_render(cfg, cap, save=False, quiet=False):
    renders = cap / "renders"
    renders.mkdir(exist_ok=True)
    n = len(list(renders.glob("render_*.png")))
    out = renders / f"render_{n:03d}.png"
    extra = ["--save"] if save else []
    run_blender(cfg, [str(cap / "scene.blend"), "--python", str(BLENDER_DIR / "render.py"),
                      "--", str(cap), str(out), *extra])
    from gtb.compare import compare
    look = json.loads((cap / "look.json").read_text()) if (cap / "look.json").exists() else {}
    metrics = compare(cap / "screenshot.png", out, cap / "comparison.png",
                      ignore_mask=look.get("compare_ignore"))
    (renders / f"render_{n:03d}.json").write_text(json.dumps({"look": look, "metrics": metrics}, indent=1))
    if not quiet:
        print(json.dumps(metrics, indent=1))
        print(f"[gtb] comparison -> {cap / 'comparison.png'}")
    return metrics


def cmd_calibrate(cfg, cap, rounds=4):
    """Shift look.exposure until the render's median brightness matches the game's.

    The view transform compresses highlights, so a 1-stop exposure change moves
    the output by less than a stop; a secant step learns that slope. Steps are
    capped (render noise can fake a tiny slope) and the best exposure tried wins."""
    look_path = cap / "look.json"
    look = json.loads(look_path.read_text()) if look_path.exists() else {}
    if look.get("calibrate") is False:  # e.g. a day look for a night capture
        print("[gtb] calibrate: skipped (look.json has calibrate: false)")
        cmd_render(cfg, cap, save=True)
        return
    tried, prev = [], None
    for i in range(rounds):
        metrics = cmd_render(cfg, cap, quiet=True)
        exp, off = look.get("exposure", 0.0), metrics["exposure_offset_stops"]
        tried.append((abs(off), exp))
        print(f"[gtb] calibrate {i + 1}: exposure {exp:+.2f}, off by {off:+.2f} stops")
        if abs(off) < 0.08:
            break
        slope = 1.0
        if prev and abs(exp - prev[0]) > 0.05:
            s_ = (off - prev[1]) / (exp - prev[0])
            if 0.2 < s_ < 1.5:  # anything else is noise
                slope = s_
        prev = (exp, off)
        step = max(-1.0, min(1.0, -off / slope))
        look["exposure"] = round(exp + step, 2)
        look_path.write_text(json.dumps(look, indent=2))
    best = min(tried)[1]
    look["exposure"] = best
    look_path.write_text(json.dumps(look, indent=2))
    print(f"[gtb] calibrate: using exposure {best:+.2f}")
    cmd_render(cfg, cap, save=True)


def cmd_closeups(cfg, cap):
    renders = cap / "renders"
    renders.mkdir(exist_ok=True)
    run_blender(cfg, [str(cap / "scene.blend"), "--python", str(BLENDER_DIR / "closeups.py"),
                      "--", str(cap), str((renders / "closeup").resolve())])


def cmd_look(cfg, cap, name):
    """Copy a preset from profiles/looks/ into the capture's look.json, then render."""
    src = ROOT / "profiles" / "looks" / f"{name}.json"
    if not src.exists():
        names = sorted(p.stem for p in (ROOT / "profiles" / "looks").glob("*.json"))
        sys.exit(f"no look preset '{name}'. Available: {', '.join(names)}")
    look_path = cap / "look.json"
    if look_path.exists():
        (cap / "look_previous.json").write_text(look_path.read_text())
    look_path.write_text(src.read_text())
    print(f"[gtb] look '{name}' applied (previous saved as look_previous.json)")
    cmd_calibrate(cfg, cap)


def cmd_import(cfg, src, name=None):
    """Create captures/<name>/capture.json for a rip that is already on disk (a
    frame_NNNN folder, or Ninja Ripper's per-process folder: its newest frame).
    `process` then reads the rip and uses Ninja Ripper's own !screenshot.dds."""
    import re
    from datetime import datetime
    from gtb import nr
    rip = Path(src).resolve()
    if not any(rip.glob("*.nr")):
        frames = sorted(rip.glob("frame_*"))
        rip = frames[-1] if frames else rip
    if not any(rip.glob("*.nr")):
        sys.exit(f"no .nr files in {rip}")
    desc = nr.read_ripdesc(rip) or nr.read_ripdesc(rip.parent)
    m = re.match(r".*?_([^_]+\.exe)_\d+$", rip.parent.name, re.IGNORECASE)
    exe = desc.get("executable") or (m.group(1) if m else "")
    size = [int(desc.get("width") or 0), int(desc.get("height") or 0)]
    shot = next(rip.glob("*screenshot*"), None)
    if (not all(size)) and shot is not None:
        from PIL import Image
        with Image.open(shot) as im:
            size = list(im.size)
    stamp = datetime.fromtimestamp(rip.stat().st_mtime)
    name = name or f"{Path(exe).stem or 'game'}_{stamp:%Y%m%d_%H%M%S}"
    out = Path(cfg["captures_dir"]) / name
    if (out / "capture.json").exists():
        sys.exit(f"{out} already exists; pick another --name")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"game_exe": exe, "window_title": "", "resolution": size, "captured_at": stamp.isoformat(timespec="seconds"),
            "rip_dir": str(rip), "rip_files": [], "imported": True}
    (out / "capture.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[gtb] capture {out.name}: {exe} {size[0]}x{size[1]} from {rip}"
          f"{'' if shot else ' (no Ninja Ripper screenshot found: add screenshot.png yourself)'}")
    print(f"[gtb] next: python gtb.py all {out.name}" if has_blender(cfg) else
          f"[gtb] next: python gtb.py process {out.name}, then python gtb.py unreal {out.name}")
    return out


def cmd_open(cfg, cap):
    need_blender(cfg)
    subprocess.Popen([cfg["blender_exe"], str(cap / "scene.blend")])


def on_capture(cfg):
    def handler(cap):
        if not cfg["auto_build"]:
            return
        try:
            cmd_process(cfg, cap)
            if not has_blender(cfg):
                print(f"[gtb] processed; no Blender, so no scene.blend. Unreal: python gtb.py unreal {cap.name}")
                return
            cmd_build(cfg, cap)
            cmd_calibrate(cfg, cap)
            if cfg["auto_open"]:
                cmd_open(cfg, cap)
        except SystemExit as e:
            print(f"[gtb] pipeline stopped: {e}")
        except Exception as e:  # keep the daemon alive for the next capture
            import traceback
            traceback.print_exc()
            print(f"[gtb] pipeline failed: {e}")
    return handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["watch", "import", "process", "textures", "audit", "known", "build", "render", "calibrate", "closeups", "look", "open",
                                        "all", "unreal", "unreal-look", "unreal-render", "unreal-calibrate",
                                        "unreal-open"])
    ap.add_argument("capture", nargs="?", default="latest")
    ap.add_argument("preset", nargs="?", help="look: preset name from profiles/looks/; known: a texture id (t0018)")
    ap.add_argument("--role", help="known: the checked role: albedo, normal:AG (or :RG, :RGB), gray, packed, shared, detail")
    ap.add_argument("--what", help="known: a note on what the texture is")
    ap.add_argument("--merge", metavar="FILE", help="known: merge another copy of the database (a .json someone sent)")
    ap.add_argument("--save", action="store_true", help="render: also save look into scene.blend")
    ap.add_argument("--save-known", action="store_true",
                    help="textures: add this capture's texture roles to the known-texture database (only once "
                         "they are checked, e.g. with `audit`)")
    ap.add_argument("--name", help="import: capture folder name (default <game>_<rip time>)")
    ap.add_argument("--anim", action="store_true", help="unreal-render: render the animation sequence")
    ap.add_argument("--no-render", action="store_true", help="unreal / unreal-look: skip rendering")
    a = ap.parse_args()
    cfg = config.load()

    if a.command == "watch":
        from gtb.capture import CaptureDaemon
        CaptureDaemon(cfg, on_capture=on_capture(cfg)).run()
        return
    if a.command == "import":
        cmd_import(cfg, a.capture, a.name)
        return
    cap = resolve_capture(cfg, a.capture)
    if a.command == "process":
        cmd_process(cfg, cap)
    elif a.command == "textures":
        from gtb.textures import export_known, profile_of, reclassify
        if a.save_known:
            prof = profile_of(json.loads((cap / "manifest.json").read_text(encoding="utf-8")))
            if not prof.get("known_textures"):
                sys.exit("the game profile has no known_textures database to add to")
            n = export_known(cap, ROOT / "profiles" / prof["known_textures"])
            print(f"[textures] profiles/{prof['known_textures']}: {n} textures, {cap.name}'s roles added as checked")
            return
        changed = reclassify(cap)
        for tid, (old, new) in sorted(changed.items()):
            print(f"[textures] {tid}: {old[0]}{':' + old[1] if old[1] else ''} -> {new[0]}{':' + new[1] if new[1] else ''}")
        print(f"[textures] {len(changed)} role(s) changed" + ("; rebuild with `build` (Blender) and `unreal`, or "
              "`python -m ue.live materials` with the editor open" if changed else ""))
    elif a.command == "known":
        from gtb import textures as T
        manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
        prof = T.profile_of(manifest)
        if not T.known_path(prof):
            sys.exit("the game profile has no known_textures database")
        tex = manifest["textures"]
        if a.merge:
            added, updated = T.merge_known(prof, a.merge)
            print(f"[known] merged {a.merge} into profiles/{prof['known_textures']}: {added} new texture(s), "
                  f"{updated} already there. Apply it: python gtb.py textures {cap.name}")
            return

        def hashes(e):
            src = Path(e.get("source", ""))
            return e.get("hashes") or (T.mip_hashes(src) if src.suffix.lower() == ".dds" and src.exists() else [])
        if not a.preset:
            found = {t for t, e in tex.items() if T.known_get(prof, hashes(e))}
            print(f"[known] {len(found)} of {len(tex)} textures of {cap.name} are in profiles/{prof['known_textures']}; "
                  f"the others were classified by the rules:")
            for t, e in tex.items():
                if t in found:
                    continue
                ch = (e.get("details") or {}).get("channels")
                why = "" if hashes(e) else "  (no hashes: a single colour, a cube map or no DDS; can't be in the database)"
                print(f"  {t}  {e.get('role')}{':' + ch if ch and e.get('role') == 'normal' else ''}  {e.get('size')}  {e.get('file')}{why}")
            return
        if a.preset not in tex:
            sys.exit(f"{cap.name} has no texture {a.preset}")
        e = tex[a.preset]
        if a.role:
            role, _, channels = a.role.partition(":")
            entry = T.known_add(prof, hashes(e), role, channels or None, a.what, f"{cap.name}/{a.preset}", size=e.get("size"))
            print(f"[known] {a.preset} saved as {entry['role']}{':' + entry['channels'] if entry.get('channels') else ''} "
                  f"(entry {entry['ref']}). Apply it: python gtb.py textures {cap.name}")
        else:
            entry = T.known_get(prof, hashes(e))
            print(f"[known] {a.preset} in {cap.name}: role {e.get('role')}, {e.get('size')}, {e.get('file')}")
            print("[known] database: " + (json.dumps({k: v for k, v in entry.items() if k != "hashes"}) if entry
                                          else "not in it (the role above is the rules' guess)"))
    elif a.command == "audit":
        from gtb.audit import audit
        out, rows = audit(cap)
        guessed = sum(1 for r in rows if r["guessed"])
        print(f"[audit] {len(rows)} materials, {guessed} with guessed textures -> {out} (report.md, materials_NN.png)")
    elif a.command == "build":
        cmd_build(cfg, cap)
    elif a.command == "render":
        cmd_render(cfg, cap, a.save)
    elif a.command == "calibrate":
        cmd_calibrate(cfg, cap)
    elif a.command == "look":
        cmd_look(cfg, cap, a.preset or "frostpunk_night")
    elif a.command == "closeups":
        cmd_closeups(cfg, cap)
    elif a.command == "open":
        cmd_open(cfg, cap)
    elif a.command == "all":
        cmd_process(cfg, cap)
        cmd_build(cfg, cap)
        cmd_calibrate(cfg, cap)
    elif a.command.startswith("unreal"):
        from ue import pipeline
        if a.command == "unreal":
            pipeline.cmd_unreal(cfg, cap, do_render=not a.no_render)
        elif a.command == "unreal-look":
            pipeline.cmd_look(cfg, cap, do_render=not a.no_render)
        elif a.command == "unreal-render":
            pipeline.cmd_render(cfg, cap, anim=a.anim)
        elif a.command == "unreal-calibrate":
            pipeline.cmd_calibrate(cfg, cap)
        elif a.command == "unreal-open":
            pipeline.cmd_open(cfg, cap)


if __name__ == "__main__":
    main()
