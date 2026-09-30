"""Texture/material audit of a processed capture: python gtb.py audit <capture>

Writes captures/<name>/audit/:
  materials_NN.png  one row per material of the Unreal plan: name, visible actor count,
                    each texture with its role (K = matched the known-texture database,
                    ? = guessed by the rules) and a crop of the game screenshot where the
                    material's biggest visible mesh is, outlined. Materials with guessed
                    textures come first.
  report.md         the same as text: what to check first, and which material of the
                    reference capture uses the same albedo (names differ between rips).

Look at the sheets next to the screenshot crops: a normal map used as colour shows green,
orange or blue streaks, an atlas shows a patchwork, a mask shows black and white. Fix the
cause in the rules or the database, then `python gtb.py textures <capture>` and
`python -m ue.live materials <capture>` (see CLAUDE.md, texture roles).
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROW_H, THUMB, CROP_W, LABEL_W = 176, 150, 330, 300


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _thumb(path):
    try:
        with open(path, "rb") as fh:
            im = Image.open(fh)
            im.load()
    except Exception:  # noqa: BLE001
        return Image.new("RGB", (THUMB, THUMB), (60, 0, 0))
    im = im.convert("RGB")
    im.thumbnail((THUMB, THUMB))
    return im


def _flags(role, e):
    """Plain-language warnings from a texture's statistics."""
    d = e.get("details") or {}
    r, g, b = (d.get("mean_rgb") or [0.5, 0.5, 0.5])[:3]
    out = []
    if role == "albedo" and abs(r - 0.5) < 0.08 and abs(g - 0.5) < 0.08 and b > 0.7:
        out.append("looks like a normal map")
    if role == "albedo" and d.get("channels") == "AG":
        out.append("packed normal?")
    if role in ("albedo", "gray") and d.get("border_mean", 1) < 0.08:
        out.append("black edges: a mask?")
    if role == "albedo" and max(r, g, b) - min(r, g, b) > 0.45:
        out.append("very saturated")
    return out


def audit(capture: Path, plan=None, manifest=None, out=None):
    """manifest and out are for tests: an edited manifest, another output folder."""
    capture = Path(capture)
    manifest = manifest or json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if plan is None:
        path = capture / "unreal" / "plan.json"
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
        else:
            from ue.export import export
            plan = export(capture)
    tex = manifest["textures"]
    by_file = {e["file"]: t for t, e in tex.items() if e.get("file")}
    meshes = {m["name"]: m for m in manifest["meshes"]}
    actors = {}
    for a in plan["actors"]:
        if not a.get("hidden") and a.get("folder") != "Particles":
            actors.setdefault(a["material"], []).append(a["name"])

    # Game camera projection (manifest camera, Blender world space).
    cam = manifest["camera"]
    Mi = np.linalg.inv(np.array(cam["matrix_world"]))
    W, H = manifest["resolution"]
    f = 1 / math.tan(math.radians(cam["fov_y_deg"]) / 2)
    shot = Image.open(capture / "screenshot.png").convert("RGB")
    sx, sy = shot.width / W, shot.height / H

    def footprint(name):
        """Screen-space triangles of a mesh (screenshot pixels), those in front of the camera."""
        d = np.load(capture / meshes[name]["file"])
        v = np.c_[d["positions"], np.ones(len(d["positions"]))] @ Mi.T
        z = -v[:, 2]
        xy = np.c_[v[:, 0] / np.maximum(z, 1e-6) * f * H / 2 + W / 2, H / 2 - v[:, 1] / np.maximum(z, 1e-6) * f * H / 2]
        tris = d["indices"].reshape(-1, 3)
        tris = tris[(z[tris] > 0.1).all(1)]
        return xy * [sx, sy], tris

    rows = []
    for mat in plan["materials"]:
        users = actors.get(mat["name"])
        if not users:
            continue
        slots = []
        for slot, spec in (mat.get("textures") or {}).items():
            t = by_file.get(spec.get("file"))
            if t:
                e = tex[t]
                slots.append({"slot": slot, "tid": t, "role": e.get("role"), "known": e.get("known"),
                              "flags": _flags(e.get("role"), e), "file": e["file"]})
        # The visible mesh that covers most of the game frame.
        best, best_area, best_geo = None, 0.0, None
        for name in users:
            if name not in meshes:
                continue
            xy, tris = footprint(name)
            if not len(tris):
                continue
            p = xy[tris]
            inside = ((p[..., 0] >= 0) & (p[..., 0] < shot.width) & (p[..., 1] >= 0) & (p[..., 1] < shot.height)).any(1)
            if not inside.any():
                continue
            a2 = np.abs(np.cross(p[inside, 1] - p[inside, 0], p[inside, 2] - p[inside, 0])).sum() / 2
            if a2 > best_area:
                best, best_area, best_geo = name, a2, (xy, tris[inside])
        guessed = [s for s in slots if not s["known"]]
        ref = next((s["known"].get("materials") for s in slots if s["known"] and s["slot"] in ("Albedo", "Atlas")), None)
        rows.append({"mat": mat, "users": users, "slots": slots, "guessed": guessed, "best": best,
                     "area": best_area, "geo": best_geo, "ref": ref,
                     "flags": sorted({fl for s in slots for fl in s["flags"]})})
    # Guessed textures first, then by how much of the game frame the material covers.
    rows.sort(key=lambda r: (not r["guessed"], -r["area"]))

    out = Path(out or capture / "audit")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("materials_*.png"):
        old.unlink()
    font, small = _font(15), _font(12)
    per_sheet = 12
    max_slots = max([len(r["slots"]) for r in rows] + [1])
    width = LABEL_W + max_slots * (THUMB + 10) + CROP_W + 20
    for s0 in range(0, len(rows), per_sheet):
        chunk = rows[s0:s0 + per_sheet]
        sheet = Image.new("RGB", (width, ROW_H * len(chunk)), (24, 24, 28))
        dr = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            y = i * ROW_H
            if i % 2:
                dr.rectangle([0, y, width, y + ROW_H - 1], fill=(32, 32, 38))
            m = r["mat"]
            col = (255, 170, 80) if r["guessed"] else (150, 220, 150)
            dr.text((8, y + 8), m["name"], font=font, fill=col)
            dr.text((8, y + 30), f"{m['parent']}, {len(r['users'])} visible actor(s)", font=small, fill=(200, 200, 200))
            dr.text((8, y + 48), f"e.g. {r['best'] or r['users'][0]}", font=small, fill=(170, 170, 170))
            if r["ref"]:
                dr.text((8, y + 66), "reference: " + ", ".join(list(r["ref"])[:3]), font=small, fill=(150, 200, 255))
            if not r["slots"]:
                dr.text((8, y + 84), "no textures: flat colour", font=small, fill=(200, 200, 120))
            for k, fl in enumerate(r["flags"][:3]):
                dr.text((8, y + 102 + 16 * k), "! " + fl, font=small, fill=(255, 110, 110))
            x = LABEL_W
            for s in r["slots"]:
                th = _thumb(capture / "textures" / s["file"])
                sheet.paste(th, (x, y + 4))
                tag = "K" if s["known"] else "?"
                dr.text((x, y + THUMB + 6), f"{tag} {s['slot']}: {s['tid']} {s['role']}", font=small,
                        fill=(150, 220, 150) if s["known"] else (255, 170, 80))
                x += THUMB + 10
            if r["geo"] is not None:
                xy, tris = r["geo"]
                p = xy[tris].reshape(-1, 2)
                p = p[(p[:, 0] > -shot.width) & (p[:, 0] < 2 * shot.width) & (p[:, 1] > -shot.height) & (p[:, 1] < 2 * shot.height)]
                x0, y0 = np.clip(p.min(0), 0, [shot.width - 1, shot.height - 1])
                x1, y1 = np.clip(p.max(0), 0, [shot.width - 1, shot.height - 1])
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                half_w = max((x1 - x0) / 2 * 1.3, 48.0)
                half_h = max((y1 - y0) / 2 * 1.3, 48.0, half_w * (ROW_H - 12) / CROP_W)
                half_w = max(half_w, half_h * CROP_W / (ROW_H - 12))
                box = [int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)]
                crop = shot.crop(box).resize((CROP_W, ROW_H - 12))
                cd = ImageDraw.Draw(crop)
                scale = CROP_W / (box[2] - box[0])
                for t in tris[:3000]:
                    q = [((xy[j, 0] - box[0]) * scale, (xy[j, 1] - box[1]) * scale) for j in t]
                    cd.polygon(q, outline=(255, 255, 0))
                sheet.paste(crop, (width - CROP_W - 10, y + 6))
            else:
                dr.text((width - CROP_W, y + 70), "not in the game camera's view", font=small, fill=(160, 160, 160))
        sheet.save(out / f"materials_{s0 // per_sheet + 1:02d}.png")

    n_known = sum(1 for e in tex.values() if e.get("known"))
    lines = [f"# Material audit: {capture.name}", "",
             f"{len(rows)} materials on visible surfaces, {len(list(out.glob('materials_*.png')))} sheet(s) "
             f"(`audit/materials_NN.png`, {per_sheet} per sheet).",
             f"Textures: {n_known} of {len(tex)} match the known-texture database (roles checked by eye); "
             f"the rest were classified by the rules.", "",
             "## Check first: materials with guessed textures, biggest on screen first", ""]
    for r in [r for r in rows if r["guessed"]][:40]:
        g = ", ".join(f"{s['slot']} {s['tid']} ({s['role']})" for s in r["guessed"])
        fl = f"; flags: {', '.join(r['flags'])}" if r["flags"] else ""
        lines.append(f"- **{r['mat']['name']}** ({len(r['users'])} actors, e.g. `{r['best'] or r['users'][0]}`): {g}{fl}")
    lines += ["", "## Same albedo as the reference capture", ""]
    for r in rows:
        if r["ref"]:
            lines.append(f"- {r['mat']['name']} = reference {', '.join(r['ref'])}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, rows
