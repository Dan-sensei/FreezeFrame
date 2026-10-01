"""Turn a Ninja Ripper frame rip into a FreezeFrame manifest.

Geometry comes from the post-vertex-shader stream (clip-space positions, i.e.
exactly where the game drew it). Clip -> view space only needs the projection's
x/y scale (P00, P11): view = (xc/P00, yc/P11, w). We solve P00/P11 from draws
that also carry their pre-VS (object-space) positions: for a rigid object the
object->view map must be a rotation * uniform scale, which pins both values.

View space is D3D (left-handed, +Y up, +Z forward). We convert to Blender
camera-local (x, y, -z) and then rotate the whole scene so the dominant
ground plane faces +Z, which puts the camera at the origin with its real pitch.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from gtb import characters, nr, textures
from gtb.scene_common import assign_slots, below_scene, decal_volume, overlay_layer, scene_floor
from gtb.config import ROOT

PROFILES = ROOT / "profiles"


# --------------------------------------------------------------------------- helpers

def load_profile(exe: str):
    name = Path(exe or "").stem.lower()
    default = json.loads((PROFILES / "default.json").read_text()) if (PROFILES / "default.json").exists() else {}
    for p in PROFILES.glob("*.json"):
        prof = json.loads(p.read_text())
        if name and name in [Path(e).stem.lower() for e in prof.get("executables", [])]:
            print(f"[process] using profile {p.name}")
            return {**default, **prof, "_name": p.stem}
    return {**default, "_name": "default"}


def pick_reference_screenshot(capture: Path, rip: Path, size, used_textures):
    """Prefer Ninja Ripper's own frame screenshot (no overlay) over the daemon's.

    NR saves one when saveScreenshots is on; its name isn't documented, so take a
    backbuffer-sized image in the rip folder that no draw call uses as a texture."""
    from PIL import Image
    cands = sorted(rip.rglob("*"), key=lambda f: ("screenshot" not in f.name.lower(), f.name))
    for f in cands:
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".dds", ".tga") or f.name in used_textures:
            continue
        try:
            with Image.open(f) as im:
                if tuple(im.size) != tuple(size):
                    continue
                if not (capture / "screenshot_window.png").exists() and (capture / "screenshot.png").exists():
                    (capture / "screenshot.png").rename(capture / "screenshot_window.png")
                im.convert("RGB").save(capture / "screenshot.png")
        except Exception:
            continue
        print(f"[process] reference screenshot: Ninja Ripper's {f.name}")
        return f.name
    print("[process] reference screenshot: daemon window capture")
    return None


def clip_position(stream):
    """(N,4) clip-space positions from a post-VS stream."""
    cand = [a for a in stream.attrs if "POSITION" in a.semantic.upper() and a.comps == 4 and a.va_type == 0]
    if not cand:
        cand = [a for a in stream.attrs if a.comps == 4 and a.va_type == 0]
    return stream.read(cand[0]).astype(np.float64) if cand else None


def object_position(stream):
    a = stream.find("POSITION") if stream else None
    if a is None or a.va_type != 0 or a.comps < 3:
        return None
    return stream.read(a)[:, :3].astype(np.float64)


def decode_vector(stream, attr, comps=3):
    """Normals/tangents stored as float, unorm/snorm 8/16 or packed 10:10:10:2."""
    v = stream.read(attr)
    t = attr.va_type
    if t == 0:
        out = v[:, :comps].astype(np.float64)
    elif t == 3:
        out = v[:, :comps] / 255.0 * 2 - 1
    elif t == 6:
        out = np.maximum(v[:, :comps] / 127.0, -1)
    elif t == 2:
        out = v[:, :comps] / 65535.0 * 2 - 1
    elif t == 5:
        out = np.maximum(v[:, :comps] / 32767.0, -1)
    elif t == 1 and attr.comps == 1:  # R10G10B10A2 packed in one u32
        u = v[:, 0].astype(np.uint64)
        out = np.stack([(u >> (10 * i)) & 1023 for i in range(3)], 1) / 1023.0 * 2 - 1
    else:
        return None
    if out.shape[1] < 3:
        return None
    length = np.linalg.norm(out, axis=1)
    if not (0.7 < np.median(length) < 1.3):
        return None
    return out / np.maximum(length[:, None], 1e-8)


def decode_uv(stream, attr, mode="auto"):
    v = stream.read(attr)[:, :2]
    t = attr.va_type
    if t == 0:
        return v.astype(np.float64)
    if t in (2, 5) and mode in ("auto", "half"):
        h = v.astype(np.uint16).view(np.float16).astype(np.float64)
        if mode == "half" or (np.isfinite(h).mean() > .99 and np.nanmax(np.abs(h)) < 256):
            return np.nan_to_num(h)
    if mode.startswith("scale:"):
        return v.astype(np.float64) * float(mode[6:])
    if t == 2:
        return v / 65535.0
    if t == 5:
        return v / 32767.0
    if t == 3:
        return v / 255.0
    return None


def fit_affine(src, dst):
    """Least squares dst ~ [src,1] @ M. Returns (M (4,k), relative residual)."""
    X = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    err = np.linalg.norm(X @ M - dst) / max(np.linalg.norm(dst - dst.mean(0)), 1e-12)
    return M, err


# --------------------------------------------------------------------------- per-draw analysis

class DrawData:
    """One draw call's geometry in a common form.

    clip  (V,4) clip-space positions, tris (T,3) into clip,
    rows  (V,)  pre-VS stream row for each clip vertex (for UVs/normals), or None,
    obj   (V,3) object-space positions aligned with clip (rigid fit), or None.

    Ninja Ripper stores post-VS data either indexed (post[j] = VS(pre[j])) or
    expanded, one vertex per index (post[k] = VS(pre[idx[k]])), as NR 2.18 does
    for Frostpunk. Expanded draws are re-indexed back to unique vertices."""

    def __init__(self, d: nr.Draw):
        self.d = d
        self.post = d.stream_for("vs")
        self.pre = d.stream_for("pre_vs")
        self.clip = self.tris = self.rows = self.obj = None
        self.lin = None  # 3x3 object -> (xc, yc, w) linear part
        self.fit_err = None
        self.instanced = False
        raw = clip_position(self.post) if self.post else None
        if raw is None:
            return
        n = len(raw)
        npre = self.pre.count if self.pre else 0
        idx = d.indices
        if idx is not None:
            idx = idx[: len(idx) // 3 * 3]
        if idx is not None and len(idx) and n % len(idx) == 0 and self._expanded(raw, idx, npre):
            k = n // len(idx)
            ok_rows = npre and int(idx.max()) < npre
            if k == 1:
                u, inv = np.unique(idx, return_inverse=True)
                clip = np.zeros((len(u), 4))
                clip[inv] = raw
                self.clip, self.tris = clip, inv.reshape(-1, 3)
                self.rows = u if ok_rows else None
            else:
                self.instanced = True
                self.clip, self.tris = raw, np.arange(n).reshape(-1, 3)
                self.rows = np.tile(idx, k) if ok_rows else None
        else:
            if idx is None or not len(idx):
                idx = np.arange(n - n % 3)
            m = int(idx.max()) + 1 if len(idx) else 0
            k = n // m if m and n > m and n % m == 0 else 1
            if k > 1:  # instances back to back
                self.instanced = True
                idx = np.concatenate([idx + i * m for i in range(k)])
            tris = idx.reshape(-1, 3)
            self.clip, self.tris = raw, tris[tris.max(1) < n]
            if npre and n % npre == 0:
                self.rows = np.tile(np.arange(npre), n // npre)
        if self.rows is not None and not self.instanced:
            pos = object_position(self.pre)
            if pos is not None:
                self.obj = pos[self.rows]

    @staticmethod
    def _expanded(raw, idx, npre):
        n = len(raw)
        if n != len(idx) * (n // len(idx)):
            return False
        if npre and n != npre:
            return True  # post count follows the index count, not the vertex count
        # Same counts: expanded iff repeated indices produce identical outputs.
        order = np.argsort(idx[: n], kind="stable")
        same = idx[order][1:] == idx[order][:-1]
        if not same.any():
            return False
        a, b = order[1:][same][:200], order[:-1][same][:200]
        return bool(np.allclose(raw[a], raw[b], atol=1e-5))


def layout_key(stream):
    return ",".join(a.name for a in stream.attrs) if stream else ""


def classify_draw(dd: DrawData, cfg):
    d = dd.d
    rs = d.render_state
    if d.topology != "triangles":
        return "not_triangles"
    if cfg.get("layout_categories", {}).get(layout_key(dd.pre)) == "skip":
        return "profile_layout_skip"
    if cfg.get("skip_untextured", True) and not d.textures:
        return "untextured"
    if dd.clip is None:
        return "no_post_vs_position"
    if cfg.get("skip_color_write_disabled", True) and rs.get("RGBWRITE_DISABLED"):
        return "depth_only"
    if cfg.get("skip_offsize_rt", True) and rs.get("RT_WIDTH_NOT_MATCH_BACKBUF"):
        return "offscreen_target"
    w = dd.clip[:, 3]
    if np.abs(w - 1).max() < 1e-4:
        return "screen_space"
    c = dd.clip
    inside = (np.abs(c[:, 0]) <= c[:, 3]) & (np.abs(c[:, 1]) <= c[:, 3]) & (c[:, 3] > 0)
    if not inside.any():
        return "off_screen"
    return None


def solve_projection(draws, cfg):
    """Median P00/P11 over rigid draws. Returns (p00, p11, n_used)."""
    fx, fy = [], []
    for dd in draws:
        if dd.obj is None or len(dd.obj) < 8:
            continue
        span = np.ptp(dd.obj, axis=0)
        if (span > 1e-6).sum() < 3:  # flat/degenerate: rotation not recoverable
            continue
        M, err = fit_affine(dd.obj, dd.clip[:, [0, 1, 3]])
        dd.fit_err = err
        if err > 1e-3:
            continue  # skinned / wind / vertex-animated
        L = M[:3].T  # rows: xc, yc, w as functions of object position
        dd.lin = L
        nx, ny, nw = np.linalg.norm(L, axis=1)
        if nw < 1e-9:
            continue
        # With x_v = xc/P00, y_v = yc/P11 the rows must have equal norms and be
        # orthogonal. Check orthogonality to reject shears / odd matrices.
        U = L / np.linalg.norm(L, axis=1, keepdims=True)
        if np.abs(U @ U.T - np.eye(3)).max() > 0.02:
            continue
        fx.append(nx / nw)
        fy.append(ny / nw)
    if len(fy) < cfg.get("min_rigid_draws", 3):
        return None, None, len(fy)
    return float(np.median(fx)), float(np.median(fy)), len(fy)


def estimate_up(tri_positions, min_up_dot=0.25):
    """Dominant plane normal (area weighted), sign chosen so it points up-ish.

    A floor seen from above and a wall facing the camera look alike, so the
    profile's min_up_dot sets how far from the camera's up axis the ground may
    be: ~0.25 for third-person games, 0 for top-down city builders."""
    S = np.zeros((3, 3))
    for P in tri_positions:
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        n = np.cross(b - a, c - a)
        area = np.linalg.norm(n, axis=1)
        ok = area > 0
        if not ok.any():
            continue
        u = n[ok] / area[ok, None]
        S += (u * area[ok, None]).T @ u
    vals, vecs = np.linalg.eigh(S)
    cam_up = np.array([0.0, 1.0, 0.0])  # Blender camera-local up
    # Prefer the strongest orientation that isn't nearly perpendicular to the
    # camera's own up axis (that would be a wall facing the camera).
    toward_cam = np.array([0.0, 0.0, 1.0])
    for i in np.argsort(vals)[::-1]:
        v = vecs[:, i]
        if abs(v @ cam_up) >= min_up_dot:
            if abs(v @ cam_up) > 0.05:
                return v if v @ cam_up > 0 else -v
            return v if v @ toward_cam > 0 else -v
    return cam_up


def screen_winding(draws):
    """+1 if most on-screen triangle area is counter-clockwise, -1 if clockwise."""
    total = 0.0
    for dd in draws[:400]:
        c = dd.clip
        tris = dd.tris
        if len(tris) == 0:
            continue
        w = c[:, 3:4]
        ok = (w[:, 0] > 1e-6)
        ndc = np.where(ok[:, None], c[:, :2] / np.where(ok[:, None], w, 1), 0)
        t = tris[ok[tris].all(1)]
        a, b, cc = ndc[t[:, 0]], ndc[t[:, 1]], ndc[t[:, 2]]
        area = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
        total += np.sign(area).sum()
    return 1 if total >= 0 else -1


def rotation_to_z(up):
    """Rotation matrix taking `up` to +Z while keeping camera-forward heading."""
    z = up / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, -1.0])  # camera looks down -Z in Blender camera-local
    y = fwd - (fwd @ z) * z
    if np.linalg.norm(y) < 1e-6:  # looking straight down
        y = np.array([0.0, 1.0, 0.0]) - z[1] * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])  # rows: world axes expressed in camera-local coords


def sprite_info(b, tex_ids, tex_entries, lut_names):
    """Pick the flipbook sheet and classify the sprite: smoke (normal-mapped,
    lit), fire (coloured, emissive) or None for near-invisible specks."""
    cands = [tex_ids[n] for n in b["tex"] if n in tex_ids and n not in lut_names and tex_ids[n] in tex_entries]
    cands = [t for t in cands if "alpha_mean" in tex_entries[t].get("details", {})]
    if not cands:
        return None
    atlas = max(cands, key=lambda t: tex_entries[t]["details"]["alpha_std"])
    e = tex_entries[atlas]
    if e["details"]["alpha_mean"] < 0.03:
        return None
    r, g, bl = e["details"].get("mean_rgb", [0, 0, 0])
    kind = "fire" if e["role"] == "albedo" and r > 0.45 and r > 1.8 * bl else "smoke"
    return {"kind": kind, "atlas": atlas}


def sprite_attributes(pos, tris, R):
    """Per-vertex puff centre and corner offset in the game camera's plane, so
    Blender can re-face every puff toward whatever camera renders it."""
    n = len(pos)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b_, c in tris:
        for u, v in ((a, b_), (a, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    centre = np.zeros((comp.max() + 1, 3))
    np.add.at(centre, comp, pos)
    centre /= np.bincount(comp)[:, None]
    c = centre[comp]
    right, up = R[:, 0], R[:, 1]  # game camera axes in world space
    off = pos - c
    corner = np.stack([off @ right, off @ up], 1)
    return {"attr_p_center": c.astype(np.float32), "attr_p_corner": corner.astype(np.float32)}


def find_snow_box(capture, meshes, built, cfg):
    """World-space box where snow falls.

    Uses the game's own snowflake sprites (profile snow_layouts) when present;
    otherwise the surfaces' footprint from the ground up to the camera."""
    pts = []
    layouts = set(cfg.get("snow_layouts", []))
    for mm, b in zip(meshes, built):
        if layouts and layout_key(b["dd"].pre) in layouts:
            pts.append(np.load(capture / mm["file"])["positions"])
    surf = np.vstack([np.load(capture / mm["file"])["positions"][::7]
                      for mm in meshes if mm["category"] == "surface"])
    ground = float(np.percentile(surf[:, 2], 2))
    if pts:
        P = np.vstack(pts)
        lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    else:
        lo, hi = np.percentile(surf, 2, axis=0), np.percentile(surf, 98, axis=0)
    lo[2] = ground
    hi[2] = max(hi[2], 0.0)  # up to the camera (which sits at the origin)
    return [lo.round(2).tolist(), hi.round(2).tolist()]


def find_smoke_plumes(capture, meshes, built, cfg):
    """Smoke sources for Blender's volumetric plumes.

    Games draw rising smoke as a column mesh textured from a live render
    target, which the ripper can't save; so an effect draw whose textures all
    failed to save is taken as a smoke column. Its footprint becomes the plume
    base. (Particle sprites turned out to be snowfall, lamp glows and flames.)"""
    plumes = []
    for mm, b in zip(meshes, built):
        if mm["category"] != "effect" or not b["tex"]:
            continue
        if any((b["dd"].d.file.parent / t).exists() for t in b["tex"]):
            continue
        P = np.load(capture / mm["file"])["positions"]
        lo, hi = P.min(0), P.max(0)
        c = (lo + hi) / 2
        r = float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2)
        n = 6
        pts = [[c[0], c[1], lo[2] + (hi[2] - lo[2]) * k / (n - 1), r] for k in range(n)]
        plumes.append({"points": np.round(pts, 3).tolist(), "source": mm["name"]})
    return plumes


# --------------------------------------------------------------------------- main

def process(capture: Path):
    capture = Path(capture)
    meta = json.loads((capture / "capture.json").read_text())
    rip = Path(meta["rip_dir"])
    desc = nr.read_ripdesc(rip)
    exe = desc.get("executable") or meta.get("game_exe", "")
    cfg = load_profile(exe)
    width = int(desc.get("width") or meta["resolution"][0])
    height = int(desc.get("height") or meta["resolution"][1])
    aspect = width / height
    print(f"[process] rip {rip.name}: {exe} {width}x{height}")

    files = [rip / f for f in meta["rip_files"]] if meta.get("rip_files") else sorted(rip.rglob("*.nr"))
    draws, reasons = [], Counter()
    light_volumes = []

    # Deferred renderers draw light volumes, decal boxes and fog volumes as real
    # 3D meshes; they read the screen's own buffers (depth, normals, lighting),
    # which are the only screen-sized textures. Real surfaces never sample them.
    screen_sizes = {(round(width / k), round(height / k)) for k in (1, 2, 4, 8)}
    header_size = {}

    def samples_screen_buffer(d):
        for _, _, name in d.textures:
            if name not in header_size:
                # A texture the draw names may not be in the folder (an incomplete copy of a rip).
                path = d.file.parent / name
                info = textures.dds_info(path) if name.lower().endswith(".dds") and path.exists() else None
                header_size[name] = (info["width"], info["height"]) if info else None
            sz = header_size[name]
            if sz and any(abs(sz[0] - w) <= 2 and abs(sz[1] - h) <= 2 for w, h in screen_sizes):
                return True
        return False

    for f in files:
        try:
            parsed = nr.read_nr(f)
        except Exception as e:
            reasons[f"parse_error"] += 1
            print(f"[process] {f.name}: {e}")
            continue
        if any(d.stage != "pre_vs" for d in parsed):
            parsed = [d for d in parsed if d.stage != "pre_vs"]
        for d in parsed:
            dd = DrawData(d)
            why = classify_draw(dd, cfg)
            if not why and cfg.get("skip_screen_buffer_draws", True) and samples_screen_buffer(d):
                why = "samples_screen_buffer"
                # Deferred lights are drawn as bare sphere/hemisphere meshes that read
                # the G-buffer: their position and size are the light's.
                if layout_key(dd.pre) in ("POSITION0", "POSITION0,TEXCOORD0") and len(dd.tris) >= 200                         and dd.clip is not None and np.abs(dd.clip[:, 3] - 1).max() > 1e-4:
                    light_volumes.append(dd)
            if why:
                reasons[why] += 1
                continue
            draws.append(dd)
    print(f"[process] {len(files)} files, {len(draws)} draws kept, skipped: {dict(reasons)}")
    if not draws:
        raise SystemExit("no usable draws - is Ninja Ripper saving post-VS (world space) geometry?")

    p00, p11, n_rigid = solve_projection(draws, cfg)
    if p11 is None:
        fov = cfg.get("fov_y_deg", 60.0)
        p11 = 1 / math.tan(math.radians(fov) / 2)
        p00 = p11 / aspect
        print(f"[process] FOV not solvable ({n_rigid} rigid draws); using profile fov_y={fov}")
        fov_source = "profile"
    else:
        print(f"[process] solved FOV from {n_rigid} rigid draws: "
              f"fov_y={math.degrees(2 * math.atan(1 / p11)):.2f} deg, "
              f"aspect={p11 / p00:.3f} (backbuffer {aspect:.3f})")
        fov_source = "solved"
    fov_y = math.degrees(2 * math.atan(1 / p11))

    # Blender treats counter-clockwise faces as front; D3D games usually draw
    # clockwise fronts. Flip if the visible triangles are mostly clockwise.
    flip = cfg.get("flip_winding", "auto")
    if flip == "auto":
        flip = screen_winding(draws) < 0
    print(f"[process] winding: {'flipping clockwise -> counter-clockwise' if flip else 'kept'}")

    # Reconstruct camera-local Blender coordinates for every kept draw.
    seen = {}
    built = []
    for dd in draws:
        c = dd.clip
        view = np.stack([c[:, 0] / p00, c[:, 1] / p11, c[:, 3]], 1)
        pos = view * np.array([1.0, 1.0, -1.0])  # D3D view -> Blender camera-local
        tris = dd.tris
        if len(tris) == 0:
            reasons["no_triangles"] += 1
            continue
        if flip:
            tris = tris[:, [0, 2, 1]]
        key = hashlib.sha1(np.round(pos[tris[: 2000]], 3).tobytes()).hexdigest()
        tex_names = [t[2] for t in dd.d.textures]
        if key in seen:  # same surface drawn again (depth prepass, gbuffer, forward...)
            prev = seen[key]
            if len(tex_names) <= len(built[prev]["tex"]):
                reasons["duplicate"] += 1
                continue
            built[prev] = None
        seen[key] = len(built)
        # Category: "surface" (real geometry) or "effect" (blended layers, smoke,
        # decals, fog cards) which go to a hidden collection in Blender.
        cat = cfg.get("layout_categories", {}).get(layout_key(dd.pre), "surface")
        if layout_key(dd.pre) in cfg.get("sprite_layouts", []):
            cat = "sprite"  # camera-facing particle quads (smoke, steam, fire, specks)
        if cat == "surface" and tex_names and not any((dd.d.file.parent / t).exists() for t in tex_names):
            cat = "effect"  # every texture failed to save: dynamic render targets (smoke, heat, water)
        if cat == "surface" and len(tris) <= 4:
            c = dd.clip[dd.clip[:, 3] > 1e-6]
            ndc = c[:, :2] / c[:, 3:4] if len(c) else np.zeros((1, 2))
            span = np.clip(ndc.max(0), -1, 1) - np.clip(ndc.min(0), -1, 1)
            if span[0] * span[1] / 4 > 0.3:  # a single quad over a third of the screen
                cat = "effect"
        built.append({"dd": dd, "pos": pos, "tris": tris, "tex": tex_names, "cat": cat})
    built = [b for b in built if b]

    up = np.array(cfg["up_override"], float) if cfg.get("up_override") else \
        estimate_up([b["pos"][b["tris"]] for b in built if b["cat"] == "surface"], cfg.get("min_up_dot", 0.25))
    R = rotation_to_z(up)  # world = R @ camlocal
    tilt = math.degrees(math.acos(np.clip(up @ np.array([0, 1, 0]), -1, 1)))
    print(f"[process] ground plane found; camera pitched {tilt:.1f} deg from level")

    lights, seen_l = [], set()
    for dd in light_volumes:
        c = dd.clip
        v = np.stack([c[:, 0] / p00, c[:, 1] / p11, -c[:, 3]], 1) @ R.T  # -> world
        lo, hi = v.min(0), v.max(0)
        ext = hi - lo
        radius = float(ext.max() / 2)
        # Hemispheres (spot/area-ish lights): the light sits on the flat side.
        kind = "point" if ext.min() > 0.8 * ext.max() else "hemi"
        center = (lo + hi) / 2
        key = tuple(np.round(center / max(radius * 0.05, 1e-3)).astype(int))
        if key in seen_l:
            continue
        seen_l.add(key)
        lights.append({"location": center.round(4).tolist(), "radius": round(radius, 4), "kind": kind})
    if lights:  # a volume far larger than the rest is a global pass (ambient/fog), not a lamp
        cap_r = 10 * np.percentile([l["radius"] for l in lights], 90)
        lights = [l for l in lights if l["radius"] <= cap_r]
    print(f"[process] recovered {len(lights)} game lights from light volumes "
          f"({sum(l['kind'] == 'point' for l in lights)} point, {sum(l['kind'] == 'hemi' for l in lights)} hemisphere)")

    depths = np.concatenate([b["pos"][:, 2] for b in built]) * -1
    depths = depths[depths > 0]
    near, far = float(np.percentile(depths, 0.1)), float(np.percentile(depths, 99.9))

    # Textures
    tex_dir = capture / "textures"
    tex_dir.mkdir(exist_ok=True)
    tex_entries, tex_ids, jobs = {}, {}, []
    for b in built:
        for name in b["tex"]:
            src = (b["dd"].d.file.parent / name)
            if name in tex_ids or not src.exists():
                continue
            tid = f"t{len(tex_ids):04d}"
            tex_ids[name] = tid
            jobs.append((tid, src))
    # Decoding BC7 and writing PNG is CPU-bound; spread it over the cores.
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = max(1, min(len(jobs), (os.cpu_count() or 4) - 2, 24))
    print(f"[process] converting {len(jobs)} textures on {workers} cores ...", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(textures.convert, src, tex_dir): tid for tid, src in jobs}
        for i, fut in enumerate(futures, 1):
            entry = fut.result()
            if entry:
                tex_entries[futures[fut]] = entry
            if i % 25 == 0 or i == len(jobs):
                print(f"[process]   {i}/{len(jobs)} textures", flush=True)
    # A texture bound by many different shader types (snow/noise/LUT atlases) is
    # engine data, not a material texture; mark it so no slot rule can use it.
    users = {}
    for b in built:
        if b["cat"] != "surface":
            continue
        for name in b["tex"]:
            users.setdefault(name, set()).add(tuple(b["dd"].d.shaders))
    n_types = len({tuple(b["dd"].d.shaders) for b in built if b["cat"] == "surface"})
    shared = [n for n, u in users.items() if len(u) >= max(4, 0.25 * n_types)]
    for name in shared:
        if tex_ids.get(name) in tex_entries:
            tex_entries[tex_ids[name]]["role"] = "shared"
    if shared:
        print(f"[process] {len(shared)} shared engine textures ignored (bound by many shader types)")
    known = textures.load_known(cfg)
    textures.apply_known(tex_entries, known)
    fixed = textures.slot_consensus(
        [(b["dd"].d.shaders[1], {str(s): tex_ids[n] for s, n in enumerate(b["tex"]) if n in tex_ids})
         for b in built if b["cat"] == "surface" and len(b["dd"].d.shaders) >= 2], tex_entries)
    for tid, (old, new) in fixed.items():
        print(f"[process] texture {tid}: {old} -> {new} (what its shader slots read)")
    layers = textures.detail_layers(
        [(b["dd"].d.shaders[1], {str(s): tex_ids[n] for s, n in enumerate(b["tex"]) if n in tex_ids})
         for b in built if b["cat"] == "surface" and len(b["dd"].d.shaders) >= 2], tex_entries)
    for tid, (old, new) in layers.items():
        print(f"[process] texture {tid}: {old} -> {new} (a layer over several other albedos, not a base colour)")
    textures.apply_known(tex_entries, known)
    if known:
        n = sum(1 for e in tex_entries.values() if e.get("known"))
        print(f"[process] {n} of {len(tex_entries)} textures match the known-texture database "
              f"({cfg['known_textures']}): they get its checked roles. The other {len(tex_entries) - n} use the "
              f"rules; check them with `python gtb.py audit <capture>`")
    nr_shot = pick_reference_screenshot(capture, rip, (width, height), set(tex_ids))
    roles = Counter(e.get("role", "error") for e in tex_entries.values())
    print(f"[process] {len(tex_entries)} textures: {dict(roles)}")

    # Sprite draws bind a shared lookup/gradient texture next to their flipbook;
    # a texture that appears in many different sprite texture sets is that LUT.
    sets = {}
    for b in built:
        if b["cat"] == "sprite":
            for name in b["tex"]:
                sets.setdefault(name, set()).add(tuple(sorted(b["tex"])))
    lut_names = {n for n, u in sets.items() if len(u) >= 4}

    # Deferred decal volumes (boxes that only project a texture) are hidden with the effects.
    decals = 0
    for b in built:
        if b["cat"] == "surface" and decal_volume(b["pos"], b["tris"],
                                                  {(tex_entries.get(tex_ids.get(n)) or {}).get("role") for n in b["tex"]}):
            b["cat"] = "effect"
            decals += 1
    if decals:
        print(f"[process] {decals} decal volume(s) hidden (boxes that project a texture)")
    overlays = 0
    for b in built:
        if b["cat"] == "surface" and overlay_layer(assign_slots(
                {str(s): tex_ids[n] for s, n in enumerate(b["tex"]) if n in tex_ids}, tex_entries, cfg)):
            b["cat"] = "effect"
            overlays += 1
    if overlays:
        print(f"[process] {overlays} overlay layer(s) hidden (only a mask texture, blended over the ground)")

    # Geometry from another render pass lands far below the scene: hide it with the effects.
    surf = [b for b in built if b["cat"] == "surface"]
    tops = [float((b["pos"] @ R.T)[:, 2].max()) for b in surf]
    below = below_scene(tops)
    for k in below:
        surf[k]["cat"] = "effect"
    if below:
        print(f"[process] {len(below)} surface draws far below the scene hidden (another render pass)")
    floor = scene_floor(tops)
    off = [l for l in lights if floor is not None and l["location"][2] < floor - 100.0]
    for l in off:
        l["off"] = True     # kept (light names are numbered by position), but not lit
    if off:
        print(f"[process] {len(off)} game light(s) far below the scene switched off")

    # Meshes
    mesh_dir = capture / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    meshes = []
    winding_flips = 0
    uv_mode = cfg.get("uv_decode", "auto")
    for i, b in enumerate(built):
        dd, d = b["dd"], b["dd"].d
        pos = b["pos"] @ R.T
        out = {"positions": pos.astype(np.float32), "indices": b["tris"].astype(np.int32)}
        # Per-vertex attributes come from the pre-VS stream rows paired with each vertex.
        src = dd.pre if dd.rows is not None else None
        rows = dd.rows

        def attr(sem, index=None):
            a = src.find(sem, index) if src is not None else None
            return a

        a_n = attr("NORMAL")
        if a_n is not None and dd.lin is not None:
            nrm = decode_vector(src, a_n)
            if nrm is not None:
                A = np.diag([1 / p00, 1 / p11, 1.0]) @ dd.lin      # object -> D3D view
                A = np.diag([1.0, 1.0, -1.0]) @ A                   # -> Blender camera-local
                nv = nrm @ np.linalg.inv(A)                          # inverse-transpose
                nv = nv @ R.T
                out["normals"] = nv[rows].astype(np.float32)
        for k, name in ((0, "uv0"), (1, "uv1")):
            a_uv = attr("TEXCOORD", k)
            if a_uv is None:
                continue
            uv = decode_uv(src, a_uv, uv_mode)
            if uv is None:
                continue
            uv = uv.copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # D3D top-left origin -> Blender bottom-left
            out[name] = uv[rows].astype(np.float32)
        a_c = attr("COLOR")
        if a_c is not None and a_c.va_type in (0, 3) and a_c.comps >= 3:
            col = src.read(a_c).astype(np.float64)
            col = col / 255.0 if a_c.va_type == 3 else col
            if col.shape[1] == 3:
                col = np.hstack([col, np.ones((len(col), 1))])
            out["colors"] = col[rows].astype(np.float32)
        # Skinned draws keep their bind pose and bone weights: gtb/characters.py
        # solves each person's bones from them (Unreal walkers).
        out.update(characters.skin_arrays(src, rows))

        sprite = None
        if b["cat"] == "sprite":
            sprite = sprite_info(b, tex_ids, tex_entries, lut_names)
            if sprite is None:
                b["cat"] = "effect"  # snow specks etc.: the procedural snowfall replaces them
            else:
                out.update(sprite_attributes(out["positions"], out["indices"], R))

        # Per-mesh winding: the game's vertex normals are reliable, so flip the
        # triangle order wherever the faces point against them (shaders in one
        # game can use different front-face conventions).
        if "normals" in out and len(out["indices"]):
            P, T, Nv = out["positions"], out["indices"], out["normals"]
            fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
            agree = np.einsum("ij,ij->i", fn, Nv[T].sum(1))
            if (agree < 0).sum() > (agree > 0).sum():
                out["indices"] = T[:, [0, 2, 1]]
                winding_flips += 1

        fname = f"meshes/d{i:05d}.npz"
        np.savez_compressed(capture / fname, **out)
        rs = d.render_state
        meshes.append({
            "name": f"{d.file.stem}_{d.draw_call}",
            "file": fname,
            "source": d.file.name,
            "textures": {str(s): tex_ids[t] for s, t in enumerate(b["tex"]) if t in tex_ids},
            "shaders": d.shaders,
            "depth_write": bool(rs.get("DEPTH_WRITE_ENABLE", 1)),
            "category": b["cat"],
            "sprite": sprite,
            "instances": d.instances,
            "rigid_fit": dd.fit_err,
            "layout_pre": [f"{a.name}:{a.va_type}x{a.comps}" for a in (dd.pre.attrs if dd.pre else [])],
            "layout_post": [f"{a.name}:{a.va_type}x{a.comps}" for a in (dd.post.attrs if dd.post else [])],
        })

    snow_box = find_snow_box(capture, meshes, built, cfg)
    print(f"[process] snow volume {np.round(snow_box[0], 1).tolist()} .. {np.round(snow_box[1], 1).tolist()}")
    smoke = find_smoke_plumes(capture, meshes, built, cfg)
    if smoke:
        print(f"[process] {len(smoke)} smoke plume(s) from smoke columns: {[pl['source'] for pl in smoke]}")

    cam = np.eye(4)
    cam[:3, :3] = R  # camera-local -> world rotation; camera sits at the origin
    manifest = {
        "game": exe,
        "profile": cfg,
        "resolution": [width, height],
        "screenshot": "screenshot.png",
        "camera": {"fov_y_deg": fov_y, "fov_source": fov_source, "aspect_solved": p11 / p00,
                   "near": near * 0.5, "far": far * 1.5, "matrix_world": cam.tolist()},
        "textures": tex_entries,
        "meshes": meshes,
        "lights": lights,
        "smoke": smoke,
        "snow_box": snow_box,
        "stats": {"skipped": dict(reasons), "kept": len(meshes)},
        "reference_screenshot": nr_shot or "daemon",
    }
    (capture / "manifest.json").write_text(json.dumps(manifest, indent=1))

    look = capture / "look.json"
    if not look.exists():
        look.write_text(json.dumps(cfg.get("look", {}), indent=2))
    print(f"[process] per-mesh winding: flipped {winding_flips} meshes to match the game's normals")
    cats = Counter(mm["category"] for mm in meshes)
    print(f"[process] wrote manifest with {len(meshes)} meshes {dict(cats)}")
    return manifest
