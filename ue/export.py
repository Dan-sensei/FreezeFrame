"""Manifest -> Unreal build plan (runs in normal Python, no Unreal needed).

Writes captures/<name>/unreal/:
  plan.json      meshes, materials, textures, actors, cameras and game lights in
                 Unreal space (left-handed, Z up, centimetres)
  meshes/*.glb   geometry only; every mesh re-centred on its bounding box
  SM_Snowfall.glb the procedural snowfall's flake quads

Material decisions use the same rules as the Blender builder
(gtb.scene_common.assign_slots and blender/gtb_scene.build), so both engines
agree on which texture is albedo, normal, roughness, drift snow or terrain.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtb.process import sprite_attributes  # noqa: E402
from gtb.scene_common import (assign_slots, below_scene, decal_volume, overlay_layer, packed_channels,  # noqa: E402
                               scene_floor, srgb_to_linear)
from ue.gltf import GlbWriter, blender_to_gltf  # noqa: E402

CM = 100.0               # Blender metres -> Unreal centimetres
CHUNK_TRIANGLES = 400_000
SNOW_MAX_FLAKES = 100_000


def to_ue(p):
    """Blender (x, y, z) metres -> Unreal (x, -y, z) centimetres."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 1], p[..., 2]], -1) * CM


def dir_to_ue(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 1], v[..., 2]], -1)


def r3(v, nd=4):
    return [round(float(x), nd) for x in v]


# --------------------------------------------------------------------------- geometry

def smooth_normals(pos, tris):
    """Corner-angle weighted vertex normals, like Blender's smooth shading."""
    p = pos.astype(np.float64)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    nrm = np.zeros_like(p)
    for i, (u, v, w) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
        e1, e2 = v - u, w - u
        cosang = np.einsum("ij,ij->i", e1, e2) / np.maximum(
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-20)
        np.add.at(nrm, tris[:, i], fn * np.arccos(np.clip(cosang, -1, 1))[:, None])
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    return np.where(n > 1e-20, nrm / np.maximum(n, 1e-20), [0.0, 0.0, 1.0])


def sprite_frames(data):
    """Per-vertex texture axes (dCorner/du, dCorner/dv) in the camera plane,
    so the re-faced sprite can light its flipbook normal map like Blender's
    UV tangents do. Blender UV convention (v up)."""
    corner, uv, tris = data["attr_p_corner"].astype(np.float64), data["uv0"].astype(np.float64), data["indices"]
    t_acc = np.zeros((len(corner), 2))
    b_acc = np.zeros((len(corner), 2))
    for t in tris:
        c0, c1, c2 = corner[t]
        u0, u1, u2 = uv[t]
        dc = np.stack([c1 - c0, c2 - c0], 1)        # 2x2, columns = edges
        du = np.stack([u1 - u0, u2 - u0], 1)
        if abs(np.linalg.det(du)) < 1e-12:
            continue
        J = dc @ np.linalg.inv(du)                   # d(corner)/d(uv)
        t_acc[t] += J[:, 0]
        b_acc[t] += J[:, 1]
    norm = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return norm(t_acc), norm(b_acc)


class ChunkedGlb:
    """Splits meshes over several .glb files so one bad file can't sink the import."""

    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir, self.prefix = out_dir, prefix
        self.files, self.w, self.tris = {}, GlbWriter(), 0
        self.names = []

    def add(self, name, tri_count, **kw):
        if self.tris + tri_count > CHUNK_TRIANGLES and len(self.w):
            self.flush()
        self.w.add_mesh(name, **kw)
        self.names.append(name)
        self.tris += tri_count
        return f"{self.prefix}{len(self.files):02d}.glb"

    def flush(self):
        if len(self.w):
            path = self.out_dir / f"{self.prefix}{len(self.files):02d}.glb"
            self.w.save(path)
            self.files[path.name] = self.names
        self.w, self.tris, self.names = GlbWriter(), 0, []


# --------------------------------------------------------------------------- materials

def texture_spec(entry, kind):
    return {"file": entry["file"], "kind": kind}


CHANNEL_MASK = {"R": [1, 0, 0, 0], "G": [0, 1, 0, 0], "B": [0, 0, 1, 0], "A": [0, 0, 0, 1]}
MASK_NAMES = {"Roughness": "Rough", "Metallic": "Metal"}


def _channel_targets(mapping, prefix, sc, vec):
    """Profile channel maps, e.g. {"R": "Roughness", "G": "1-Metallic"} -> mask parameters."""
    for ch, target in mapping.items():
        invert = target.startswith("1-")
        target = target[2:] if invert else target
        if target in MASK_NAMES and ch in CHANNEL_MASK:
            vec[f"{prefix}{MASK_NAMES[target]}Mask"] = CHANNEL_MASK[ch]
            sc[f"{prefix}{MASK_NAMES[target]}Invert"] = 1.0 if invert else 0.0


def cloth_rules(profile):
    """profile["cloth"]; captures processed before the rule existed read it from
    the current profile file."""
    if "cloth" in profile:
        return profile["cloth"]
    f = ROOT / "profiles" / f"{profile.get('_name', 'default')}.json"
    return json.loads(f.read_text(encoding="utf-8")).get("cloth") if f.exists() else None


def is_cloth(m, pos_b, rules):
    """Hanging cloth (see profiles/*.json "cloth"): pinned at the top, free below."""
    if not rules or m.get("shaders") not in rules.get("shaders", []):
        return False
    if (m.get("rigid_fit") or 0.0) <= rules.get("min_rigid_fit", 1e-3):
        return False
    span = np.ptp(pos_b, axis=0)
    return bool(span[2] >= rules.get("min_aspect", 2.0) * max(span[0], span[1]))


def cloth_weight(z, held):
    """Motion weight of gtb_hlsl.CLOTH_WPO over a banner's height (keep in sync):
    d^2 from 0 at the pinned top (the upper part stays calm) to 1 at a free
    bottom; a held bottom fades it out over the lowest 35%."""
    d = np.clip((z.max() - z) / max(np.ptp(z), 1e-6), 0.0, 1.0)
    return d * d * (np.clip((1.0 - d) / 0.35, 0.0, 1.0) if held else 1.0)


def cloth_taper(x, half):
    """Across-plane weight when the side edges are held (gtb_hlsl.CLOTH_WPO):
    1 over the middle, 0 at the edges; x = offset from the centre line."""
    return np.clip((1.0 - np.abs(x) / max(half, 1e-6)) / 0.4, 0.0, 1.0)


def cloth_room(pos_b, tris, soup, reach=3.0, margin=0.1, tolerance=0.03):
    """How far a banner may move before it meets other geometry (walls, beams, its
    frame). Returns (front xy, budgets, bottom held, sides held), in Blender metres.
    budgets[k] is the room per unit motion weight in direction k * 45 degrees
    from front towards across = (-front.y, front.x), covering its whole sector:
    the shader may move a vertex up to budget * weight that way. Held: the bottom
    edge / side edges have geometry close on both sides (inside a beam, or between
    a crossbar and the wall). A held bottom fades the motion out towards it;
    held sides stop the sideways swing and fade the
    motion across the plane to 0 at the edges (cloth_taper). A vertex may go
    `tolerance` into geometry it already touches (near the mounting bracket), or
    one such vertex with a tiny weight would freeze the whole banner."""
    n = smooth_normals(pos_b, tris)[:, :2].mean(0)
    n = n / (np.linalg.norm(n) + 1e-12)
    ac = np.array([-n[1], n[0]])
    near = soup.near(pos_b.min(0) - reach, pos_b.max(0) + reach)
    z0, z1 = pos_b[:, 2].min(), pos_b[:, 2].max()
    lo, hi = pos_b.min(0), pos_b.max(0)
    # Across offset from the bounds centre, as the shader computes it.
    x = (pos_b[:, :2] - (lo[:2] + hi[:2]) / 2) @ ac
    half = 0.5 * float(np.abs(ac) @ (hi[:2] - lo[:2]))
    # 32 horizontal directions, angle 0 = front, towards across.
    ang = np.arange(32) * np.pi / 16
    dirs = np.outer(np.cos(ang), n) + np.outer(np.sin(ang), ac)
    front_back = dirs[[0, 16]]
    middle = np.abs(x) <= 0.6 * half          # the two inner columns (at +-1/3 of the width)
    edges = np.abs(x) >= 0.9 * half
    bottom = pos_b[(pos_b[:, 2] <= z0 + 0.1 * (z1 - z0)) & middle]
    side = pos_b[edges & (pos_b[:, 2] <= z1 - 0.25 * (z1 - z0)) & (pos_b[:, 2] >= z0 + 0.1 * (z1 - z0))]
    # Held = geometry within 15 cm on both sides (inside a beam, or between a
    # crossbar and the wall); a wall on one side only leaves it free.
    held = bool(len(bottom)) and cast_level(near, bottom, front_back).min(0).max() < 0.15
    sides = bool(len(side)) and cast_level(near, side, front_back).min(0).max() < 0.15
    w = cloth_weight(pos_b[:, 2], held)
    if sides:
        w = w * cloth_taper(x, half)       # the motion is along +-front only
    mv = w > 0.02
    free = (np.maximum(np.minimum(cast_level(near, pos_b[mv], dirs, corners=True), reach) - margin, 0.0)
            + tolerance) / w[mv, None]
    budgets = []
    for k in range(8):
        # The whole sector to both neighbouring directions (the shader
        # interpolates), so obstacles between sample directions still count.
        sector = [(4 * k + j) % 32 for j in range(-4, 5)]
        b = free[:, sector].min() if mv.any() else reach
        budgets.append(float(min(b, reach / 0.02)))
    return n, budgets, held, sides


def surface_material(slots, profile, has_uv):
    """Unreal version of blender/gtb_scene.build_material, as parameter values
    for the M_GTB_Surface master (ue/editor/gtb_hlsl.SURFACE)."""
    mat = {"parent": "surface", "textures": {}, "scalars": {}, "vectors": {}}
    mean = None
    if not has_uv:
        albedo = slots.get("albedo") or next(iter(slots.values()), None)
        slots = {}
        mean = (albedo or {}).get("details", {}).get("mean_rgb")
    tex, sc, vec = mat["textures"], mat["scalars"], mat["vectors"]
    ground = False
    albedo = slots.get("albedo")
    if albedo:
        tex["Albedo"] = texture_spec(albedo, "albedo")
        sc["UseAlbedo"] = 1.0
        # Games put a cut-out mask in albedo alpha: alpha-tested, not blended.
        if albedo.get("details", {}).get("has_alpha") and not profile.get("ignore_albedo_alpha"):
            mat["parent"] = "surface_masked"
        if profile.get("multiply_vertex_color"):
            sc["MultiplyVertexColor"] = 1.0
    elif not has_uv and (profile.get("no_uv_base_color") or mean):
        ground = True
        vec["BaseColor"] = srgb_to_linear(profile.get("no_uv_base_color") or mean) + [1.0]
    elif not slots:
        vec["BaseColor"] = srgb_to_linear(profile.get("untextured_base_color", [0.3, 0.3, 0.32])) + [1.0]
    else:
        ground = True
        vec["BaseColor"] = srgb_to_linear(profile.get("default_base_color", [0.5, 0.5, 0.5])) + [1.0]
    sc["IsGround"] = 1.0 if ground else 0.0

    normal = slots.get("normal")
    if normal:
        tex["Normal"] = texture_spec(normal, "linear")
        sc["UseNormal"] = 1.0
        channels = normal.get("details", {}).get("channels", "RGB")
        sc["NormalAG"] = 1.0 if channels == "AG" else 0.0
        sc["NormalRebuildZ"] = 0.0 if channels == "RGB" else 1.0
        # Blender flips green for DirectX-style maps; Unreal expects that style already.
        sc["NormalFlipGreen"] = 0.0 if profile.get("normal_flip_green", True) else 1.0
        if channels == "AG":
            _channel_targets(packed_channels({k: v for k, v in profile.get("normal_packed_channels", {}).items()
                                              if k in "RB"}, normal), "Normal", sc, vec)
    gray = slots.get("gray")
    if gray:
        tex["Gray"] = texture_spec(gray, "linear")
        sc["UseGray"] = 1.0
        sc["GrayIsGloss"] = 1.0 if profile.get("gray_is_gloss") else 0.0
    packed = slots.get("packed")
    if packed and profile.get("packed_channels"):
        tex["Packed"] = texture_spec(packed, "linear")
        sc["UsePacked"] = 1.0
        _channel_targets(packed_channels(profile["packed_channels"], packed), "Packed", sc, vec)
    emissive = slots.get("emissive")
    if emissive:
        tex["Emissive"] = texture_spec(emissive, "albedo")
        sc["UseEmissive"] = 1.0
    sc["UseSnow"] = 0.0 if ground else 1.0   # the untextured ground already is snow
    return mat


# --------------------------------------------------------------------------- closeups

def ray_hit(origin, direction, meshes):
    """Nearest hit of a ray against (positions, tris) lists (Möller-Trumbore)."""
    best = np.inf
    for pos, tris in meshes:
        a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
        e1, e2 = b - a, c - a
        pv = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = origin - a
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = (qv @ direction) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        if hit.any():
            best = min(best, float(t[hit].min()))
    return best


class TriSoup:
    """All triangles of a mesh list in flat arrays, for fast batched ray casts."""
    def __init__(self, meshes):
        tri = [pos[t] for pos, t in meshes if len(t)] or [np.zeros((0, 3, 3))]
        self.tri = np.concatenate(tri)                     # (T, 3 corners, xyz)
        self.lo, self.hi = self.tri.min(1), self.tri.max(1)

    def near(self, lo, hi):
        """Triangles whose bounds overlap the box lo..hi."""
        keep = (self.hi >= lo).all(1) & (self.lo <= hi).all(1)
        sub = TriSoup.__new__(TriSoup)
        sub.tri, sub.lo, sub.hi = self.tri[keep], self.lo[keep], self.hi[keep]
        return sub

    def cast(self, origins, dirs, chunk=4_000_000):
        """Nearest hit distance of each ray (inf = none), Moller-Trumbore on
        rays x triangles in chunks."""
        origins, dirs = np.atleast_2d(origins), np.atleast_2d(dirs)
        dirs = np.broadcast_to(dirs, origins.shape)
        out = np.full(len(origins), np.inf)
        if not len(self.tri):
            return out
        a = self.tri[:, 0]
        e1, e2 = self.tri[:, 1] - a, self.tri[:, 2] - a
        step = max(1, chunk // len(a))
        for i in range(0, len(origins), step):
            o, d = origins[i:i + step, None, :], dirs[i:i + step, None, :]
            pv = np.cross(d, e2)
            det = (e1 * pv).sum(-1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = o - a
            u = (tv * pv).sum(-1) * inv
            qv = np.cross(tv, e1)
            v = (qv * d).sum(-1) * inv
            t = (e2 * qv).sum(-1) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
            out[i:i + step] = np.where(hit, t, np.inf).min(1)
        return out


    def slice(self, z):
        """Cross-section at height z: 2D segments (S, 2 ends, xy) where triangles cut it."""
        tri = self.tri[(self.lo[:, 2] <= z) & (self.hi[:, 2] >= z)]
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[:, i], tri[:, j]
            da, db = a[:, 2] - z, b[:, 2] - z
            cut = (da * db < 0) | ((da == 0) & (db != 0))
            f = np.where(cut, da / np.where(da != db, da - db, 1.0), 0.0)
            pts.append((np.where(cut[:, None], a[:, :2] + f[:, None] * (b[:, :2] - a[:, :2]), np.nan), cut))
        (p0, c0), (p1, c1), (p2, c2) = pts
        first = np.where(c0[:, None], p0, p1)
        second = np.where((c0 & c1)[:, None], p1, p2)
        ok = (c0.astype(int) + c1 + c2) >= 2
        return np.stack([first[ok], second[ok]], 1)


def cast2d(origins, dirs, segs):
    """Nearest hit distance of 2D rays against segments (inf = none)."""
    out = np.full(len(origins), np.inf)
    if not len(segs) or not len(origins):
        return out
    p, e = segs[:, 0], segs[:, 1] - segs[:, 0]
    step = max(1, 2_000_000 // len(p))
    for i in range(0, len(origins), step):
        o, d = origins[i:i + step, None, :], dirs[i:i + step, None, :]
        den = d[..., 0] * e[:, 1] - d[..., 1] * e[:, 0]
        ok = np.abs(den) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, den, 1.0), 0.0)
        po = p - o
        t = (po[..., 0] * e[:, 1] - po[..., 1] * e[:, 0]) * inv
        u = (po[..., 0] * d[..., 1] - po[..., 1] * d[..., 0]) * inv
        hit = ok & (t > 1e-4) & (u >= 0) & (u <= 1)
        out[i:i + step] = np.where(hit, t, np.inf).min(1)
    return out


def cast_level(soup, pts, dirs, corners=False):
    """Horizontal rays: for each start point, the nearest hit along each 2D
    direction (P x D), slicing the geometry once per distinct height.
    corners: dirs are evenly spaced around the circle; each direction's result
    becomes the nearest geometry anywhere within half a step of it, not just on
    the ray. The slice's segment ends within that angle count too, so thin
    things (poles, ropes) between two rays are not missed: mesh_1049's edge went
    6 cm into a pole that sat between its 11.25-degree rays."""
    out = np.full((len(pts), len(dirs)), np.inf)
    base = np.arctan2(dirs[0, 1], dirs[0, 0])
    turn = np.sign(dirs[0, 0] * dirs[1, 1] - dirs[0, 1] * dirs[1, 0]) if len(dirs) > 1 else 1.0
    for z in np.unique(np.round(pts[:, 2], 3)):
        rows = np.where(np.round(pts[:, 2], 3) == z)[0]
        segs = soup.slice(z)
        o = np.repeat(pts[rows, :2], len(dirs), 0)
        d = np.tile(dirs, (len(rows), 1))
        out[rows] = cast2d(o, d, segs).reshape(len(rows), len(dirs))
        if corners and len(segs):
            ends = segs.reshape(-1, 2)
            for r in rows:
                v = ends - pts[r, :2]
                dist = np.hypot(v[:, 0], v[:, 1])
                ang = (turn * (np.arctan2(v[:, 1], v[:, 0]) - base)) % (2 * np.pi)
                k = np.rint(ang / (2 * np.pi) * len(dirs)).astype(int) % len(dirs)
                np.minimum.at(out[r], k, dist)
    return out


def closeup_views(cam_m, surfaces):
    """Same three check views as blender/closeups.py: the game camera dollied to
    35% and 15% of the distance to what it looks at, and a low 20-degree view."""
    origin = cam_m[:3, 3]
    fwd = -cam_m[:3, 2] / np.linalg.norm(cam_m[:3, 2])
    up = cam_m[:3, 1]
    t = ray_hit(origin, fwd, surfaces)
    target = origin + fwd * (t if np.isfinite(t) else 100.0)
    views = [("Closeup35", target + (origin - target) * 0.35, fwd, up),
             ("Closeup15", target + (origin - target) * 0.15, fwd, up)]
    dist = np.linalg.norm(origin - target) * 0.15
    flat = np.array([fwd[0], fwd[1], 0.0])
    flat /= np.linalg.norm(flat)
    a = math.radians(20)
    low = target - flat * dist * math.cos(a) + np.array([0, 0, dist * math.sin(a)])
    lf = (target - low) / np.linalg.norm(target - low)
    lup = np.array([0.0, 0.0, 1.0]) - lf[2] * lf
    views.append(("CloseupLow", low, lf, lup / np.linalg.norm(lup)))
    return target, [{"name": n, "location": r3(to_ue(p), 2), "forward": r3(dir_to_ue(f), 6),
                     "up": r3(dir_to_ue(u), 6)} for n, p, f, u in views]


# --------------------------------------------------------------------------- snowfall

def write_snowfall(path: Path, count=SNOW_MAX_FLAKES, seed=1):
    """Flake quads for the snowfall material. Everything the material needs is
    per-flake random numbers in UV channels; the vertex positions are only a
    tiny placeholder (the material places every flake from time and camera).
      UV0 corner (0..1)   UV1 random start xy (0..1)   UV2 (random start z, size 0.5..1.5)
      UV3 (flake index / max, random phase 0..1)"""
    rng = np.random.default_rng(seed)
    start = rng.random((count, 3))
    size = 0.5 + rng.random(count)
    phase = rng.random(count)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    uv0 = np.tile(corners, (count, 1))
    rep = lambda a: np.repeat(a, 4, axis=0)
    uv1 = rep(start[:, :2])
    uv2 = rep(np.stack([start[:, 2], size], 1))
    uv3 = rep(np.stack([np.arange(count) / count, phase], 1))
    # Spread the placeholder quads out (100 m cube, 2 cm quads): coincident
    # vertices make Unreal's vertex welding quadratic.
    pos = rep(start * 100.0 - 50.0)
    pos[:, 0] += (uv0[:, 0] - 0.5) * 0.02
    pos[:, 2] += (uv0[:, 1] - 0.5) * 0.02
    base = np.arange(count)[:, None] * 4
    tris = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], 1).reshape(-1, 3)
    nrm = np.tile([0.0, -1.0, 0.0], (count * 4, 1))
    w = GlbWriter()
    w.add_mesh("SM_Snowfall", blender_to_gltf(pos), tris, blender_to_gltf(nrm), [uv0, uv1, uv2, uv3])
    w.save(path)


# Smoke sprite cards at least this wide (m) are mist, not puffs. Frostpunk: the ground
# mist's cards are 75-80 m, the haze on the crater walls (mesh_5266) 23 m, the wind-blown
# haze and wisps 21-27 m; the biggest chimney puffs are 12 m.
MIST_CARD_M = 16.0
# process.py drops particle draws with a nearly empty flipbook as snowflakes (the
# procedural snowfall replaces them). Frostpunk's snowflake sheets (t0124/t0127) have a
# mean alpha of 0.0001; the faint mist flipbooks it also dropped have 0.009-0.018.
SPECK_ALPHA = 0.001


def card_width(corner):
    """Median sprite card width (m), from the corner offsets in the camera plane."""
    return 2.0 * float(np.median(np.abs(corner).max(1)))


def faint_mist(m, data, textures, profile, cam_axes):
    """A particle effect process.py dropped as snowflakes that is really a mist layer:
    its flipbook is faint rather than empty, and its cards are mist-sized. Frostpunk:
    mesh_5273 (wind-blown haze over the ice) and mesh_5255/5268 (wisps over the east
    crater wall). Returns (sprite info, billboard attributes) or None."""
    layout = ",".join(a.split(":")[0] for a in m.get("layout_pre", []))
    if m.get("category") != "effect" or layout not in profile.get("sprite_layouts", []) or "uv0" not in data:
        return None
    cands = [t for t in (m.get("textures") or {}).values() if "alpha_mean" in (textures.get(t) or {}).get("details", {})]
    if not cands:
        return None
    atlas = max(cands, key=lambda t: textures[t]["details"]["alpha_std"])       # as process.sprite_info
    if textures[atlas]["details"]["alpha_mean"] < SPECK_ALPHA:
        return None
    attrs = sprite_attributes(data["positions"], data["indices"].reshape(-1, 3), cam_axes)
    if card_width(attrs["attr_p_corner"]) < MIST_CARD_M:
        return None
    return {"kind": "mist", "atlas": atlas}, attrs


def write_plume(path: Path, frames, count=320, seed=7):
    """Puff quads for a rising smoke column (M_GTB_Plume places them from time).
      UV0 flipbook frame (glTF/Unreal v)   UV1 (phase 0..1, spin angle)
      UV2 lateral jitter (-1..1)            UV3 corner (+-0.5, Blender v up)"""
    rng = np.random.default_rng(seed)
    phase = (np.arange(count) + rng.random(count) * 0.8) / count
    angle = rng.random(count) * 2 * math.pi
    jitter = rng.uniform(-1, 1, (count, 2))
    fr = np.array(frames, dtype=np.float64)[rng.integers(0, len(frames), count)]    # u0, v0, u1, v1 (v up)
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    rep = lambda a: np.repeat(a, 4, axis=0)
    c = np.tile(corners, (count, 1))
    f = rep(fr)
    u = f[:, 0] + (c[:, 0] + 0.5) * (f[:, 2] - f[:, 0])
    v = f[:, 1] + (c[:, 1] + 0.5) * (f[:, 3] - f[:, 1])
    uv0 = np.stack([u, 1.0 - v], 1)
    pos = rep(rng.random((count, 3)) * 100.0 - 50.0)          # spread placeholders (see write_snowfall)
    pos[:, 0] += c[:, 0] * 0.02
    pos[:, 2] += c[:, 1] * 0.02
    base = np.arange(count)[:, None] * 4
    tris = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], 1).reshape(-1, 3)
    colors = np.ones((count * 4, 4))
    colors[:, 3] = rep(rng.uniform(0.6, 1.0, count))
    w = GlbWriter()
    w.add_mesh("SM_Plume", blender_to_gltf(pos), tris, blender_to_gltf(np.tile([0.0, -1.0, 0.0], (count * 4, 1))),
               [uv0, rep(np.stack([phase, angle], 1)), rep(jitter), c], colors)
    w.save(path)


# --------------------------------------------------------------------------- main

def export(capture: Path, solo=False):
    """solo: also write every sprite to its own .glb (meshes/solo/<asset>.glb, listed
    in plan["solo_glbs"]), so `ue.live sprites` can re-import one without its chunk."""
    capture = Path(capture)
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    profile = manifest.get("profile", {})
    cloth = cloth_rules(profile)
    textures = manifest["textures"]
    out = capture / "unreal"
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in list(mesh_dir.glob("*.glb")) + list(mesh_dir.glob("solo/*.glb")):
        old.unlink()
    if solo:
        (mesh_dir / "solo").mkdir(exist_ok=True)
    solo_glbs = {}
    cam_m = np.array(manifest["camera"]["matrix_world"], dtype=np.float64)
    cam_axes = cam_m[:3, :3] / np.linalg.norm(cam_m[:3, :3], axis=0)     # game camera right/up/back, world

    chunks = ChunkedGlb(mesh_dir, "geo")
    materials, used_textures, actors = {}, {}, []
    surfaces_for_ray = []
    solids, cloth_actors = [], []       # cloth: how far each banner may move (see cloth_room)
    sprite_frames_by_atlas = {}
    counts = {"surface": 0, "cloth": 0, "effect": 0, "sprite": 0, "snowdrift": 0}

    def use_tex(spec):
        prev = used_textures.get(spec["file"])
        # A file used both as colour and data keeps the colour (sRGB) import.
        if not prev or (prev["kind"] != "albedo" and spec["kind"] == "albedo"):
            used_textures[spec["file"]] = spec

    # Geometry another render pass drew lands far below the scene (process.py hides it
    # from new captures; this covers manifests written before that rule).
    # Decal volumes (boxes that only project a texture) are hidden too; see scene_common.
    surf, tops, decals = [], [], set()
    for m in manifest["meshes"]:
        if m.get("depth_write", True) and m.get("category", "surface") == "surface":
            d = np.load(capture / m["file"])
            p = d["positions"]
            if len(p):
                surf.append(m["name"])
                tops.append(float(p[:, 2].max()))
                roles = {(textures.get(t) or {}).get("role") for t in (m.get("textures") or {}).values()}
                if decal_volume(p, d["indices"].reshape(-1, 3), roles) or                         overlay_layer(assign_slots(m.get("textures", {}), textures, profile)):
                    decals.add(m["name"])
    below = {surf[k] for k in below_scene(tops)} | decals

    for i, m in enumerate(manifest["meshes"]):
        data = dict(np.load(capture / m["file"]))
        tris = data["indices"].astype(np.int64).reshape(-1, 3)
        if len(tris) == 0:
            continue
        pos_b = data["positions"].astype(np.float64)
        lo, hi = pos_b.min(0), pos_b.max(0)
        centre_b = (lo + hi) / 2
        local_b = pos_b - centre_b
        name = f"SM_{i:05d}_{m['name']}"
        kw = {"positions": blender_to_gltf(local_b), "indices": tris}
        uv_list = []
        if "uv0" in data:
            uv0 = data["uv0"].astype(np.float64).copy()
            uv0[:, 1] = 1.0 - uv0[:, 1]            # Blender bottom-left -> glTF/Unreal top-left
            uv_list.append(uv0)
        actor = {"name": m["name"], "mesh": name, "location": r3(to_ue(centre_b), 3),
                 "source": m.get("source", ""), "hidden": False, "cast_shadow": True}

        promoted = faint_mist(m, data, textures, profile, cam_axes)
        if promoted:
            data.update(promoted[1])
        if (m.get("category") == "sprite" and m.get("sprite")) or promoted:
            sp = promoted[0] if promoted else m["sprite"]
            # The game's mist and haze are smoke sprites on big cards (Frostpunk: the ground
            # mist's 322 soft blobs, 75-80 m wide, and the haze layers; see MIST_CARD_M). They
            # get their own look (MI_Look_Mist): more opacity, a long soft fade into geometry
            # (no hard lines through buildings), a near fade.
            kind = sp["kind"]
            if kind == "smoke" and "attr_p_corner" in data and card_width(data["attr_p_corner"]) >= MIST_CARD_M:
                kind = "mist"
            key = f"sprite|{kind}|{sp['atlas']}"
            if key not in materials:
                entry = textures[sp["atlas"]]
                spec = texture_spec(entry, "albedo" if kind == "fire" else "linear")
                use_tex(spec)
                materials[key] = {"name": f"MI_Sprite_{kind}_{sp['atlas']}", "parent": kind,
                                  "textures": {"Atlas": spec}, "scalars": {}, "vectors": {}}
            # Billboard data (Unreal world cm): offset to the puff centre, corner
            # offsets along the game camera's right/up, texture axes.
            d = to_ue(data["attr_p_center"].astype(np.float64)) - to_ue(pos_b)
            corner = data["attr_p_corner"].astype(np.float64) * CM
            tan, bit = sprite_frames(data)
            # Per-puff random phase: the editor preview loops the puffs, out of step.
            puff_id = np.unique(np.round(data["attr_p_center"], 3), axis=0, return_inverse=True)[1].ravel()
            phase = np.random.default_rng(i).random(puff_id.max() + 1)[puff_id]
            uv_list += [d[:, :2], np.stack([d[:, 2], np.zeros(len(d))], 1), corner, tan, bit,
                        np.stack([phase, np.zeros(len(d))], 1)]
            if kind == "smoke":
                for k in np.unique(puff_id):
                    uv = data["uv0"][puff_id == k]
                    sprite_frames_by_atlas.setdefault(sp["atlas"], set()).add(
                        tuple(np.round([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], 4)))
            kw["colors"] = data["colors"] if "colors" in data else np.ones((len(pos_b), 4))
            kw["normals"] = blender_to_gltf(smooth_normals(pos_b, tris))
            actor.update(cast_shadow=False, folder="Particles")
            counts["sprite"] += 1
            if not promoted:                         # Blender's close-up ray hits its sprites too
                surfaces_for_ray.append((pos_b, tris))
        else:
            tex_entries = [textures[t] for t in m.get("textures", {}).values() if t in textures]
            kw["normals"] = blender_to_gltf(data["normals"] if "normals" in data else smooth_normals(pos_b, tris))
            if tex_entries and all(e.get("role") == "shared" for e in tex_entries) and "uv0" in data:
                key = "snowmesh|" + tex_entries[0]["file"]
                if key not in materials:
                    spec = texture_spec(tex_entries[0], "linear")
                    use_tex(spec)
                    materials[key] = {"name": "MI_SnowDrift", "parent": "snowdrift", "textures": {"SnowTex": spec},
                                      "scalars": {
                                          "NormalFlipGreen": 0.0 if profile.get("normal_flip_green", True) else 1.0},
                                      "vectors": {}}
                actor["folder"] = "Geometry"
                counts["snowdrift"] += 1
                surfaces_for_ray.append((pos_b, tris))
                solids.append((pos_b, tris))
            else:
                slots = assign_slots(m.get("textures", {}), textures, profile)
                has_uv = "uv0" in data
                key = m.get("material_key") or json.dumps(
                    {r: e["file"] for r, e in sorted(slots.items())}, sort_keys=True) + ("" if has_uv else "|nouv")
                flutter = is_cloth(m, pos_b, cloth)
                key += "|cloth" if flutter else ""
                if key not in materials:
                    mat = surface_material(slots, profile, has_uv)
                    if flutter:
                        mat["parent"] = "cloth"
                    mat["name"] = f"MI_M{len(materials):03d}"
                    for spec in mat["textures"].values():
                        use_tex(spec)
                    materials[key] = mat
                if materials[key]["scalars"].get("MultiplyVertexColor") and "colors" in data:
                    kw["colors"] = data["colors"]
                is_surface = (m.get("depth_write", True) and m.get("category", "surface") == "surface"
                              and m["name"] not in below)
                if is_surface:
                    actor["folder"] = "Geometry"
                    counts["surface"] += 1
                    counts["cloth"] += flutter
                    surfaces_for_ray.append((pos_b, tris))
                    if flutter:     # banners come as front/back twins that move together
                        cloth_actors.append((actor, pos_b, tris))
                    else:
                        solids.append((pos_b, tris))
                else:
                    actor.update(folder="Effects (hidden)", hidden=True)
                    counts["effect"] += 1
        kw["uvs"] = uv_list
        actor["material"] = materials[key]["name"]
        actor["glb"] = chunks.add(name, len(tris), **kw)
        if solo and actor.get("folder") == "Particles":
            w = GlbWriter()
            w.add_mesh(name, **kw)
            w.save(mesh_dir / "solo" / f"{name}.glb")    # single-mesh file: the asset is named after it
            solo_glbs[m["name"]] = str((mesh_dir / "solo" / f"{name}.glb").resolve())
        actors.append(actor)
    chunks.flush()
    write_snowfall(out / "SM_Snowfall.glb")
    # Custom primitive data 0-11 of M_GTB_Cloth: front (Unreal xy), bottom held, sides held,
    # then the 8 direction budgets (cm per unit motion weight, see cloth_room).
    # Banners hung on walls and in frames must not flutter into them.
    soup = TriSoup(solids)
    for actor, pos_b, tris in cloth_actors:
        n, budgets, held, sides = cloth_room(pos_b, tris, soup)
        nu = dir_to_ue([n[0], n[1], 0.0])
        # Unreal's y flip mirrors the across axis, so direction k becomes -k.
        budgets = [budgets[-k % 8] for k in range(8)]
        if nu[0] < 0 or (nu[0] == 0 and nu[1] < 0):      # one front direction for twins
            nu, budgets = -nu, [budgets[(k + 4) % 8] for k in range(8)]
        actor["cloth"] = ([round(float(nu[0]), 4), round(float(nu[1]), 4), 1.0 if held else 0.0, 1.0 if sides else 0.0]
                          + [round(min(b * CM, 10000.0), 1) for b in budgets])

    # Smoke columns (the generator's is drawn from a live render target the ripper
    # can't save): rebuilt from the game's own smoke flipbook, the sheet with the
    # most frames.
    plume = None
    if manifest.get("smoke") and sprite_frames_by_atlas:
        atlas = max(sprite_frames_by_atlas, key=lambda a: len(sprite_frames_by_atlas[a]))
        spec = texture_spec(textures[atlas], "linear")
        use_tex(spec)
        write_plume(out / "SM_Plume.glb", sorted(sprite_frames_by_atlas[atlas]))
        plumes = []
        for pl in manifest["smoke"]:
            pts = np.array(pl["points"], dtype=np.float64)
            low, top = pts[np.argmin(pts[:, 2])], pts[np.argmax(pts[:, 2])]
            plumes.append({"location": r3(to_ue(low[:3]), 2), "radius_cm": round(float(low[3]) * CM, 1),
                           "column_cm": round(float(top[2] - low[2]) * CM, 1), "source": pl.get("source", "")})
        plume = {"glb": str((out / "SM_Plume.glb").resolve()), "atlas": spec, "plumes": plumes}

    cam = manifest.get("camera", {})
    M = np.array(cam.get("matrix_world") or np.eye(4), dtype=np.float64)
    w, h = manifest["resolution"]
    fov_y = float(cam.get("fov_y_deg") or 60.0)
    target, closeups = closeup_views(M, surfaces_for_ray)
    camera = {"name": "GameCamera", "location": r3(to_ue(M[:3, 3]), 2),
              "forward": r3(dir_to_ue(-M[:3, 2] / np.linalg.norm(M[:3, 2])), 6),
              "up": r3(dir_to_ue(M[:3, 1] / np.linalg.norm(M[:3, 1])), 6),
              "fov_y_deg": fov_y, "aspect": w / h, "resolution": [w, h],
              "near_cm": round(max(cam.get("near") or 0.1, 0.01) * CM, 2),
              "focus_cm": round(float(np.linalg.norm(target - M[:3, 3])) * CM, 1)}

    floor = scene_floor(tops)
    lights = [{"location": r3(to_ue(L["location"]), 2), "range_cm": round(L["radius"] * CM, 2), "kind": L["kind"],
               "off": bool(L.get("off")) or (floor is not None and L["location"][2] < floor - 100.0)}
              for L in manifest.get("lights", [])]
    box = manifest.get("snow_box")
    plan = {
        "capture": capture.name,
        "capture_dir": str(capture.resolve()),
        "textures_dir": str((capture / "textures").resolve()),
        "meshes_dir": str(mesh_dir.resolve()),
        "snowfall_glb": str((out / "SM_Snowfall.glb").resolve()),
        "glb_meshes": chunks.files,
        "textures": sorted(used_textures.values(), key=lambda s: s["file"]),
        "materials": list(materials.values()),
        "actors": actors,
        "camera": camera,
        "closeups": closeups,
        "game_lights": lights,
        # Blender-space snow box, kept for the look mapping (box_scale is a look setting).
        "snow_box_blender": box,
        "plume": plume,
        "counts": counts,
    }
    if solo:
        plan["solo_glbs"] = solo_glbs
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"[ue] plan: {counts}, {len(materials)} materials, {len(used_textures)} textures, "
          f"{len(chunks.files)} glb chunks, {len(lights)} game lights")
    return plan


if __name__ == "__main__":
    export(Path(sys.argv[1]))
