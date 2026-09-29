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
from gtb.scene_common import assign_slots, srgb_to_linear  # noqa: E402
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
            _channel_targets({k: v for k, v in profile.get("normal_packed_channels", {}).items() if k in "RB"},
                             "Normal", sc, vec)
    gray = slots.get("gray")
    if gray:
        tex["Gray"] = texture_spec(gray, "linear")
        sc["UseGray"] = 1.0
        sc["GrayIsGloss"] = 1.0 if profile.get("gray_is_gloss") else 0.0
    packed = slots.get("packed")
    if packed and profile.get("packed_channels"):
        tex["Packed"] = texture_spec(packed, "linear")
        sc["UsePacked"] = 1.0
        _channel_targets(profile["packed_channels"], "Packed", sc, vec)
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


def write_plume(path: Path, frames, count=128, seed=7):
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

def export(capture: Path):
    capture = Path(capture)
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    profile = manifest.get("profile", {})
    textures = manifest["textures"]
    out = capture / "unreal"
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in mesh_dir.glob("*.glb"):
        old.unlink()

    chunks = ChunkedGlb(mesh_dir, "geo")
    materials, used_textures, actors = {}, {}, []
    surfaces_for_ray = []
    sprite_frames_by_atlas = {}
    counts = {"surface": 0, "effect": 0, "sprite": 0, "snowdrift": 0}

    def use_tex(spec):
        prev = used_textures.get(spec["file"])
        # A file used both as colour and data keeps the colour (sRGB) import.
        if not prev or (prev["kind"] != "albedo" and spec["kind"] == "albedo"):
            used_textures[spec["file"]] = spec

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

        if m.get("category") == "sprite" and m.get("sprite"):
            sp = m["sprite"]
            key = f"sprite|{sp['kind']}|{sp['atlas']}"
            if key not in materials:
                entry = textures[sp["atlas"]]
                spec = texture_spec(entry, "albedo" if sp["kind"] == "fire" else "linear")
                use_tex(spec)
                materials[key] = {"name": f"MI_Sprite_{sp['kind']}_{sp['atlas']}",
                                  "parent": "fire" if sp["kind"] == "fire" else "smoke",
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
            if sp["kind"] == "smoke":
                for k in np.unique(puff_id):
                    uv = data["uv0"][puff_id == k]
                    sprite_frames_by_atlas.setdefault(sp["atlas"], set()).add(
                        tuple(np.round([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], 4)))
            kw["colors"] = data["colors"] if "colors" in data else np.ones((len(pos_b), 4))
            kw["normals"] = blender_to_gltf(smooth_normals(pos_b, tris))
            actor.update(cast_shadow=False, folder="Particles")
            counts["sprite"] += 1
            surfaces_for_ray.append((pos_b, tris))   # Blender's close-up ray hits sprites too
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
            else:
                slots = assign_slots(m.get("textures", {}), textures, profile)
                has_uv = "uv0" in data
                key = m.get("material_key") or json.dumps(
                    {r: e["file"] for r, e in sorted(slots.items())}, sort_keys=True) + ("" if has_uv else "|nouv")
                if key not in materials:
                    mat = surface_material(slots, profile, has_uv)
                    mat["name"] = f"MI_M{len(materials):03d}"
                    for spec in mat["textures"].values():
                        use_tex(spec)
                    materials[key] = mat
                if materials[key]["scalars"].get("MultiplyVertexColor") and "colors" in data:
                    kw["colors"] = data["colors"]
                is_surface = m.get("depth_write", True) and m.get("category", "surface") == "surface"
                if is_surface:
                    actor["folder"] = "Geometry"
                    counts["surface"] += 1
                    surfaces_for_ray.append((pos_b, tris))
                else:
                    actor.update(folder="Effects (hidden)", hidden=True)
                    counts["effect"] += 1
        kw["uvs"] = uv_list
        actor["material"] = materials[key]["name"]
        actor["glb"] = chunks.add(name, len(tris), **kw)
        actors.append(actor)
    chunks.flush()
    write_snowfall(out / "SM_Snowfall.glb")

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

    lights = [{"location": r3(to_ue(L["location"]), 2), "range_cm": round(L["radius"] * CM, 2), "kind": L["kind"]}
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
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"[ue] plan: {counts}, {len(materials)} materials, {len(used_textures)} textures, "
          f"{len(chunks.files)} glb chunks, {len(lights)} game lights")
    return plan


if __name__ == "__main__":
    export(Path(sys.argv[1]))
