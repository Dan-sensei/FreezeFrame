"""Engine-independent scene rules shared by the Blender and Unreal builders:
the default look, look.json merging and which texture fills which material role."""
import json
from pathlib import Path

DEFAULT_LOOK = {
    "view_transform": "AgX",
    "look": "None",
    "exposure": 0.0,
    "gamma": 1.0,
    "sun": {"azimuth": 135.0, "elevation": 35.0, "strength": 4.0,
            "color": [1.0, 0.96, 0.9], "angle": 1.5},
    # type: any Blender sky_type (MULTIPLE_SCATTERING, SINGLE_SCATTERING, ...) or "color"
    "sky": {"type": "MULTIPLE_SCATTERING", "strength": 0.35, "color": [0.5, 0.6, 0.75],
            "air_density": 1.0, "aerosol_density": 1.0, "ozone_density": 1.0},
    # Distance fog done in the compositor from the depth pass, like games do:
    # opacity = min(max_opacity, 1 - exp(-density * max(depth - start, 0))).
    "fog": {"density": 0.0, "color": [0.8, 0.85, 0.9], "strength": 1.0,
            "start": 0.0, "max_opacity": 1.0},
    "bloom": {"strength": 0.0, "threshold": 1.0, "size": 0.5},
    "grade": {"saturation": 1.0, "contrast": 0.0, "brightness": 0.0,
              "lift": [1, 1, 1], "gamma": [1, 1, 1], "gain": [1, 1, 1]},
    # snow_cover: 0..1 snow on up-facing surfaces (the game adds it in-shader);
    # snow_threshold: how flat a surface must be (world normal Z) to hold snow.
    "materials": {"albedo_gain": 1.0, "saturation": 1.0, "roughness": 0.65,
                  "roughness_from_gray": 1.0, "normal_strength": 1.0, "specular": 0.5,
                  "emission_strength": 1.0, "snow_cover": 0.0, "snow_threshold": 0.55},
    "lights": [],   # [{type, location, color, energy, radius, rotation?}]
    # Volumetric smoke built from the game's smoke particles. Plumes continue
    # `rise` units above what the game drew, widening by `grow` per unit and
    # drifting with the plume's own lean; noise billows/rises at `speed`.
    # The game's own particle sprites (smoke/steam flipbooks with normal maps,
    # fire flipbooks), re-faced to the render camera. rise: units/second drift.
    "particles": {"enabled": True, "billboard": True, "opacity": 0.7, "smoke_color": [0.7, 0.72, 0.78],
                  "translucency": 0.6, "ambient": 0.25, "normal_strength": 1.0, "fire_strength": 6.0,
                  "rise": 0.6},
    # Procedural snowfall (geometry nodes): positions are a pure function of the
    # frame, so any frame renders directly and scrubbing works. Units: world
    # units per second; wind is [x, y]. box_scale grows/shrinks the capture box.
    # follow_camera: snow only in a box of +-extent around the active camera
    # (count is then per that box); otherwise the capture's snow volume.
    "snow": {"enabled": True, "follow_camera": True, "extent": [45.0, 45.0, 30.0],
             "count": 60000, "size": 0.1, "opacity": 0.85, "fall_speed": 4.0, "wind": [2.5, 1.0],
             "flutter": 1.2, "brightness": 1.0, "box_scale": 1.0, "seed": 1},
    # emitters: extra plumes [[x, y, z, radius], ...] in world units (e.g. chimneys).
    "smoke": {"enabled": True, "density": 0.35, "color": [0.33, 0.34, 0.38], "anisotropy": 0.3,
              "rise": 60.0, "grow": 0.03, "spread": 0.9, "noise_scale": 0.25, "erosion": [0.4, 0.7],
              "emitters": [], "distortion": 0.6,
              "speed": 1.0, "voxel_size": 0.5},
    # Lights recovered from the game's light volumes. Power scales with each
    # light's range squared; cutoff at its range like the game's falloff.
    # power = power_per_range2 * range ** range_exponent (2 = inverse-square
    # physical scaling; ~1 keeps big-range game lights from flooding the scene).
    "game_lights": {"enabled": True, "power_per_range2": 2.0, "range_exponent": 2.0,
                    "color": [1.0, 0.55, 0.25], "hemi_scale": 1.0, "soft_size": 0.3,
                    # Hundreds of shadowed lights overflow EEVEE's shadow pool (and
                    # games rarely shadow small lamps anyway).
                    "shadows": False,
                    # Multiplies each light's cutoff distance (1 = the game volume's radius).
                    "range_scale": 1.0},
    "render": {"engine": "BLENDER_EEVEE", "samples": 32, "preview_scale": 0.5,
               "raytracing": True, "shadow_ray_count": 1,
               # Bounce light strength. Games usually have far less than EEVEE's
               # physically based GI, especially off snow.
               "indirect_intensity": 1.0},
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_look(path: Path):
    user = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return deep_merge(DEFAULT_LOOK, user)


def srgb_to_linear(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def assign_slots(mesh_tex, textures, profile):
    """Pick one texture per role for a draw call; profile slot map wins."""
    slot_roles = {str(k): v for k, v in profile.get("slot_roles", {}).items()}
    slots, extras = {}, 0
    # With several colour-looking textures, the largest is the material's albedo;
    # small ones are masks/detail. Visit it first so it claims the role.
    def order(kv):
        e = textures.get(kv[1]) or {}
        is_albedo = e.get("role") == "albedo" and str(kv[0]) not in slot_roles
        size = (e.get("size") or [0, 0])[0] * (e.get("size") or [0, 0])[1]
        return (0, -size, int(kv[0])) if is_albedo else (1, 0, int(kv[0]))
    for slot, tid in sorted(mesh_tex.items(), key=order):
        entry = textures.get(tid)
        if not entry or "file" not in entry:
            continue
        if entry["role"] == "shared":  # engine-global texture, never a material input
            continue
        role = slot_roles.get(str(slot), entry["role"])
        if role in ("ignore", "constant", "environment", "hdr"):
            continue
        if ":" in role:  # e.g. "normal:AG" - role plus channel layout from the profile
            role, channels = role.split(":", 1)
            entry = dict(entry, details=dict(entry.get("details", {}), channels=channels))
        if role in slots:
            role, extras = f"extra{extras}", extras + 1
        slots[role] = dict(entry, role=role if not role.startswith("extra") else entry["role"])
    # Engines almost always bind the base colour first; if nothing looked like
    # albedo, promote the lowest-slot colour texture rather than render grey.
    # No colour texture: a large grayscale one is most likely a desaturated
    # albedo (metal, stone) rather than a roughness map, which would come
    # alongside a colour texture. Only after that fall back to a packed one.
    # Exception: when that grayscale map is the material's only texture and is
    # mostly black, it is a mask over a colour the shader computes (Frostpunk's
    # frost streaks on the snow plateau), not a base colour. It becomes a "mask"
    # (not wired), so the material keeps the constant ground/snow colour.
    # Also a mask: a bright map whose edges fade to black (a stamped terrain
    # snow/height blend) - tiling base colours look the same at the edges.
    if "albedo" not in slots and "gray" in slots and (slots["gray"].get("size") or [0])[0] >= 512:
        d = slots["gray"].get("details", {})
        mean = d.get("mean_rgb") or [1.0]
        mean = sum(mean) / len(mean)
        border = d.get("border_mean")
        stamped = border is not None and border < 0.05 and mean > 0.15
        if (len(slots) == 1 and mean < 0.1) or stamped:
            slots["mask"] = dict(slots.pop("gray"), role="mask")
        else:
            slots["albedo"] = dict(slots.pop("gray"), role="albedo")
    if "albedo" not in slots and "albedo" not in slot_roles.values():
        for role in ("packed",) + tuple(r for r in slots if r.startswith("extra")):
            if role in slots and slots[role]["role"] in ("packed", "albedo"):
                slots["albedo"] = dict(slots.pop(role), role="albedo")
                break
    return slots
