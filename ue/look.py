"""look.json -> Unreal values (runs in normal Python).

Blender's units carry over one to one, which the probes confirmed:
  sun strength (W/m2)       -> directional light illuminance (lux)
  point light power P (W)   -> intensity P / (4 pi) candela
  world colour x strength   -> sky luminance (cd/m2)
so the same look.json lights both scenes the same. Colour (grade, exposure,
AgX look) goes through a LUT baked by Blender itself (ue/blender_lut.py).

Keys under look["unreal"] only affect Unreal; Blender ignores them:
  exposure_offset  stops added before the view transform (Unreal calibration)
  bloom_intensity / bloom_threshold   override the mapped bloom
  warmup_frames, temporal_samples     Movie Render Queue quality
  cvars            console variables for Movie Render Queue renders, e.g. {"r.Lumen.ScreenProbeGather.DownsampleFactor": 8}
"""
import math

import numpy as np

from gtb.scene_common import srgb_to_linear
from ue.export import CM, dir_to_ue, to_ue

LUT_SIZE, LUT_LO, LUT_HI = 64, -12.0, 8.0
UNREAL_DEFAULTS = {"exposure_offset": 0.0, "bloom_intensity": None, "bloom_threshold": None,
                   "warmup_frames": 64, "temporal_samples": 8, "anim_frames": 250, "fps": 24, "cvars": {}}


def lut_key(look):
    """Everything the baked LUT depends on (grade + view transform)."""
    g = look["grade"]
    return repr((look["view_transform"], look["look"], round(look["exposure"], 4), round(look["gamma"], 4),
                 [g.get(k) for k in ("saturation", "contrast", "brightness", "lift", "gamma", "gain")],
                 LUT_SIZE, LUT_LO, LUT_HI))


def lut_display_linear(lut_path, x):
    """Where a grey scene-linear value x ends up after Blender's view transform
    (display value decoded back to linear), read from the baked LUT's diagonal."""
    import cv2
    lut = cv2.imread(str(lut_path), cv2.IMREAD_UNCHANGED).astype(np.float64) / 65535.0
    n = lut.shape[0]
    diag = np.array([lut[i, i * n + i, :3].mean() for i in range(n)])
    t = np.clip((math.log2(max(x, 1e-12)) - LUT_LO) / (LUT_HI - LUT_LO), 0, 1) * (n - 1)
    d = float(np.interp(t, np.arange(n), diag))
    return round(d / 12.92 if d <= 0.04045 else ((d + 0.055) / 1.055) ** 2.4, 4)


def _euler_dir(rot_deg, local):
    """Blender XYZ Euler (degrees) applied to a local vector."""
    x, y, z = (math.radians(a) for a in rot_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx @ np.asarray(local, float)


def rl(v, nd=5):
    return [round(float(x), nd) for x in v]


def ue_look(look, plan, lut_path):
    u = dict(UNREAL_DEFAULTS, **(look.get("unreal") or {}))
    s = look["sun"]
    az, el = math.radians(s["azimuth"]), math.radians(s["elevation"])
    to_sun = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    sky = look["sky"]
    fog = look["fog"]
    mats = look["materials"]
    gl = look["game_lights"]
    parts = look["particles"]
    snow = look["snow"]
    r = look["render"]

    # Blender's bloom (compositor glare) thresholds scene-linear values before the
    # view transform. Unreal's bloom runs after GTB's colour pass, on display-
    # referred values, so the threshold goes through the same LUT.
    b = look["bloom"]
    bloom = {"intensity": (u["bloom_intensity"] if u["bloom_intensity"] is not None else b["strength"] * 0.8)
             if b["strength"] > 0 else 0.0,
             "threshold": u["bloom_threshold"] if u["bloom_threshold"] is not None
             else lut_display_linear(lut_path, b["threshold"]),
             "size_scale": 4.0 * max(b["size"], 0.05) / 0.5}

    out = {
        "lut": {"file": str(lut_path), "size": LUT_SIZE, "lo": LUT_LO, "hi": LUT_HI},
        "exposure_scale": 2.0 ** u["exposure_offset"],
        "sun": {"forward": rl(dir_to_ue(-to_sun)), "lux": s["strength"], "color": rl(s["color"]),
                "angle": s["angle"]},
        "sky": {"type": sky["type"], "color": rl(np.array(sky["color"]) * sky["strength"]),
                "strength": sky["strength"], "tint": rl(sky["color"]) if sky.get("tint") else [1, 1, 1]},
        # Blender: opacity = min(max_opacity, 1 - exp(-density * (depth - start))), depth in metres.
        # Unreal (HeightFogCommon.ush, falloff -> 0): 1 - exp2(-(FogDensity / 1000) * ln2 * d_cm),
        # so FogDensity = 10 * density / ln(2)^2.
        "fog": {"enabled": fog["density"] > 0, "density": 10.0 * fog["density"] / math.log(2) ** 2,
                "start_cm": fog["start"] * CM, "max_opacity": fog["max_opacity"],
                "color": rl(np.array(fog["color"]) * fog["strength"])},
        "bloom": bloom,
        "indirect_intensity": r.get("indirect_intensity", 1.0),
        "surface": {
            "AlbedoGain": mats["albedo_gain"], "Saturation": mats["saturation"], "Roughness": mats["roughness"],
            "RoughnessFromGray": mats["roughness_from_gray"], "NormalStrength": mats["normal_strength"],
            "Specular": mats["specular"], "EmissionStrength": mats["emission_strength"],
            "SnowCover": mats["snow_cover"], "SnowThreshold": mats["snow_threshold"],
            "GroundColorWeight": 1.0 if mats.get("ground_color") else 0.0,
        },
        "ground_color": srgb_to_linear(mats.get("ground_color") or [0.5, 0.5, 0.5]) + [1.0],
        "drift_color": srgb_to_linear(mats.get("snow_color") or [0.86, 0.9, 0.97]) + [1.0],
        "normal_strength": mats["normal_strength"],
        "game_lights": {"enabled": gl["enabled"], "color": rl(gl["color"]), "shadows": gl["shadows"],
                        "source_radius_cm": gl["soft_size"] * CM,
                        "lights": [{"candela": gl["power_per_range2"] * (L["range_cm"] / CM) ** gl["range_exponent"]
                                    * (gl["hemi_scale"] if L["kind"] == "hemi" else 1.0) / (4 * math.pi),
                                    "radius_cm": L["range_cm"] * gl["range_scale"]} for L in plan["game_lights"]]},
        "lights": [_extra_light(L) for L in look.get("lights", [])],
        "particles": {"enabled": parts["enabled"], "Opacity": parts["opacity"],
                      "Billboard": 1.0 if parts["billboard"] else 0.0, "Rise": parts["rise"] * CM,
                      "Translucency": parts["translucency"], "Ambient": parts["ambient"],
                      "NormalStrength": parts["normal_strength"], "FireStrength": parts["fire_strength"],
                      "SmokeColor": srgb_to_linear(parts["smoke_color"]) + [1.0]},
        "snow": _snow(snow, plan),
        "render": {"preview_scale": r["preview_scale"], "resolution": plan["camera"]["resolution"],
                   "warmup_frames": u["warmup_frames"], "temporal_samples": u["temporal_samples"],
                   "anim_frames": u["anim_frames"], "fps": u["fps"], "cvars": u["cvars"]},
        "smoke_volumes": look["smoke"]["enabled"],
    }
    return out


def _extra_light(L):
    """look.json "lights" entries (Blender lights in Blender space)."""
    kind = L.get("type", "POINT")
    rot = L.get("rotation", [0, 0, 0])
    fwd = dir_to_ue(_euler_dir(rot, [0, 0, -1]))       # Blender lights shine along local -Z
    energy = L.get("energy", 100.0)
    intensity = {"SUN": energy, "AREA": energy / math.pi}.get(kind, energy / (4 * math.pi))
    return {"type": kind, "location": rl(to_ue(L["location"]), 2), "forward": rl(fwd),
            "color": rl(L.get("color", [1, 1, 1])), "intensity": intensity,
            "source_radius_cm": L.get("radius", 0.2) * CM, "name": L.get("name", "")}


def _snow(cfg, plan):
    box = plan.get("snow_box_blender") or [[-50, -50, -50], [50, 50, 0]]
    lo, hi = (np.array(v, float) for v in box)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * cfg["box_scale"]
    a, b = to_ue(c - half), to_ue(c + half)                # y flips, so re-sort each axis
    from ue.export import SNOW_MAX_FLAKES
    return {"enabled": cfg["enabled"], "FollowCamera": 1.0 if cfg["follow_camera"] else 0.0,
            "Extent": rl(np.array(cfg["extent"]) * CM, 2) + [0.0],
            "BoxMin": rl(np.minimum(a, b), 2) + [0.0], "BoxMax": rl(np.maximum(a, b), 2) + [0.0],
            "Wind": [cfg["wind"][0] * CM, -cfg["wind"][1] * CM, 0.0, 0.0],
            "FallSpeed": cfg["fall_speed"] * CM, "Flutter": cfg["flutter"] * CM, "Size": cfg["size"] * CM,
            "CountFraction": min(cfg["count"] / SNOW_MAX_FLAKES, 1.0), "Opacity": cfg["opacity"],
            "Brightness": cfg["brightness"], "seed": cfg["seed"]}
