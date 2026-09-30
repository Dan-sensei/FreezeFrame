"""Builds a Blender scene from a FreezeFrame manifest and applies look.json.

Runs inside Blender (5.x). The manifest is produced by `gtb.py process`; its
geometry is already converted to Blender space (right-handed, Z up, metres).

manifest.json
  resolution: [w, h]            screenshot size, used for render resolution
  screenshot: "screenshot.png"
  camera: {matrix_world: 4x4 row-major | null, fov_y_deg, near, far}
  textures: {id: {file, role, details}}
  meshes: [{name, file (npz), textures: {slot: id}, material_key?}]
  profile: {...}                optional game profile (slot roles, conventions)

npz arrays: positions (N,3) f32, indices (T,3) i32, optional normals (N,3),
uv0/uv1 (N,2), colors (N,4).
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtb.scene_common import DEFAULT_LOOK, assign_slots, deep_merge, load_look, packed_channels, srgb_to_linear  # noqa: E402,F401

PARAMS_GROUP = "GTB_Params"


# --------------------------------------------------------------------------- scene reset

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Capture"
    return scene


# --------------------------------------------------------------------------- meshes

def build_mesh(name, data):
    pos = data["positions"].astype(np.float32)
    tris = data["indices"].astype(np.int32).reshape(-1, 3)
    n, t = len(pos), len(tris)
    me = bpy.data.meshes.new(name)
    me.vertices.add(n)
    me.vertices.foreach_set("co", pos.ravel())
    me.loops.add(t * 3)
    me.loops.foreach_set("vertex_index", tris.ravel())
    me.polygons.add(t)
    me.polygons.foreach_set("loop_start", np.arange(0, t * 3, 3, dtype=np.int32))

    corner = tris.ravel()
    for key in ("uv0", "uv1"):
        if key in data:
            uv = data[key].astype(np.float32)[corner]
            layer = me.uv_layers.new(name=key.upper())
            layer.data.foreach_set("uv", uv.ravel())
    if "colors" in data:
        col = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        col.data.foreach_set("color", data["colors"].astype(np.float32).ravel())

    for key in data:
        if key.startswith("attr_"):  # extra per-vertex data (e.g. sprite centre/corner)
            arr = data[key].astype(np.float32)
            kind = {3: "FLOAT_VECTOR", 2: "FLOAT2"}.get(arr.shape[1] if arr.ndim == 2 else 1, "FLOAT")
            me.attributes.new(key[5:], kind, "POINT").data.foreach_set(
                "vector" if kind != "FLOAT" else "value", arr.ravel())

    me.update()
    me.validate(clean_customdata=False)
    if "normals" in data and len(me.vertices) == n:
        nrm = data["normals"].astype(np.float32)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
        me.normals_split_custom_set_from_vertices(nrm)
    return me


# --------------------------------------------------------------------------- materials

def params_group():
    """One node group every material reads from, so look.json tweaks are global."""
    ng = bpy.data.node_groups.get(PARAMS_GROUP)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(PARAMS_GROUP, "ShaderNodeTree")
    out = ng.nodes.new("NodeGroupOutput")
    for i, name in enumerate(k for k in DEFAULT_LOOK["materials"] if k not in ("ground_color", "snow_color")):
        ng.interface.new_socket(name, in_out="OUTPUT", socket_type="NodeSocketFloat")
        v = ng.nodes.new("ShaderNodeValue")
        v.name = v.label = name
        v.location = (-200, -i * 90)
        ng.links.new(v.outputs[0], out.inputs[name])
    return ng


def set_params(values):
    ng = params_group()
    for k, v in values.items():
        if k in ng.nodes and isinstance(v, (int, float)):
            ng.nodes[k].outputs[0].default_value = float(v)


def load_image(path: Path, non_color):
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    # Games pack masks into alpha; EEVEE would premultiply colour by it (alpha 0
    # -> black). Channel-packed keeps RGB and A independent.
    img.alpha_mode = "NONE" if non_color else "CHANNEL_PACKED"
    return img


def math_node(nt, op, a=None, b=None, loc=(0, 0), c=None):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    m.location = loc
    for i, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            m.inputs[i].default_value = x
        else:
            nt.links.new(x, m.inputs[i])
    return m.outputs[0]


def build_material(name, slots, tex_dir: Path, profile, has_uv=True):
    """slots: {role: texture_entry}; texture_entry has file/role/details.

    Meshes without UVs (e.g. Frostpunk's heightfield terrain) can't map any
    texture, so they get the albedo's average colour instead."""
    if not has_uv:
        albedo = slots.get("albedo") or next(iter(slots.values()), None)
        slots = {}
        mean = (albedo or {}).get("details", {}).get("mean_rgb")
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = params_group()
    grp.location = (-900, 300)
    P = grp.outputs
    normal_rough = None  # roughness socket found inside a packed normal texture
    nt.links.new(P["specular"], bsdf.inputs["Specular IOR Level"])
    y = 0

    def tex(entry, non_color):
        nonlocal y
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load_image(tex_dir / entry["file"], non_color)
        node.location = (-700, y)
        node.label = entry["role"]
        y -= 300
        return node

    # Base colour (x gain, saturation) and alpha.
    albedo = slots.get("albedo")
    if albedo:
        t = tex(albedo, False)
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.location = (-350, 250)
        nt.links.new(t.outputs["Color"], hsv.inputs["Color"])
        nt.links.new(P["saturation"], hsv.inputs["Saturation"])
        nt.links.new(P["albedo_gain"], hsv.inputs["Value"])
        color = hsv.outputs["Color"]
        if profile.get("multiply_vertex_color"):
            vc = nt.nodes.new("ShaderNodeVertexColor")
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            nt.links.new(color, mix.inputs["A"])
            nt.links.new(vc.outputs["Color"], mix.inputs["B"])
            color = mix.outputs["Result"]
        nt.links.new(color, bsdf.inputs["Base Color"])
        if albedo.get("details", {}).get("has_alpha") and not profile.get("ignore_albedo_alpha"):
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.surface_render_method = "DITHERED"
    elif not has_uv and (profile.get("no_uv_base_color") or mean):
        mat["gtb_const"] = "ground"
        # Profile colour wins: a UV-less mesh's textures are usually overlays/masks.
        bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(profile.get("no_uv_base_color") or mean), 1)
    elif not slots:  # the ripper failed to save any texture for this draw
        bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(profile.get("untextured_base_color", [0.3, 0.3, 0.32])), 1)
    else:
        mat["gtb_const"] = "ground"
        bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(profile.get("default_base_color", [0.5, 0.5, 0.5])), 1)

    # Normal map: rebuild Z for two-channel formats, flip green for DirectX.
    normal = slots.get("normal")
    if normal:
        t = tex(normal, True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-450, y + 300)
        nt.links.new(t.outputs["Color"], sep.inputs[0])
        channels = normal.get("details", {}).get("channels", "RGB")
        x = t.outputs["Alpha"] if channels == "AG" else sep.outputs["Red"]
        g = sep.outputs["Green"]
        if profile.get("normal_flip_green", True):
            g = math_node(nt, "SUBTRACT", 1.0, g, (-300, y + 250))
        if channels == "RGB":
            z = sep.outputs["Blue"]
        else:  # z = sqrt(1 - x'^2 - y'^2) in -1..1 space, remapped back to 0..1
            xs = math_node(nt, "MULTIPLY_ADD", x, 2.0, (-300, y + 150), -1.0)
            ys = math_node(nt, "MULTIPLY_ADD", g, 2.0, (-300, y + 100), -1.0)
            d = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "MULTIPLY", xs, xs), (-150, y + 150))
            d = math_node(nt, "SUBTRACT", d, math_node(nt, "MULTIPLY", ys, ys), (-100, y + 150))
            z = math_node(nt, "SQRT", math_node(nt, "MAXIMUM", d, 0.0), None, (-50, y + 150))
            z = math_node(nt, "MULTIPLY_ADD", z, 0.5, (0, y + 150), 0.5)
        comb = nt.nodes.new("ShaderNodeCombineColor")
        comb.location = (-50, y + 300)
        nt.links.new(x, comb.inputs[0])
        nt.links.new(g, comb.inputs[1])
        nt.links.new(z, comb.inputs[2])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (100, -300)
        nt.links.new(comb.outputs[0], nm.inputs["Color"])
        nt.links.new(P["normal_strength"], nm.inputs["Strength"])
        nt.links.new(nm.outputs[0], bsdf.inputs["Normal"])
        # Engines that pack the normal into A+G often put masks in R and B.
        if channels == "AG":
            names = {"R": "Red", "B": "Blue"}
            for ch, target in packed_channels(profile.get("normal_packed_channels", {}), normal).items():
                src = sep.outputs[names[ch]]
                if target.startswith("1-"):
                    target, src = target[2:], math_node(nt, "SUBTRACT", 1.0, src)
                if target == "Roughness":
                    normal_rough = src
                else:
                    nt.links.new(src, bsdf.inputs[target])

    # Roughness: from a grayscale map (gloss maps inverted per profile) blended
    # with the global default by roughness_from_gray.
    rough = normal_rough or P["roughness"]
    gray = slots.get("gray")
    if gray:
        t = tex(gray, True)
        r = t.outputs["Color"]
        if profile.get("gray_is_gloss"):
            r = math_node(nt, "SUBTRACT", 1.0, r)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.location = (-100, -50)
        nt.links.new(P["roughness_from_gray"], mix.inputs["Factor"])
        nt.links.new(P["roughness"], mix.inputs["A"])
        nt.links.new(r, mix.inputs["B"])
        rough = mix.outputs["Result"]

    # Packed masks: profile maps channels to inputs, e.g. {"R": "Metallic", "G": "Roughness"}.
    packed = slots.get("packed")
    if packed:
        t = tex(packed, True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-400, y + 300)
        nt.links.new(t.outputs["Color"], sep.inputs[0])
        names = {"R": "Red", "G": "Green", "B": "Blue"}
        for ch, target in packed_channels(profile.get("packed_channels", {}), packed).items():
            src = t.outputs["Alpha"] if ch == "A" else sep.outputs[names[ch]]
            if target.startswith("1-"):
                target, src = target[2:], math_node(nt, "SUBTRACT", 1.0, src)
            if target == "Roughness":
                rough = src
            else:
                nt.links.new(src, bsdf.inputs[target])
    nt.links.new(rough, bsdf.inputs["Roughness"])

    emissive = slots.get("emissive")
    if emissive:
        t = tex(emissive, False)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Emission Color"])
        # Principled's emission colour defaults to white, so strength is only
        # driven for materials that actually have an emissive texture.
        nt.links.new(P["emission_strength"], bsdf.inputs["Emission Strength"])

    if mat.get("gtb_const") != "ground":  # the untextured ground already is snow
        add_surface_snow(nt, bsdf, P)

    # Anything we couldn't place stays in the tree (unconnected) for manual wiring.
    for role, entry in slots.items():
        if role.startswith("extra"):
            tex(entry, entry.get("role") != "albedo")
    return mat


def build_snowmesh_material(entry, tex_dir, profile):
    """Snow drifts: snow colour, the shared snow texture's RG as a normal map."""
    mat = bpy.data.materials.new("SnowDrift")
    mat["gtb_const"] = "snow"
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = next(n for n in N if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear([0.86, 0.9, 0.97]), 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    grp = N.new("ShaderNodeGroup")
    grp.node_tree = params_group()
    t = N.new("ShaderNodeTexImage")
    t.image = load_image(tex_dir / entry["file"], True)
    sep = N.new("ShaderNodeSeparateColor")
    L.new(t.outputs["Color"], sep.inputs[0])
    g = math_node(nt, "SUBTRACT", 1.0, sep.outputs["Green"]) if profile.get("normal_flip_green", True) \
        else sep.outputs["Green"]
    comb = N.new("ShaderNodeCombineColor")
    L.new(sep.outputs["Red"], comb.inputs[0])
    L.new(g, comb.inputs[1])
    comb.inputs[2].default_value = 1.0
    nm = N.new("ShaderNodeNormalMap")
    L.new(comb.outputs[0], nm.inputs["Color"])
    L.new(grp.outputs["normal_strength"], nm.inputs["Strength"])
    L.new(nm.outputs[0], bsdf.inputs["Normal"])
    return mat


def add_surface_snow(nt, bsdf, P):
    """White snow on up-facing surfaces, broken up by noise; amount from GTB_Params."""
    N, L = nt.nodes, nt.links
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], sep.inputs[0])
    lo = math_node(nt, "SUBTRACT", P["snow_threshold"], 0.12)
    hi = math_node(nt, "ADD", P["snow_threshold"], 0.12)
    ramp = N.new("ShaderNodeMapRange")
    ramp.clamp = True
    ramp.interpolation_type = "SMOOTHSTEP"
    L.new(sep.outputs["Z"], ramp.inputs["Value"])
    L.new(lo, ramp.inputs["From Min"])
    L.new(hi, ramp.inputs["From Max"])
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    noise.inputs["Detail"].default_value = 8
    L.new(geo.outputs["Position"], noise.inputs["Vector"])
    patch = N.new("ShaderNodeMapRange")  # noise -> ragged cover edges
    patch.clamp = True
    L.new(noise.outputs["Fac"], patch.inputs["Value"])
    L.new(math_node(nt, "SUBTRACT", 1.0, P["snow_cover"]), patch.inputs["From Min"])
    L.new(math_node(nt, "ADD", math_node(nt, "SUBTRACT", 1.0, P["snow_cover"]), 0.2), patch.inputs["From Max"])
    fac = math_node(nt, "MULTIPLY", ramp.outputs["Result"], patch.outputs["Result"])
    for name, snow_value in (("Base Color", (0.86, 0.9, 0.97, 1.0)), ("Roughness", 0.55)):
        sock = bsdf.inputs[name]
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA" if name == "Base Color" else "FLOAT"
        a = next(i for i in mix.inputs if i.name == "A" and i.enabled)
        b = next(i for i in mix.inputs if i.name == "B" and i.enabled)
        out = next(o for o in mix.outputs if o.enabled)
        if sock.links:
            L.new(sock.links[0].from_socket, a)
        else:
            a.default_value = sock.default_value
        b.default_value = snow_value
        L.new(fac, mix.inputs["Factor"])
        L.new(out, sock)


# --------------------------------------------------------------------------- camera

def setup_camera(scene, cam_info, res, screenshot: Path | None):
    cam_data = bpy.data.cameras.new("GameCamera")
    cam = bpy.data.objects.new("GameCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    w, h = res
    scene.render.resolution_x, scene.render.resolution_y = w, h
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(cam_info.get("fov_y_deg") or 60.0)
    cam_data.clip_start = max(cam_info.get("near") or 0.1, 0.01)
    cam_data.clip_end = cam_info.get("far") or 5000.0
    if cam_info.get("matrix_world"):
        cam.matrix_world = Matrix(cam_info["matrix_world"])
    else:
        cam.location = cam_info.get("location", (0, 0, 0))
        # Default: looking down +Y (the game's forward axis after conversion).
        cam.rotation_euler = [math.radians(a) for a in cam_info.get("rotation_deg", (90, 0, 0))]
    if screenshot and screenshot.exists():
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.image = bpy.data.images.load(str(screenshot), check_existing=True)
        bg.alpha = 0.5
        bg.display_depth = "FRONT"
        bg.frame_method = "FIT"
    return cam


# --------------------------------------------------------------------------- look

def _world_nodes(scene):
    world = scene.world or bpy.data.worlds.new("GameWorld")
    scene.world = world
    world.use_nodes = True
    return world.node_tree


def apply_look(scene, look):
    vs = scene.view_settings
    vs.view_transform = look["view_transform"]
    try:
        vs.look = look["look"]
    except TypeError:
        vs.look = "None"
    vs.exposure = look["exposure"]
    vs.gamma = look["gamma"]

    r = look["render"]
    scene.render.engine = r["engine"]
    if r["engine"] == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = r["samples"]
        scene.eevee.use_raytracing = r["raytracing"]
        scene.eevee.shadow_ray_count = r["shadow_ray_count"]
        scene.eevee.indirect_light_intensity = r["indirect_intensity"]
        scene.eevee.volumetric_end = 2000.0
    else:
        scene.cycles.samples = r["samples"]

    # Sun: azimuth measured clockwise from +Y (Blender "north"), elevation above horizon.
    s = look["sun"]
    sun = bpy.data.objects.get("Sun")
    if not sun:
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
        scene.collection.objects.link(sun)
    az, el = math.radians(s["azimuth"]), math.radians(s["elevation"])
    to_sun = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    sun.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()
    sun.data.energy = s["strength"]
    sun.data.color = s["color"]
    sun.data.angle = math.radians(s["angle"])

    # World: physical sky (aligned with the sun) or flat colour, plus optional fog.
    nt = _world_nodes(scene)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    sky = look["sky"]
    bg.inputs["Strength"].default_value = sky["strength"]
    if sky["type"] == "color":
        bg.inputs["Color"].default_value = (*sky["color"], 1)
    else:
        tex = nt.nodes.new("ShaderNodeTexSky")
        tex.sky_type = sky["type"]
        if sky["type"] in ("SINGLE_SCATTERING", "MULTIPLE_SCATTERING"):
            tex.sun_elevation, tex.sun_rotation = el, -az
            tex.sun_disc = False  # the Sun lamp provides the direct light
            tex.air_density = sky["air_density"]
            tex.aerosol_density = sky["aerosol_density"]
            tex.ozone_density = sky["ozone_density"]
        else:
            tex.sun_direction = to_sun
        tint = nt.nodes.new("ShaderNodeMix")
        tint.data_type, tint.blend_type = "RGBA", "MULTIPLY"
        tint.inputs["Factor"].default_value = 1.0
        tint.inputs["B"].default_value = (*sky["color"], 1) if sky.get("tint") else (1, 1, 1, 1)
        nt.links.new(tex.outputs[0], tint.inputs["A"])
        nt.links.new(tint.outputs["Result"], bg.inputs["Color"])

    # Show fog/grading/bloom in the viewport too, not only in renders.
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.use_compositor = "ALWAYS"
    set_params(look["materials"])
    colours = {"ground": look["materials"].get("ground_color"),  # untextured terrain
               "snow": look["materials"].get("snow_color")}       # snow drift meshes
    for mat in bpy.data.materials:
        c = colours.get(mat.get("gtb_const"))
        if c:
            b = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
            b.inputs["Base Color"].default_value = (*srgb_to_linear(c), 1)
    apply_smoke(scene, look["smoke"])
    apply_snow(scene, look["snow"])
    apply_particles(scene, look["particles"])
    apply_lights(scene, look["lights"])
    apply_game_lights(look["game_lights"])
    apply_compositor(scene, look)


def apply_lights(scene, lights):
    coll = bpy.data.collections.get("GTB_Lights")
    if coll:
        for ob in list(coll.objects):
            bpy.data.objects.remove(ob)
    else:
        coll = bpy.data.collections.new("GTB_Lights")
        scene.collection.children.link(coll)
    for i, L in enumerate(lights):
        data = bpy.data.lights.new(f"L{i}", L.get("type", "POINT"))
        data.color = L.get("color", (1, 1, 1))
        data.energy = L.get("energy", 100.0)
        if hasattr(data, "shadow_soft_size"):
            data.shadow_soft_size = L.get("radius", 0.2)
        ob = bpy.data.objects.new(L.get("name", f"L{i}"), data)
        ob.location = L["location"]
        if "rotation" in L:
            ob.rotation_euler = [math.radians(a) for a in L["rotation"]]
        coll.objects.link(ob)


SMOKE_GN, SMOKE_MAT = "GTB_SmokeVolume", "GTB_Smoke"


def smoke_node_group():
    ng = bpy.data.node_groups.get(SMOKE_GN)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(SMOKE_GN, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    rad = n.new("GeometryNodeInputNamedAttribute")
    rad.data_type = "FLOAT"
    rad.inputs["Name"].default_value = "radius"
    m2p = n.new("GeometryNodeMeshToPoints")
    p2v = n.new("GeometryNodePointsToVolume")
    p2v.name = "ToVolume"
    p2v.inputs["Resolution Mode"].default_value = "Size"
    setm = n.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = smoke_material()
    L = ng.links
    L.new(gin.outputs[0], m2p.inputs["Mesh"])
    L.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
    L.new(m2p.outputs["Points"], p2v.inputs["Points"])
    L.new(rad.outputs["Attribute"], p2v.inputs["Radius"])
    L.new(p2v.outputs["Volume"], setm.inputs["Geometry"])
    L.new(setm.outputs["Geometry"], gout.inputs[0])
    for i, node in enumerate((gin, rad, m2p, p2v, setm, gout)):
        node.location = (i * 220, 0)
    return ng


def smoke_material():
    mat = bpy.data.materials.get(SMOKE_MAT)
    if mat:
        return mat
    mat = bpy.data.materials.new(SMOKE_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.name = "Smoke"
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.name = "Rise"
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.name = "Billow"
    noise.noise_dimensions = "4D"
    noise.inputs["Detail"].default_value = 6
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.name = "Erosion"
    ramp.clamp = True
    dens_attr = nt.nodes.new("ShaderNodeAttribute")
    dens_attr.attribute_name = "density"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.name = "Density"
    L = nt.links
    L.new(coord.outputs["Object"], mapping.inputs["Vector"])
    L.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    L.new(noise.outputs["Fac"], ramp.inputs["Value"])
    L.new(ramp.outputs["Result"], mul.inputs[0])
    L.new(dens_attr.outputs["Fac"], mul.inputs[1])
    L.new(mul.outputs[0], mul2.inputs[0])
    L.new(mul2.outputs[0], vol.inputs["Density"])
    L.new(vol.outputs[0], out.inputs["Volume"])
    for i, node in enumerate((coord, mapping, noise, ramp, dens_attr, mul, mul2, vol, out)):
        node.location = (i * 200, 0)
    return mat


def build_smoke(scene, plumes):
    coll = bpy.data.collections.new("Smoke")
    scene.collection.children.link(coll)
    for i, pl in enumerate(plumes):
        ob = bpy.data.objects.new(f"Smoke{i:03d}", bpy.data.meshes.new(f"Smoke{i:03d}"))
        ob["gtb_points"] = json.dumps(pl["points"])
        mod = ob.modifiers.new("SmokeVolume", "NODES")
        mod.node_group = smoke_node_group()
        coll.objects.link(ob)


def _frame_driver(sock, expr, index=-1):
    sock.driver_remove("default_value", index)
    fc = sock.driver_add("default_value", index)
    fc.driver.expression = expr


def apply_smoke(scene, cfg):
    coll = bpy.data.collections.get("Smoke")
    if not coll:
        return
    coll.hide_render = not cfg["enabled"]
    # Plumes requested in look.json (chimneys etc.): single-point sources.
    for ob in [o for o in coll.objects if o.get("gtb_emitter")]:
        bpy.data.objects.remove(ob)
    for i, (x, y, z, r) in enumerate(cfg["emitters"]):
        ob = bpy.data.objects.new(f"SmokeEmitter{i:02d}", bpy.data.meshes.new(f"SmokeEmitter{i:02d}"))
        ob["gtb_points"] = json.dumps([[x, y, z, r], [x, y, z + r, r]])
        ob["gtb_emitter"] = True
        ob.modifiers.new("SmokeVolume", "NODES").node_group = smoke_node_group()
        coll.objects.link(ob)
    for ob in coll.objects:
        pts = np.array(json.loads(ob["gtb_points"]), dtype=np.float64)
        base = pts[:, :3]
        r = pts[:, 3] * cfg["spread"]
        # Continue the plume upward from its highest puff, leaning like the game's.
        top = base[np.argmax(base[:, 2])]
        low = base[np.argmin(base[:, 2])]
        lean = top - low
        lean[2] = max(lean[2], 1e-3)
        lean = lean / lean[2]  # horizontal drift per unit of rise
        steps = max(2, int(cfg["rise"] / max(r.mean(), 0.5)))
        h = np.linspace(0, cfg["rise"], steps + 1)[1:]
        col = top + (lean * np.array([1, 1, 0]))[None, :] * h[:, None] + np.array([0, 0, 1]) * h[:, None]
        col_r = r.max() * (1 + cfg["grow"] * h)
        verts = np.vstack([base, col])
        radii = np.concatenate([r, col_r])
        me = ob.data
        me.clear_geometry()
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", (verts - low).astype(np.float32).ravel())
        attr = me.attributes.get("radius") or me.attributes.new("radius", "FLOAT", "POINT")
        attr.data.foreach_set("value", radii.astype(np.float32))
        me.update()
        ob.location = low
    smoke_node_group().nodes["ToVolume"].inputs["Voxel Size"].default_value = cfg["voxel_size"]
    nt = smoke_material().node_tree
    vol = nt.nodes["Smoke"]
    vol.inputs["Color"].default_value = (*cfg["color"], 1)
    vol.inputs["Anisotropy"].default_value = cfg["anisotropy"]
    nt.nodes["Density"].inputs[1].default_value = cfg["density"]
    nt.nodes["Billow"].inputs["Scale"].default_value = cfg["noise_scale"]
    nt.nodes["Billow"].inputs["Distortion"].default_value = cfg["distortion"]
    ramp = nt.nodes["Erosion"]
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = cfg["erosion"]
    # Animate: the noise evolves (W) and scrolls upward with the frame number.
    _frame_driver(nt.nodes["Billow"].inputs["W"], "frame*%.5f" % (0.02 * cfg["speed"]))
    _frame_driver(nt.nodes["Rise"].inputs["Location"], "-frame*%.5f" % (0.08 * cfg["speed"]), 2)
    scene.eevee.use_volumetric_shadows = True


SNOW_GN, SNOW_MAT = "GTB_Snowfall", "GTB_Snowflake"


def _sock(node, name, kind, out=False):
    """Socket by name and type (several nodes have one socket per data type)."""
    return next(s for s in (node.outputs if out else node.inputs) if s.name == name and s.type == kind)


def snow_material():
    """Soft round flake: radial falloff on a camera-facing quad, lit by the
    scene (warm near lamps) plus a little self-glow so it reads against dark."""
    mat = bpy.data.materials.get(SNOW_MAT)
    if mat:
        return mat
    mat = bpy.data.materials.new(SNOW_MAT)
    mat.use_nodes = True
    mat.surface_render_method = "BLENDED"
    mat.use_backface_culling = False
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new("ShaderNodeOutputMaterial")
    uv = N.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    dist = N.new("ShaderNodeVectorMath")
    dist.operation = "DISTANCE"
    L.new(uv.outputs["UV"], dist.inputs[0])
    dist.inputs[1].default_value = (0.5, 0.5, 0.0)
    fall = N.new("ShaderNodeMapRange")  # 1 at the centre -> 0 at the rim
    fall.clamp = True
    L.new(dist.outputs["Value"], fall.inputs["Value"])
    fall.inputs["From Min"].default_value, fall.inputs["From Max"].default_value = 0.5, 0.0
    soft = math_node(nt, "POWER", fall.outputs["Result"], 1.6)
    opacity = N.new("ShaderNodeValue")
    opacity.name = "Opacity"
    alpha = math_node(nt, "MULTIPLY", soft, opacity.outputs[0])
    diff = N.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (0.95, 0.97, 1.0, 1)
    glow = N.new("ShaderNodeEmission")
    glow.name = "Glow"
    glow.inputs["Color"].default_value = (0.75, 0.83, 1.0, 1)
    add = N.new("ShaderNodeAddShader")
    L.new(diff.outputs[0], add.inputs[0])
    L.new(glow.outputs[0], add.inputs[1])
    mix = N.new("ShaderNodeMixShader")
    L.new(alpha, mix.inputs["Fac"])
    L.new(N.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    L.new(add.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def snow_node_group():
    ng = bpy.data.node_groups.get(SNOW_GN)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(SNOW_GN, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for name, typ in (("Count", "NodeSocketInt"), ("Box Min", "NodeSocketVector"), ("Box Max", "NodeSocketVector"),
                      ("Size", "NodeSocketFloat"), ("Fall Speed", "NodeSocketFloat"),
                      ("Wind", "NodeSocketVector"), ("Flutter", "NodeSocketFloat"), ("Seed", "NodeSocketInt"),
                      ("Follow Camera", "NodeSocketBool"), ("Extent", "NodeSocketVector")):
        ng.interface.new_socket(name, in_out="INPUT", socket_type=typ)
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    def math(op, a, b=None, c=None):
        m = N.new("ShaderNodeMath")
        m.operation = op
        for i, x in enumerate((a, b, c)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = x
            else:
                L.new(x, m.inputs[i])
        return m.outputs[0]

    def vmath(op, a, b):
        m = N.new("ShaderNodeVectorMath")
        m.operation = op
        L.new(a, m.inputs[0])
        L.new(b, m.inputs[1])
        return m.outputs[0]

    def switch(a_true, a_false):
        sw = N.new("GeometryNodeSwitch")
        sw.input_type = "VECTOR"
        L.new(gin.outputs["Follow Camera"], sw.inputs["Switch"])
        L.new(a_true, sw.inputs["True"])
        L.new(a_false, sw.inputs["False"])
        return sw.outputs[0]

    cam = N.new("GeometryNodeObjectInfo")
    L.new(N.new("GeometryNodeInputActiveCamera").outputs[0], cam.inputs["Object"])
    neg_ext = N.new("ShaderNodeVectorMath")
    neg_ext.operation = "SCALE"
    L.new(gin.outputs["Extent"], neg_ext.inputs[0])
    neg_ext.inputs["Scale"].default_value = -1.0
    box_lo = switch(vmath("ADD", cam.outputs["Location"], neg_ext.outputs[0]), gin.outputs["Box Min"])
    box_hi = switch(vmath("ADD", cam.outputs["Location"], gin.outputs["Extent"]), gin.outputs["Box Max"])
    # Start positions must not depend on the camera, or flakes would ride along with it.
    start_lo = switch(neg_ext.outputs[0], gin.outputs["Box Min"])
    start_hi = switch(gin.outputs["Extent"], gin.outputs["Box Max"])

    # Random start position and random size per flake.
    rnd = N.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT_VECTOR"
    L.new(start_lo, _sock(rnd, "Min", "VECTOR"))
    L.new(start_hi, _sock(rnd, "Max", "VECTOR"))
    L.new(gin.outputs["Seed"], _sock(rnd, "Seed", "INT"))
    rsize = N.new("FunctionNodeRandomValue")
    rsize.data_type = "FLOAT"
    _sock(rsize, "Min", "VALUE").default_value = 0.5
    _sock(rsize, "Max", "VALUE").default_value = 1.5
    pts = N.new("GeometryNodePoints")
    L.new(gin.outputs["Count"], pts.inputs["Count"])
    L.new(_sock(rnd, "Value", "VECTOR", out=True), pts.inputs["Position"])
    L.new(math("MULTIPLY", _sock(rsize, "Value", "VALUE", out=True), gin.outputs["Size"]), pts.inputs["Radius"])

    # Move: fall + wind + flutter with scene time, wrapped inside the box.
    t = N.new("GeometryNodeInputSceneTime").outputs["Seconds"]
    pos = N.new("GeometryNodeInputPosition").outputs["Position"]
    sep_p = N.new("ShaderNodeSeparateXYZ")
    L.new(pos, sep_p.inputs[0])
    sep_w = N.new("ShaderNodeSeparateXYZ")
    L.new(gin.outputs["Wind"], sep_w.inputs[0])
    sep_lo, sep_hi = N.new("ShaderNodeSeparateXYZ"), N.new("ShaderNodeSeparateXYZ")
    L.new(box_lo, sep_lo.inputs[0])
    L.new(box_hi, sep_hi.inputs[0])
    noise = N.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 0.15
    L.new(pos, noise.inputs["Vector"])
    L.new(math("MULTIPLY", t, 0.3), noise.inputs["W"])
    nsep = N.new("FunctionNodeSeparateColor")
    L.new(noise.outputs["Color"], nsep.inputs[0])
    comb = N.new("ShaderNodeCombineXYZ")
    for axis, wind_ax, fl in ((0, "X", "Red"), (1, "Y", "Green")):
        drift = math("MULTIPLY", sep_w.outputs[wind_ax], t)
        wob = math("MULTIPLY", math("SUBTRACT", nsep.outputs[fl], 0.5), gin.outputs["Flutter"])
        v = math("ADD", math("ADD", sep_p.outputs["XYZ"[axis]], drift), wob)
        L.new(math("WRAP", v, sep_hi.outputs["XYZ"[axis]], sep_lo.outputs["XYZ"[axis]]), comb.inputs[axis])
    fall = math("MULTIPLY", gin.outputs["Fall Speed"], t)
    L.new(math("WRAP", math("SUBTRACT", sep_p.outputs["Z"], fall), sep_hi.outputs["Z"], sep_lo.outputs["Z"]),
          comb.inputs[2])
    setpos = N.new("GeometryNodeSetPosition")
    L.new(pts.outputs["Points"], setpos.inputs["Geometry"])
    L.new(comb.outputs[0], setpos.inputs["Position"])

    # A soft quad per flake, turned to face the render camera.
    grid = N.new("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = grid.inputs["Size Y"].default_value = 1.0
    grid.inputs["Vertices X"].default_value = grid.inputs["Vertices Y"].default_value = 2
    store = N.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT2", "CORNER"
    store.inputs["Name"].default_value = "UVMap"
    L.new(grid.outputs["Mesh"], store.inputs["Geometry"])
    L.new(grid.outputs["UV Map"], _sock(store, "Value", "VECTOR"))
    face = N.new("FunctionNodeAlignRotationToVector")
    face.axis = "Z"
    L.new(vmath("SUBTRACT", cam.outputs["Location"], N.new("GeometryNodeInputPosition").outputs[0]),
          face.inputs["Vector"])
    inst = N.new("GeometryNodeInstanceOnPoints")
    L.new(setpos.outputs["Geometry"], inst.inputs["Points"])
    L.new(store.outputs["Geometry"], inst.inputs["Instance"])
    L.new(face.outputs["Rotation"], inst.inputs["Rotation"])
    L.new(math("MULTIPLY", N.new("GeometryNodeInputRadius").outputs[0], 2.0), inst.inputs["Scale"])
    setm = N.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = snow_material()
    L.new(inst.outputs["Instances"], setm.inputs["Geometry"])
    L.new(setm.outputs["Geometry"], gout.inputs[0])
    return ng


def set_modifier_input(mod, identifier, value):
    """Geometry-nodes modifier input. Blender 5 exposes them as
    mod.properties.inputs.<identifier>.value; older versions as ID properties."""
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, identifier).value = value
    else:
        mod[identifier] = value


def build_snow(scene, box):
    coll = bpy.data.collections.new("Snow")
    scene.collection.children.link(coll)
    ob = bpy.data.objects.new("Snowfall", bpy.data.meshes.new("Snowfall"))
    ob["gtb_box"] = json.dumps(box)
    mod = ob.modifiers.new("Snowfall", "NODES")
    mod.node_group = snow_node_group()
    coll.objects.link(ob)


def apply_snow(scene, cfg):
    ob = bpy.data.objects.get("Snowfall")
    if not ob:
        return
    bpy.data.collections["Snow"].hide_render = not cfg["enabled"]
    lo, hi = (np.array(v, dtype=float) for v in json.loads(ob["gtb_box"]))
    c, half = (lo + hi) / 2, (hi - lo) / 2 * cfg["box_scale"]
    mod = ob.modifiers["Snowfall"]
    ng = mod.node_group
    values = {"Count": int(cfg["count"]), "Box Min": [float(x) for x in c - half],
              "Box Max": [float(x) for x in c + half], "Size": float(cfg["size"]),
              "Fall Speed": float(cfg["fall_speed"]), "Wind": [float(cfg["wind"][0]), float(cfg["wind"][1]), 0.0],
              "Flutter": float(cfg["flutter"]), "Seed": int(cfg["seed"]),
              "Follow Camera": bool(cfg["follow_camera"]), "Extent": [float(x) for x in cfg["extent"]]}
    for item in ng.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name in values:
            set_modifier_input(mod, item.identifier, values[item.name])
    ob.update_tag()
    nodes = snow_material().node_tree.nodes
    nodes["Glow"].inputs["Strength"].default_value = 0.25 * cfg["brightness"]
    nodes["Opacity"].outputs[0].default_value = cfg["opacity"]


BILLBOARD_GN = "GTB_Billboard"


def billboard_node_group():
    ng = bpy.data.node_groups.get(BILLBOARD_GN)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(BILLBOARD_GN, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Billboard", in_out="INPUT", socket_type="NodeSocketBool").default_value = True
    ng.interface.new_socket("Rise", in_out="INPUT", socket_type="NodeSocketFloat")
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    def named(name):
        a = N.new("GeometryNodeInputNamedAttribute")
        a.data_type = "FLOAT_VECTOR"
        a.inputs["Name"].default_value = name
        return a.outputs["Attribute"]

    def vmath(op, a, b=None, scale=None):
        m = N.new("ShaderNodeVectorMath")
        m.operation = op
        L.new(a, m.inputs[0])
        if b is not None:
            L.new(b, m.inputs[1])
        if scale is not None:
            L.new(scale, m.inputs["Scale"])
        return m.outputs["Vector"]

    centre, corner = named("p_center"), named("p_corner")
    info = N.new("GeometryNodeObjectInfo")
    L.new(N.new("GeometryNodeInputActiveCamera").outputs[0], info.inputs["Object"])
    axes = []
    for vec in ((1, 0, 0), (0, 1, 0)):
        rv = N.new("FunctionNodeRotateVector")
        rv.inputs["Vector"].default_value = vec
        L.new(info.outputs["Rotation"], rv.inputs["Rotation"])
        axes.append(rv.outputs["Vector"])
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(corner, sep.inputs[0])
    faced = vmath("ADD", centre, vmath("ADD", vmath("SCALE", axes[0], scale=sep.outputs["X"]),
                                        vmath("SCALE", axes[1], scale=sep.outputs["Y"])))
    sw = N.new("GeometryNodeSwitch")
    sw.input_type = "VECTOR"
    L.new(gin.outputs["Billboard"], sw.inputs["Switch"])
    L.new(N.new("GeometryNodeInputPosition").outputs[0], sw.inputs["False"])
    L.new(faced, sw.inputs["True"])
    # Gentle rise over time (smoke drifting up).
    t = N.new("GeometryNodeInputSceneTime").outputs["Seconds"]
    lift = N.new("ShaderNodeCombineXYZ")
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(t, mul.inputs[0])
    L.new(gin.outputs["Rise"], mul.inputs[1])
    L.new(mul.outputs[0], lift.inputs["Z"])
    setp = N.new("GeometryNodeSetPosition")
    L.new(gin.outputs["Geometry"], setp.inputs["Geometry"])
    L.new(sw.outputs[0], setp.inputs["Position"])
    L.new(lift.outputs[0], setp.inputs["Offset"])
    L.new(setp.outputs["Geometry"], gout.inputs[0])
    return ng


def build_sprite_material(sprite, entry, tex_dir, profile):
    """Smoke: flipbook alpha x particle opacity, lit through the sheet's RGB
    normal map with some translucency. Fire: emissive colour."""
    mat = bpy.data.materials.new(f"Sprite_{sprite['kind']}_{sprite['atlas']}")
    mat["gtb_sprite"] = sprite["kind"]
    mat.use_nodes = True
    mat.surface_render_method = "BLENDED"  # smooth soft alpha (dithered looks grainy)
    mat.use_backface_culling = False
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new("ShaderNodeOutputMaterial")
    uv = N.new("ShaderNodeUVMap")
    uv.uv_map = "UV0"
    tex = N.new("ShaderNodeTexImage")
    fire = sprite["kind"] == "fire"
    tex.image = load_image(tex_dir / entry["file"], not fire)
    tex.image.alpha_mode = "CHANNEL_PACKED"
    L.new(uv.outputs["UV"], tex.inputs["Vector"])
    vcol = N.new("ShaderNodeVertexColor")
    vcol.layer_name = "Col"
    opacity = N.new("ShaderNodeValue")
    opacity.name = "Opacity"
    a1 = math_node(nt, "MULTIPLY", tex.outputs["Alpha"], vcol.outputs["Alpha"])
    alpha = math_node(nt, "MINIMUM", math_node(nt, "MULTIPLY", a1, opacity.outputs[0]), 1.0)
    transparent = N.new("ShaderNodeBsdfTransparent")
    mix = N.new("ShaderNodeMixShader")
    L.new(alpha, mix.inputs["Fac"])
    L.new(transparent.outputs[0], mix.inputs[1])
    if fire:
        em = N.new("ShaderNodeEmission")
        em.name = "Fire"
        L.new(tex.outputs["Color"], em.inputs["Color"])
        L.new(em.outputs[0], mix.inputs[2])
    else:
        col = N.new("ShaderNodeRGB")
        col.name = "SmokeColor"
        sep = N.new("ShaderNodeSeparateColor")
        L.new(tex.outputs["Color"], sep.inputs[0])
        # Sheet RGB is a tangent-space normal (R, G; Z rebuilt).
        xs = math_node(nt, "MULTIPLY_ADD", sep.outputs["Red"], 2.0, c=-1.0)
        ys = math_node(nt, "MULTIPLY_ADD", sep.outputs["Green"], 2.0, c=-1.0)
        z = math_node(nt, "SQRT", math_node(nt, "MAXIMUM", math_node(
            nt, "SUBTRACT", math_node(nt, "SUBTRACT", 1.0, math_node(nt, "MULTIPLY", xs, xs)),
            math_node(nt, "MULTIPLY", ys, ys)), 0.0))
        comb = N.new("ShaderNodeCombineColor")
        L.new(sep.outputs["Red"], comb.inputs[0])
        L.new(sep.outputs["Green"], comb.inputs[1])
        L.new(math_node(nt, "MULTIPLY_ADD", z, 0.5, c=0.5), comb.inputs[2])
        nm = N.new("ShaderNodeNormalMap")
        nm.name = "SpriteNormal"
        nm.uv_map = "UV0"
        L.new(comb.outputs[0], nm.inputs["Color"])
        diff, trans = N.new("ShaderNodeBsdfDiffuse"), N.new("ShaderNodeBsdfTranslucent")
        for sh in (diff, trans):
            L.new(col.outputs[0], sh.inputs["Color"])
            L.new(nm.outputs[0], sh.inputs["Normal"])
        blend = N.new("ShaderNodeMixShader")
        blend.name = "Translucency"
        L.new(diff.outputs[0], blend.inputs[1])
        L.new(trans.outputs[0], blend.inputs[2])
        # Games light smoke with a flat ambient term too; without it smoke
        # against the sky reads as black.
        amb = N.new("ShaderNodeEmission")
        amb.name = "SmokeAmbient"
        L.new(col.outputs[0], amb.inputs["Color"])
        add = N.new("ShaderNodeAddShader")
        L.new(blend.outputs[0], add.inputs[0])
        L.new(amb.outputs[0], add.inputs[1])
        L.new(add.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def apply_particles(scene, cfg):
    coll = bpy.data.collections.get("Particles")
    if not coll:
        return
    coll.hide_render = not cfg["enabled"]
    for ob in coll.objects:
        mod = ob.modifiers.get("Billboard")
        if mod:
            for item in mod.node_group.interface.items_tree:
                if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name in ("Billboard", "Rise"):
                    set_modifier_input(mod, item.identifier,
                                       bool(cfg["billboard"]) if item.name == "Billboard" else float(cfg["rise"]))
    for mat in bpy.data.materials:
        kind = mat.get("gtb_sprite")
        if not kind:
            continue
        N = mat.node_tree.nodes
        N["Opacity"].outputs[0].default_value = cfg["opacity"]
        if kind == "fire":
            N["Fire"].inputs["Strength"].default_value = cfg["fire_strength"]
        else:
            N["SmokeColor"].outputs[0].default_value = (*srgb_to_linear(cfg["smoke_color"]), 1)
            N["Translucency"].inputs["Fac"].default_value = cfg["translucency"]
            N["SmokeAmbient"].inputs["Strength"].default_value = cfg["ambient"]
            N["SpriteNormal"].inputs["Strength"].default_value = cfg["normal_strength"]


def build_game_lights(scene, lights):
    coll = bpy.data.collections.new("Game Lights")
    scene.collection.children.link(coll)
    for i, L in enumerate(lights):
        if L.get("off"):    # far below the scene (process.py): another pass's light
            continue
        data = bpy.data.lights.new(f"GL{i:04d}", "POINT")
        ob = bpy.data.objects.new(f"GL{i:04d}_{L['kind']}", data)
        ob.location = L["location"]
        ob["gtb_range"] = L["radius"]
        ob["gtb_kind"] = L["kind"]
        coll.objects.link(ob)


def apply_game_lights(cfg):
    coll = bpy.data.collections.get("Game Lights")
    if not coll:
        return
    coll.hide_render = not cfg["enabled"]
    for ob in coll.objects:
        r = ob.get("gtb_range", 1.0)
        scale = cfg["hemi_scale"] if ob.get("gtb_kind") == "hemi" else 1.0
        ob.data.energy = cfg["power_per_range2"] * r ** cfg["range_exponent"] * scale
        ob.data.color = cfg["color"]
        ob.data.shadow_soft_size = cfg["soft_size"]
        ob.data.use_shadow = cfg["shadows"]
        ob.data.use_custom_distance = True
        ob.data.cutoff_distance = r * cfg["range_scale"]


def apply_compositor(scene, look):
    ng = scene.compositing_node_group
    if ng is None:
        ng = bpy.data.node_groups.new("GTB_Compositor", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = ng
    ng.nodes.clear()
    scene.view_layers[0].use_pass_z = True
    rl = ng.nodes.new("CompositorNodeRLayers")
    out = ng.nodes.new("NodeGroupOutput")
    image = rl.outputs["Image"]
    fog = look["fog"]
    if fog["density"] > 0:
        image = _depth_fog(ng, rl, image, fog)
    g = look["grade"]
    cb = ng.nodes.new("CompositorNodeColorBalance")
    cb.inputs["Factor"].default_value = 1.0
    _set_balance(cb, g)
    hs = ng.nodes.new("CompositorNodeHueSat")
    hs.inputs["Saturation"].default_value = g["saturation"]
    bc = ng.nodes.new("CompositorNodeBrightContrast")
    bc.inputs["Bright"].default_value = g["brightness"]
    bc.inputs["Contrast"].default_value = g["contrast"]
    chain = [image, cb, hs, bc]
    b = look["bloom"]
    if b["strength"] > 0:
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Threshold"].default_value = b["threshold"]
        gl.inputs["Strength"].default_value = b["strength"]
        gl.inputs["Size"].default_value = b["size"]
        chain.append(gl)
    src = chain[0]
    for node in chain[1:]:
        ng.links.new(src, node.inputs["Image"])
        src = node.outputs["Image"]
    ng.links.new(src, out.inputs[0])
    for i, n in enumerate(ng.nodes):
        n.location = (i * 220, 0)


def _depth_fog(ng, rl, image, fog):
    def math(op, a, b=None):
        m = ng.nodes.new("ShaderNodeMath")
        m.operation = op
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = x
            else:
                ng.links.new(x, m.inputs[i])
        return m.outputs[0]

    d = math("MAXIMUM", math("SUBTRACT", rl.outputs["Depth"], fog["start"]), 0.0)
    opacity = math("SUBTRACT", 1.0, math("EXPONENT", math("MULTIPLY", d, -fog["density"])))
    opacity = math("MINIMUM", opacity, fog["max_opacity"])
    mix = ng.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    a = next(i for i in mix.inputs if i.name == "A" and i.type == "RGBA")
    b = next(i for i in mix.inputs if i.name == "B" and i.type == "RGBA")
    ng.links.new(opacity, mix.inputs["Factor"])
    ng.links.new(image, a)
    b.default_value = (*[c * fog["strength"] for c in fog["color"]], 1)
    return next(o for o in mix.outputs if o.type == "RGBA")


def _set_balance(cb, g):
    """Lift/gamma/gain; socket names changed across versions, so match loosely."""
    for key in ("lift", "gamma", "gain"):
        for inp in cb.inputs:
            if inp.name.lower().startswith(key) and inp.type == "RGBA":
                inp.default_value = (*g[key], 1)
                break


# --------------------------------------------------------------------------- entry

def build(capture: Path):
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    profile = manifest.get("profile", {})
    scene = clear_scene()
    tex_dir = capture / "textures"
    textures = manifest["textures"]

    geo = bpy.data.collections.new("Geometry")
    scene.collection.children.link(geo)
    # Draws without depth writes (sky, particles, decals, glass) and profile/
    # heuristic "effect" draws (blended snow layers, smoke, fog cards). The game
    # blends these; drawn opaque they would bury the scene, so they start hidden.
    overlay = bpy.data.collections.new("Effects (hidden)")
    scene.collection.children.link(overlay)
    overlay.hide_render = True
    bpy.context.view_layer.layer_collection.children[overlay.name].hide_viewport = True
    sprites = bpy.data.collections.new("Particles")
    scene.collection.children.link(sprites)
    materials = {}
    for m in manifest["meshes"]:
        data = dict(np.load(capture / m["file"]))
        if len(data["indices"]) == 0:
            continue
        me = build_mesh(m["name"], data)
        if m.get("category") == "sprite" and m.get("sprite"):
            key = "sprite|" + m["sprite"]["kind"] + "|" + m["sprite"]["atlas"]
            if key not in materials:
                materials[key] = build_sprite_material(m["sprite"], textures[m["sprite"]["atlas"]], tex_dir, profile)
            me.materials.append(materials[key])
            ob = bpy.data.objects.new(m["name"], me)
            ob.visible_shadow = False
            ob.modifiers.new("Billboard", "NODES").node_group = billboard_node_group()
            sprites.objects.link(ob)
            continue
        tex_entries = [textures[t] for t in m.get("textures", {}).values() if t in textures]
        if tex_entries and all(e.get("role") == "shared" for e in tex_entries) and "uv0" in data:
            # Only the engine's shared snow texture bound: snow drifts / snow caps.
            key = "snowmesh|" + tex_entries[0]["file"]
            if key not in materials:
                materials[key] = build_snowmesh_material(tex_entries[0], tex_dir, profile)
            me.materials.append(materials[key])
            ob = bpy.data.objects.new(m["name"], me)
            ob["gtb_source"] = m.get("source", "")
            geo.objects.link(ob)
            continue
        slots = assign_slots(m.get("textures", {}), textures, profile)
        has_uv = "uv0" in data
        key = m.get("material_key") or json.dumps(
            {r: e["file"] for r, e in sorted(slots.items())}, sort_keys=True) + ("" if has_uv else "|nouv")
        if key not in materials:
            materials[key] = build_material(f"M{len(materials):03d}", slots, tex_dir, profile, has_uv)
        me.materials.append(materials[key])
        ob = bpy.data.objects.new(m["name"], me)
        ob["gtb_source"] = m.get("source", "")
        ob["gtb_shaders"] = ",".join(m.get("shaders", []))
        is_surface = m.get("depth_write", True) and m.get("category", "surface") == "surface"
        (geo if is_surface else overlay).objects.link(ob)

    build_game_lights(scene, manifest.get("lights", []))
    build_smoke(scene, manifest.get("smoke", []))
    if manifest.get("snow_box"):
        build_snow(scene, manifest["snow_box"])
    shot = capture / manifest.get("screenshot", "screenshot.png")
    setup_camera(scene, manifest.get("camera", {}), manifest["resolution"], shot)
    apply_look(scene, load_look(capture / "look.json"))
    print(f"[gtb] built {len(geo.objects)} + {len(overlay.objects)} hidden effect objects, "
          f"{len(sprites.objects)} particle sprites, "
          f"{len(materials)} materials")
    return scene
