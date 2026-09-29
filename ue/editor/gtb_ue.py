"""Runs inside Unreal: UnrealEditor-Cmd <project> -run=pythonscript -script="gtb_ue.py <job.json>"

job.json (written by ue/pipeline.py):
  action   "build" (import everything, make the level) or "look" (re-apply look only)
  plan     captures/<name>/unreal/plan.json        (ue/export.py)
  look     captures/<name>/unreal/look_ue.json     (ue/look.py)
  shared   {default_color, default_linear, white_cube}: source files for shared assets
  log      progress/result log (the commandlet's stdout is not reliable for Python output)

Content: /Game/GTB/Shared (masters, time MPC, defaults) and /Game/GTB/<capture>/
(Textures, Meshes, Materials, the level <capture>, Render/ sequences + MRQ configs).
"""
import json
import os
import sys
import time
import traceback

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gtb_materials  # noqa: E402

AT = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary
SHARED = "/Game/GTB/Shared"

job = json.load(open(sys.argv[1], encoding="utf-8"))
LOG = job["log"]
_t0 = time.time()
_log_lines = []
WARNINGS = []


def log(msg):
    line = f"[{time.time() - _t0:7.1f}s] {msg}"
    _log_lines.append(line)
    unreal.log(f"[gtb] {msg}")
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


def warn(msg):
    WARNINGS.append(msg)
    log(f"WARNING {msg}")


def setp(obj, name, value):
    """set_editor_property that reports instead of aborting (API names drift between versions)."""
    try:
        obj.set_editor_property(name, value)
        return True
    except Exception as e:  # noqa: BLE001
        warn(f"{type(obj).__name__}.{name}: {e}")
        return False


def lc(v, a=1.0):
    v = list(v) + [a] * (4 - len(v))
    return unreal.LinearColor(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def vec(v):
    return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))


# --------------------------------------------------------------------------- assets

def load_or_create(name, path, cls, factory):
    full = f"{path}/{name}"
    if EAL.does_asset_exist(full):
        a = EAL.load_asset(full)
        if isinstance(a, cls):
            return a
        EAL.delete_asset(full)
    return AT.create_asset(name, path, cls, factory)


def import_files(pairs, dest, options=None):
    """pairs: [(source file, asset name or None)]. Returns imported object paths."""
    tasks = []
    for src, name in pairs:
        t = unreal.AssetImportTask()
        props = {"filename": src, "destination_path": dest, "automated": True, "save": False,
                 "replace_existing": True}
        if name:
            props["destination_name"] = name
        t.set_editor_properties(props)
        if options is not None:
            t.set_editor_property("options", options)
        tasks.append(t)
    AT.import_asset_tasks(tasks)
    paths = []
    for t in tasks:
        paths += list(t.get_editor_property("imported_object_paths"))
    return paths


def configure_texture(tex, kind):
    TC = unreal.TextureCompressionSettings
    if kind == "lut":
        props = {"srgb": False, "compression_settings": TC.TC_HDR,
                 "mip_gen_settings": unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS,
                 "filter": unreal.TextureFilter.TF_BILINEAR, "address_x": unreal.TextureAddress.TA_CLAMP,
                 "address_y": unreal.TextureAddress.TA_CLAMP,
                 "lod_group": unreal.TextureGroup.TEXTUREGROUP_COLOR_LOOKUP_TABLE, "never_stream": True}
    elif kind == "cube":
        props = {"srgb": False, "compression_settings": TC.TC_HDR}
    else:
        props = {"srgb": kind == "albedo", "compression_settings": TC.TC_BC7}
    try:
        tex.set_editor_properties(props)
    except Exception as e:  # noqa: BLE001
        warn(f"texture {tex.get_name()}: {e}")


def import_texture(src, dest, name, kind):
    paths = import_files([(src, name)], dest)
    tex = EAL.load_asset(paths[0]) if paths else None
    if tex is None:
        raise RuntimeError(f"texture import failed: {src}")
    configure_texture(tex, kind)
    return tex


def mesh_import_options():
    p = unreal.InterchangeGenericAssetsPipeline()
    c = p.get_editor_property("common_meshes_properties")
    for k, v in {"recompute_normals": False, "recompute_tangents": True, "use_mikk_t_space": True,
                 "use_full_precision_u_vs": True, "remove_degenerates": False, "import_lods": False,
                 "bake_meshes": True, "force_all_mesh_as_type": unreal.InterchangeForceMeshType.IFMT_STATIC_MESH}.items():
        setp(c, k, v)
    m = p.get_editor_property("mesh_pipeline")
    for k, v in {"import_static_meshes": True, "import_skeletal_meshes": False, "generate_lightmap_u_vs": False,
                 "build_nanite": False, "collision": False,
                 "generate_distance_field_as_if_two_sided": True}.items():
        setp(m, k, v)
    mp = p.get_editor_property("material_pipeline")
    setp(mp, "import_materials", False)
    setp(mp.get_editor_property("texture_pipeline"), "import_textures", False)
    stack = unreal.InterchangePipelineStackOverride()
    stack.add_pipeline(p)
    return stack


def ensure_shared(shared_src):
    """Default textures, the white sky cubemap, the time MPC and the masters."""
    defaults = {}
    for key, kind in (("default_color", "albedo"), ("default_linear", "linear"), ("white_cube", "cube"),
                      ("default_lut", "lut")):
        name = "T_GTB_" + "".join(w.title() for w in key.split("_"))
        full = f"{SHARED}/{name}"
        tex = EAL.load_asset(full) if EAL.does_asset_exist(full) else import_texture(shared_src[key], SHARED, name, kind)
        defaults[key] = tex
    if not isinstance(defaults["white_cube"], unreal.TextureCube):
        warn(f"white_cube imported as {type(defaults['white_cube']).__name__}, not a cubemap")
    mpc = load_or_create("MPC_GTB_Time", SHARED, unreal.MaterialParameterCollection,
                         unreal.MaterialParameterCollectionFactoryNew())
    names = [p.get_editor_property("parameter_name") for p in mpc.get_editor_property("scalar_parameters")]
    if names != ["SceneTime", "EngineTimeWeight"]:
        params = []
        for n, dv in (("SceneTime", 0.0), ("EngineTimeWeight", 1.0)):
            sp = unreal.CollectionScalarParameter()
            sp.set_editor_property("parameter_name", n)
            sp.set_editor_property("default_value", dv)
            params.append(sp)
        mpc.set_editor_property("scalar_parameters", params)
        EAL.save_loaded_asset(mpc)
    masters = gtb_materials.ensure_masters(
        SHARED, {"color": defaults["default_color"], "linear": defaults["default_linear"],
                 "lut": defaults["default_lut"]}, mpc, log)
    return defaults, mpc, masters


# --------------------------------------------------------------------------- materials

def material_instance(name, path, parent):
    mi = load_or_create(name, path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    unreal.MaterialEditingLibrary.set_material_instance_parent(mi, parent)
    return mi


def set_params(mi, scalars=None, vectors=None, textures=None):
    MEL = unreal.MaterialEditingLibrary
    for k, v in (scalars or {}).items():
        MEL.set_material_instance_scalar_parameter_value(mi, k, float(v))
    for k, v in (vectors or {}).items():
        MEL.set_material_instance_vector_parameter_value(mi, k, lc(v, 0.0 if len(v) < 4 else v[3]))
    for k, v in (textures or {}).items():
        MEL.set_material_instance_texture_parameter_value(mi, k, v)


def build_materials(plan, cap, masters, textures):
    """Per capture: one 'look' instance per master (look.json values live there),
    and the per-material instances as its children."""
    mdir = f"{cap}/Materials"
    look_mis = {}
    for key in ("surface", "surface_masked", "snowdrift", "smoke", "fire", "snowfall", "sky", "tonemap"):
        name = {"snowfall": "MI_Snowfall", "sky": "MI_Sky", "tonemap": "MI_Tonemap"}.get(
            key, "MI_Look_" + "".join(w.title() for w in key.split("_")))
        look_mis[key] = material_instance(name, mdir, masters[key])
    mis = {}
    for m in plan["materials"]:
        mi = material_instance(m["name"], mdir, look_mis[m["parent"]])
        tex = {k: textures[spec["file"]] for k, spec in m["textures"].items()}
        set_params(mi, m.get("scalars"), m.get("vectors"), tex)
        mis[m["name"]] = mi
    log(f"materials: {len(mis)} instances")
    return look_mis, mis


# --------------------------------------------------------------------------- level

def open_level(cap_name):
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    path = f"/Game/GTB/{cap_name}/{cap_name}"
    if EAL.does_asset_exist(path):
        les.load_level(path)
    elif not les.new_level(path, False):
        raise RuntimeError(f"could not create level {path}")
    return les, path


def all_actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def by_label():
    return {a.get_actor_label(): a for a in all_actors()}


def spawn(cls_or_obj, label, folder, location=(0, 0, 0), rotation=None):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rot = rotation or unreal.Rotator(0, 0, 0)
    if isinstance(cls_or_obj, unreal.StaticMesh):
        a = eas.spawn_actor_from_class(unreal.StaticMeshActor, vec(location), rot)
        a.static_mesh_component.set_static_mesh(cls_or_obj)
    else:
        a = eas.spawn_actor_from_class(cls_or_obj, vec(location), rot)
    if a is None:
        raise RuntimeError(f"could not spawn {label}")
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


def set_color(light, c):
    """Light colours are stored sRGB-encoded; Blender gives linear."""
    try:
        light.set_light_color(lc(c), True)
    except TypeError:
        light.set_light_color(lc(c))   # sky light: always converts from linear


def comp(actor, cls):
    return actor.get_component_by_class(cls)


def movable(actor):
    actor.root_component.set_mobility(unreal.ComponentMobility.MOVABLE)


def make_rot(forward, up=None):
    if up is None:
        return unreal.MathLibrary.make_rot_from_x(vec(forward))
    return unreal.MathLibrary.make_rot_from_xz(vec(forward), vec(up))


def place_camera(view, cam_info, label):
    a = spawn(unreal.CineCameraActor, label, "Cameras", view["location"], make_rot(view["forward"], view["up"]))
    c = a.get_cine_camera_component()
    import math
    sh = 24.0
    sw = sh * cam_info["aspect"]
    fb = unreal.CameraFilmbackSettings()
    fb.set_editor_property("sensor_width", sw)
    fb.set_editor_property("sensor_height", sh)
    setp(c, "filmback", fb)
    setp(c, "current_focal_length", (sh / 2) / math.tan(math.radians(cam_info["fov_y_deg"]) / 2))
    fs = c.get_editor_property("focus_settings")
    fs.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    setp(c, "focus_settings", fs)
    setp(c, "constrain_aspect_ratio", True)
    return a


def build_level(plan, cap_name, meshes, mis, look_mis, snow_mesh, defaults):
    les, level_path = open_level(cap_name)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    old = [a for a in all_actors() if not isinstance(a, (unreal.WorldSettings, unreal.Brush))]
    if old:
        eas.destroy_actors(old)
    n = 0
    for act in plan["actors"]:
        sm = meshes.get(act["mesh"])
        if sm is None:
            warn(f"mesh missing: {act['mesh']}")
            continue
        a = spawn(sm, act["name"], act["folder"], act["location"])
        c = a.static_mesh_component
        c.set_material(0, mis[act["material"]])
        if not act["cast_shadow"]:
            c.set_cast_shadow(False)
        if act["hidden"]:
            c.set_visibility(False)
            a.set_actor_hidden_in_game(True)
        a.tags = ["gtb_" + act["folder"].split()[0].lower()]
        n += 1
    log(f"level: {n} mesh actors")

    cam = place_camera(plan["camera"], plan["camera"], "GameCamera")
    for v in plan["closeups"]:
        place_camera(v, plan["camera"], v["name"])

    for i, L in enumerate(plan["game_lights"]):
        a = spawn(unreal.PointLight, f"GL{i:04d}_{L['kind']}", "Game Lights", L["location"])
        movable(a)
        a.tags = ["gtb_game_light"]
    sun = spawn(unreal.DirectionalLight, "Sun", "Lighting")
    movable(sun)
    sky = spawn(unreal.SkyLight, "SkyLight", "Lighting")
    movable(sky)
    fog = spawn(unreal.ExponentialHeightFog, "HeightFog", "Lighting", plan["camera"]["location"])
    dome = spawn(EAL.load_asset("/Engine/BasicShapes/Sphere"), "SkyDome", "Lighting", plan["camera"]["location"])
    dome.set_actor_scale3d(unreal.Vector(10000, 10000, 10000))   # 5 km radius
    dc = dome.static_mesh_component
    dc.set_material(0, look_mis["sky"])
    dc.set_cast_shadow(False)
    for k in ("affect_dynamic_indirect_lighting", "affect_distance_field_lighting", "visible_in_ray_tracing"):
        setp(dc, k, False)
    ppv = spawn(unreal.PostProcessVolume, "PostProcess", "Lighting")
    setp(ppv, "unbound", True)
    snow = spawn(snow_mesh, "Snowfall", "Effects", plan["camera"]["location"])
    snow.static_mesh_component.set_material(0, look_mis["snowfall"])
    snow.static_mesh_component.set_cast_shadow(False)
    les.save_current_level()
    log(f"level saved: {level_path}")
    return level_path


# --------------------------------------------------------------------------- look

def apply_look(look, plan, cap, look_mis, defaults):
    actors = by_label()
    physical = look["sky"]["type"] != "color"

    # Colour pipeline: Blender's grade + view transform as a LUT.
    lut = import_texture(look["lut"]["file"], cap, "T_LUT", "lut")
    set_params(look_mis["tonemap"], {"ExposureScale": look["exposure_scale"], "LutSize": look["lut"]["size"],
                                     "LutLo": look["lut"]["lo"], "LutHi": look["lut"]["hi"]}, None, {"LUT": lut})

    # Materials (GTB_Params in Blender).
    s = look["surface"]
    for key in ("surface", "surface_masked"):
        set_params(look_mis[key], s, {"GroundColor": look["ground_color"]})
    set_params(look_mis["snowdrift"], {"NormalStrength": look["normal_strength"]}, {"DriftColor": look["drift_color"]})
    p = look["particles"]
    common = {k: p[k] for k in ("Opacity", "Billboard", "Rise")}
    set_params(look_mis["smoke"], dict(common, Translucency=p["Translucency"], Ambient=p["Ambient"],
                                       NormalStrength=p["NormalStrength"]), {"SmokeColor": p["SmokeColor"]})
    set_params(look_mis["fire"], dict(common, FireStrength=p["FireStrength"]))
    sn = look["snow"]
    set_params(look_mis["snowfall"], {k: sn[k] for k in ("FollowCamera", "FallSpeed", "Flutter", "Size",
                                                          "CountFraction", "Opacity", "Brightness")},
               {k: sn[k] for k in ("Extent", "BoxMin", "BoxMax", "Wind")})
    set_params(look_mis["sky"], None, {"SkyColor": look["sky"]["color"]})
    for mi in look_mis.values():
        EAL.save_loaded_asset(mi)

    for a in all_actors():
        tags = [str(t) for t in a.tags]
        if "gtb_particles" in tags:
            a.set_actor_hidden_in_game(not p["enabled"])
            a.static_mesh_component.set_visibility(p["enabled"])
    snow = actors.get("Snowfall")
    if snow:
        snow.set_actor_hidden_in_game(not sn["enabled"])
        snow.static_mesh_component.set_visibility(sn["enabled"])

    # Sun.
    sun = actors["Sun"]
    sun.set_actor_rotation(make_rot(look["sun"]["forward"]), False)
    sc = comp(sun, unreal.DirectionalLightComponent)
    sc.set_intensity(look["sun"]["lux"])
    set_color(sc, look["sun"]["color"])
    setp(sc, "light_source_angle", look["sun"]["angle"])
    setp(sc, "atmosphere_sun_light", physical)
    sc.set_cast_shadows(True)

    # Sky: flat colour = white cubemap tinted by the sky colour (Blender world
    # background), or Unreal's physical atmosphere for Blender's sky models.
    skl = comp(actors["SkyLight"], unreal.SkyLightComponent)
    dome = actors["SkyDome"]
    atmo = actors.get("SkyAtmosphere")
    if physical:
        if atmo is None:
            atmo = spawn(unreal.SkyAtmosphere, "SkyAtmosphere", "Lighting")
        setp(skl, "source_type", unreal.SkyLightSourceType.SLS_CAPTURED_SCENE)
        setp(skl, "real_time_capture", True)
        skl.set_intensity(look["sky"]["strength"])
        set_color(skl, look["sky"]["tint"])
        dome.set_actor_hidden_in_game(True)
        dome.static_mesh_component.set_visibility(False)
    else:
        if atmo is not None:
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(atmo)
        setp(skl, "source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
        setp(skl, "cubemap", defaults["white_cube"])
        setp(skl, "real_time_capture", False)
        c = look["sky"]["color"]
        m = max(max(c), 1e-6)
        skl.set_intensity(m)
        set_color(skl, [x / m for x in c])
        dome.set_actor_hidden_in_game(False)
        dome.static_mesh_component.set_visibility(True)
    setp(skl, "lower_hemisphere_is_black", False)

    # Fog: exponential height fog with ~zero falloff = Blender's depth fog.
    fc = comp(actors["HeightFog"], unreal.ExponentialHeightFogComponent)
    f = look["fog"]
    fc.set_visibility(f["enabled"])
    actors["HeightFog"].set_actor_hidden_in_game(not f["enabled"])
    fc.set_fog_density(f["density"])
    fc.set_fog_height_falloff(0.001)
    fc.set_fog_max_opacity(f["max_opacity"])
    fc.set_start_distance(f["start_cm"])
    fc.set_fog_inscattering_color(lc(f["color"]))
    fc.set_directional_inscattering_color(lc([0, 0, 0]))
    fc.set_volumetric_fog(False)
    setp(fc, "sky_atmosphere_ambient_contribution_color_scale", lc([0, 0, 0]))

    # Post process: Blender's view transform (our LUT pass) and a neutral Unreal
    # tonemapper that only encodes for the output; manual exposure, bloom, GI strength.
    ppv = actors["PostProcess"]
    ps = ppv.get_editor_property("settings")
    b = look["bloom"]
    for k, v in {"tonemapping_method": unreal.TonemappingMethod.FILMIC, "tone_curve_amount": 0.0,
                 "blue_correction": 0.0, "expand_gamut": 0.0,
                 "local_exposure_highlight_contrast_scale": 1.0, "local_exposure_shadow_contrast_scale": 1.0,
                 "local_exposure_detail_strength": 1.0,
                 "auto_exposure_method": unreal.AutoExposureMethod.AEM_MANUAL, "auto_exposure_bias": 0.0,
                 "auto_exposure_apply_physical_camera_exposure": False,
                 "bloom_method": unreal.BloomMethod.BM_SOG, "bloom_intensity": b["intensity"],
                 "bloom_threshold": b["threshold"], "bloom_size_scale": b["size_scale"],
                 "indirect_lighting_intensity": look["indirect_intensity"],
                 "motion_blur_amount": 0.0, "vignette_intensity": 0.0, "scene_fringe_intensity": 0.0,
                 "film_grain_intensity": 0.0, "lens_flare_intensity": 0.0,
                 "lumen_final_gather_quality": 2.0, "lumen_reflection_quality": 2.0}.items():
        if setp(ps, k, v):
            setp(ps, "override_" + k, True)
    wb = unreal.WeightedBlendable()
    wb.set_editor_property("weight", 1.0)
    wb.set_editor_property("object", look_mis["tonemap"])
    wbs = unreal.WeightedBlendables()
    wbs.set_editor_property("array", [wb])
    setp(ps, "weighted_blendables", wbs)
    ppv.set_editor_property("settings", ps)

    # Game lights recovered from the light volumes.
    g = look["game_lights"]
    for i, (L, spec) in enumerate(zip(plan["game_lights"], g["lights"])):
        a = actors.get(f"GL{i:04d}_{L['kind']}")
        if a is None:
            continue
        pc = comp(a, unreal.PointLightComponent)
        setp(pc, "intensity_units", unreal.LightUnits.CANDELAS)
        pc.set_intensity(spec["candela"])
        pc.set_attenuation_radius(spec["radius_cm"])
        pc.set_source_radius(g["source_radius_cm"])
        set_color(pc, g["color"])
        pc.set_cast_shadows(g["shadows"])
        setp(pc, "use_inverse_squared_falloff", True)
        pc.set_visibility(g["enabled"])
        a.set_actor_hidden_in_game(not g["enabled"])

    # Extra lights from look.json (rebuilt every time).
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    old = [a for a in all_actors() if "gtb_look_light" in [str(t) for t in a.tags]]
    if old:
        eas.destroy_actors(old)
    for i, L in enumerate(look["lights"]):
        cls = {"SUN": unreal.DirectionalLight, "SPOT": unreal.SpotLight, "AREA": unreal.RectLight}.get(
            L["type"], unreal.PointLight)
        a = spawn(cls, L["name"] or f"L{i}", "Look Lights", L["location"], make_rot(L["forward"]))
        movable(a)
        a.tags = ["gtb_look_light"]
        lcmp = a.get_component_by_class(unreal.LightComponent)
        if L["type"] != "SUN":
            setp(lcmp, "intensity_units", unreal.LightUnits.CANDELAS)
            setp(lcmp, "source_radius", L["source_radius_cm"])
        lcmp.set_intensity(L["intensity"])
        set_color(lcmp, L["color"])

    if look.get("smoke_volumes"):
        warn("look.smoke.enabled: volumetric smoke plumes are Blender-only; the sprites carry the smoke")
    log("look applied")


# --------------------------------------------------------------------------- sequences / render

def time_track(seq, mpc, frames, t0, t1):
    t = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack)
    t.set_editor_property("mpc", mpc)
    sec = t.add_section()
    sec.set_range(0, frames)
    ticks = seq.get_tick_resolution().numerator / seq.get_tick_resolution().denominator
    per_frame = ticks / (seq.get_display_rate().numerator / seq.get_display_rate().denominator)
    sec.add_scalar_parameter_key("SceneTime", unreal.FrameNumber(0), t0)
    sec.add_scalar_parameter_key("SceneTime", unreal.FrameNumber(int(round(frames * per_frame))), t1)
    sec.add_scalar_parameter_key("EngineTimeWeight", unreal.FrameNumber(0), 0.0)
    for ch in sec.get_all_channels():
        for k in ch.get_keys():
            try:
                k.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_LINEAR)
            except Exception:  # noqa: BLE001
                pass


def make_sequence(name, path, cuts, frames, fps, mpc, t0, t1):
    full = f"{path}/{name}"
    if EAL.does_asset_exist(full):
        EAL.delete_asset(full)
    seq = AT.create_asset(name, path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(frames)
    cct = seq.add_track(unreal.MovieSceneCameraCutTrack)
    bindings = {}
    for actor, s, e in cuts:
        label = actor.get_actor_label()
        if label not in bindings:
            bindings[label] = seq.add_possessable(actor)
        sec = cct.add_section()
        sec.set_range(s, e)
        sec.set_camera_binding_id(seq.get_binding_id(bindings[label]))
    time_track(seq, mpc, frames, t0, t1)
    EAL.save_loaded_asset(seq)
    return seq


def make_config(name, path, out_dir, res, temporal, warmup, file_format, cvars=None):
    full = f"{path}/{name}"
    if EAL.does_asset_exist(full):
        EAL.delete_asset(full)
    cfg = AT.create_asset(name, path, unreal.MoviePipelinePrimaryConfig, None)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    dp = unreal.DirectoryPath()
    dp.set_editor_property("path", out_dir)
    out.set_editor_property("output_directory", dp)
    out.set_editor_property("file_name_format", file_format)
    out.set_editor_property("output_resolution", unreal.IntPoint(int(res[0]), int(res[1])))
    out.set_editor_property("override_existing_output", True)
    aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    for k, v in {"spatial_sample_count": 1, "temporal_sample_count": temporal, "override_anti_aliasing": True,
                 "anti_aliasing_method": unreal.AntiAliasingMethod.AAM_NONE,
                 "engine_warm_up_count": warmup, "render_warm_up_count": 32, "render_warm_up_frames": True}.items():
        setp(aa, k, v)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    if cvars:
        cv = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        for k, v in cvars.items():
            try:
                cv.add_or_update_console_variable(k, float(v))
            except Exception as e:  # noqa: BLE001
                warn(f"cvar {k}: {e}")
    EAL.save_loaded_asset(cfg)
    return cfg


def build_render_setup(plan, look, cap, mpc):
    actors = by_label()
    r = look["render"]
    fps, frames = r["fps"], r["anim_frames"]
    rdir = f"{cap}/Render"
    views = ["GameCamera"] + [v["name"] for v in plan["closeups"]]
    t_still = 1.0 / fps          # Blender renders frame 1
    make_sequence("LS_Stills", rdir, [(actors[v], i, i + 1) for i, v in enumerate(views)], len(views), fps,
                  mpc, t_still, t_still)
    make_sequence("LS_Anim", rdir, [(actors["GameCamera"], 0, frames)], frames, fps, mpc,
                  1.0 / fps, (frames + 1) / fps)
    w, h = r["resolution"]
    res = (int(round(w * r["preview_scale"])), int(round(h * r["preview_scale"])))
    out = os.path.join(plan["capture_dir"], "unreal", "renders")
    make_config("MRQ_Stills", rdir, os.path.join(out, "stills"), res, r["temporal_samples"], r["warmup_frames"],
                "{camera_name}", r.get("cvars"))
    make_config("MRQ_Anim", rdir, os.path.join(out, "anim"), res, max(1, r["temporal_samples"] // 2), 32,
                "frame.{frame_number}", r.get("cvars"))
    log(f"render setup: stills {views} at {res[0]}x{res[1]}, anim {frames} frames")


# --------------------------------------------------------------------------- main

def main():
    plan = json.load(open(job["plan"], encoding="utf-8"))
    look = json.load(open(job["look"], encoding="utf-8"))
    cap_name = plan["capture"]
    cap = f"/Game/GTB/{cap_name}"
    defaults, mpc, masters = ensure_shared(job["shared"])

    if job["action"] == "build":
        tdir = f"{cap}/Textures"
        textures = {}
        for spec in plan["textures"]:
            name = "T_" + os.path.splitext(spec["file"])[0]
            textures[spec["file"]] = import_texture(os.path.join(plan["textures_dir"], spec["file"]), tdir, name,
                                                    spec["kind"])
        log(f"textures: {len(textures)} imported")
        mdir = f"{cap}/Meshes"
        opts = mesh_import_options()
        meshes = {}
        for glb, names in plan["glb_meshes"].items():
            got = [EAL.load_asset(p) for p in import_files([(os.path.join(plan["meshes_dir"], glb), None)], mdir, opts)]
            got = {a.get_name(): a for a in got if isinstance(a, unreal.StaticMesh)}
            # Interchange names a single-mesh file's asset after the file.
            if len(names) == 1 and len(got) == 1:
                got = {names[0]: next(iter(got.values()))}
            meshes.update({n: got[n] for n in names if n in got})
            log(f"meshes: {glb} -> {len(got)} of {len(names)}")
        for act in plan["actors"]:
            if act.get("folder") == "Particles" and act["mesh"] in meshes:
                sm = meshes[act["mesh"]]
                setp(sm, "positive_bounds_extension", unreal.Vector(2000, 2000, 5000))
                setp(sm, "negative_bounds_extension", unreal.Vector(2000, 2000, 2000))
        snow_path = f"{SHARED}/SM_Snowfall"   # single-mesh file: asset named after it
        if not EAL.does_asset_exist(snow_path):
            import_files([(plan["snowfall_glb"], None)], SHARED, opts)
        snow_mesh = EAL.load_asset(snow_path)
        if snow_mesh is None:
            raise RuntimeError("snowfall mesh import failed")
        setp(snow_mesh, "positive_bounds_extension", unreal.Vector(1e6, 1e6, 1e6))
        setp(snow_mesh, "negative_bounds_extension", unreal.Vector(1e6, 1e6, 1e6))
        EAL.save_loaded_asset(snow_mesh)
        look_mis, mis = build_materials(plan, cap, masters, textures)
        build_level(plan, cap_name, meshes, mis, look_mis, snow_mesh, defaults)
    else:
        les, _ = open_level(cap_name)
        look_mis, _ = build_materials({"materials": []}, cap, masters, {})

    apply_look(look, plan, cap, look_mis, defaults)
    build_render_setup(plan, look, cap, mpc)
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    EAL.save_directory(cap, only_if_is_dirty=True, recursive=True)
    EAL.save_directory(SHARED, only_if_is_dirty=True, recursive=True)
    log(f"done ({len(WARNINGS)} warnings)")


try:
    main()
except Exception:  # noqa: BLE001
    log("FAILED\n" + traceback.format_exc())
