"""Master materials, built inside Unreal with MaterialEditingLibrary.

Each master is a thin graph: parameter nodes feeding one Custom (HLSL) node
from gtb_hlsl.py. Per-material choices (which textures exist, channel layouts)
are plain scalar parameters instead of static switches, so every capture's
material instances share one compiled shader per master.
"""
import unreal

import gtb_hlsl as hlsl

MEL = unreal.MaterialEditingLibrary
MP = unreal.MaterialProperty
F1, F2, F3, F4 = (unreal.CustomMaterialOutputType.CMOT_FLOAT1, unreal.CustomMaterialOutputType.CMOT_FLOAT2,
                  unreal.CustomMaterialOutputType.CMOT_FLOAT3, unreal.CustomMaterialOutputType.CMOT_FLOAT4)

# Bump when a master graph changes: existing masters are rebuilt in place.
MASTER_VERSION = "9"


class Graph:
    def __init__(self, mat, time_mpc=None):
        self.mat, self.mpc = mat, time_mpc
        self.y = -1200

    def node(self, cls, x=-900, **props):
        e = MEL.create_material_expression(self.mat, cls, x, self.y)
        self.y += 90
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def scalar(self, name, default=0.0, group="Material"):
        return self.node(unreal.MaterialExpressionScalarParameter, parameter_name=name,
                         default_value=float(default), group=group)

    def vector(self, name, default=(0, 0, 0, 0), group="Material"):
        return self.node(unreal.MaterialExpressionVectorParameter, parameter_name=name,
                         default_value=unreal.LinearColor(*[float(x) for x in default]), group=group)

    def texture(self, name, tex, group="Textures"):
        e = self.node(unreal.MaterialExpressionTextureObjectParameter, parameter_name=name, group=group)
        e.set_editor_property("texture", tex)
        return e

    def texcoord(self, index):
        return self.node(unreal.MaterialExpressionTextureCoordinate, coordinate_index=index)

    def view_axis(self, x, y, z):
        """A view-space axis in world space (x: right, y: up, z: forward)."""
        c = self.node(unreal.MaterialExpressionConstant3Vector, x=-1300,
                      constant=unreal.LinearColor(float(x), float(y), float(z), 0.0))
        t = self.node(unreal.MaterialExpressionTransform,
                      transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_VIEW,
                      transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_WORLD)
        MEL.connect_material_expressions(c, "", t, "")
        return t

    def time(self):
        """Scene time: driven by the Level Sequence (MPC), or engine time in the editor."""
        st = self.node(unreal.MaterialExpressionCollectionParameter, collection=self.mpc)
        st.set_editor_property("parameter_name", "SceneTime")
        w = self.node(unreal.MaterialExpressionCollectionParameter, collection=self.mpc)
        w.set_editor_property("parameter_name", "EngineTimeWeight")
        tm = self.node(unreal.MaterialExpressionTime)
        return self.custom("return SceneTime + Weight * EngineTime;",
                           {"SceneTime": st, "Weight": w, "EngineTime": tm}, F1, desc="GTB Time")

    def sequence_time(self):
        """Level Sequence time only (0 outside Sequencer): for motion that must not
        accumulate while the editor runs, like smoke drifting upward."""
        st = self.node(unreal.MaterialExpressionCollectionParameter, collection=self.mpc)
        st.set_editor_property("parameter_name", "SceneTime")
        return st

    def engine_weight(self):
        """1 outside Sequencer (editor, Play), 0 while a GTB sequence evaluates."""
        w = self.node(unreal.MaterialExpressionCollectionParameter, collection=self.mpc)
        w.set_editor_property("parameter_name", "EngineTimeWeight")
        return w

    def loop_inputs(self):
        """Inputs of gtb_hlsl.LOOP_FADE (editor-preview loop of the game's sprites)."""
        return {"Phase": self.texcoord(6), "TAll": self.time(), "W": self.engine_weight(),
                "LoopSeconds": self.scalar("LoopSeconds", 6.0, "Look")}

    def custom(self, code, inputs, out_type, outputs=None, desc="GTB"):
        c = self.node(unreal.MaterialExpressionCustom, x=-300)
        c.set_editor_property("description", desc)
        c.set_editor_property("output_type", out_type)
        c.set_editor_property("code", code)
        ins = []
        for name in inputs:
            ci = unreal.CustomInput()
            ci.set_editor_property("input_name", name)
            ins.append(ci)
        c.set_editor_property("inputs", ins)
        outs = []
        for name, typ in (outputs or {}).items():
            co = unreal.CustomOutput()
            co.set_editor_property("output_name", name)
            co.set_editor_property("output_type", typ)
            outs.append(co)
        c.set_editor_property("additional_outputs", outs)
        for name, src in inputs.items():
            expr, out = src if isinstance(src, tuple) else (src, "")
            if not MEL.connect_material_expressions(expr, out, c, name):
                raise RuntimeError(f"could not connect {name} in {self.mat.get_name()}")
        return c

    def out(self, expr, out_name, prop):
        if not MEL.connect_material_property(expr, out_name, prop):
            raise RuntimeError(f"could not connect {out_name} -> {prop} in {self.mat.get_name()}")


def _reset(mat, **props):
    MEL.delete_all_material_expressions(mat)
    for k, v in props.items():
        mat.set_editor_property(k, v)


def build_surface(mat, defaults, mpc, masked, vnormal=None):
    """vnormal(g): replaces the vertex normal the snow reads (walkers turn it)."""
    _reset(mat, two_sided=True, shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           blend_mode=unreal.BlendMode.BLEND_MASKED if masked else unreal.BlendMode.BLEND_OPAQUE,
           opacity_mask_clip_value=0.5)
    g = Graph(mat, mpc)
    col, lin = defaults["color"], defaults["linear"]
    inputs = {
        "UV": g.texcoord(0),
        "VColor": g.node(unreal.MaterialExpressionVertexColor),
        "VNormal": vnormal(g) if vnormal else g.node(unreal.MaterialExpressionVertexNormalWS),
        "CamVec": g.node(unreal.MaterialExpressionCameraVectorWS),
        "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
        "AlbedoTex": g.texture("Albedo", col), "NormalTex": g.texture("Normal", lin),
        "GrayTex": g.texture("Gray", lin), "PackedTex": g.texture("Packed", lin),
        "EmissiveTex": g.texture("Emissive", col),
        "BaseColor": g.vector("BaseColor", (0.5, 0.5, 0.5, 1)),
        "IsGround": g.scalar("IsGround"),
        "UseAlbedo": g.scalar("UseAlbedo"), "MultiplyVertexColor": g.scalar("MultiplyVertexColor"),
        "UseNormal": g.scalar("UseNormal"), "NormalAG": g.scalar("NormalAG"),
        "NormalRebuildZ": g.scalar("NormalRebuildZ"),
        "NormalFlipGreen": g.scalar("NormalFlipGreen"),
        "NormalRoughMask": g.vector("NormalRoughMask"), "NormalRoughInvert": g.scalar("NormalRoughInvert"),
        "NormalMetalMask": g.vector("NormalMetalMask"), "NormalMetalInvert": g.scalar("NormalMetalInvert"),
        "UseGray": g.scalar("UseGray"), "GrayIsGloss": g.scalar("GrayIsGloss"),
        "UsePacked": g.scalar("UsePacked"),
        "PackedRoughMask": g.vector("PackedRoughMask"), "PackedRoughInvert": g.scalar("PackedRoughInvert"),
        "PackedMetalMask": g.vector("PackedMetalMask"), "PackedMetalInvert": g.scalar("PackedMetalInvert"),
        "UseEmissive": g.scalar("UseEmissive"), "UseSnow": g.scalar("UseSnow", 1.0),
        # Look-wide values (GTB_Params in Blender), set on the capture's look instance.
        "AlbedoGain": g.scalar("AlbedoGain", 1.0, "Look"), "Saturation": g.scalar("Saturation", 1.0, "Look"),
        "Roughness": g.scalar("Roughness", 0.65, "Look"),
        "RoughnessFromGray": g.scalar("RoughnessFromGray", 1.0, "Look"),
        "NormalStrength": g.scalar("NormalStrength", 1.0, "Look"), "Specular": g.scalar("Specular", 0.5, "Look"),
        "EmissionStrength": g.scalar("EmissionStrength", 1.0, "Look"),
        "SnowCover": g.scalar("SnowCover", 0.0, "Look"), "SnowThreshold": g.scalar("SnowThreshold", 0.55, "Look"),
        "GroundColor": g.vector("GroundColor", (0.5, 0.5, 0.5, 1), "Look"),
        "GroundColorWeight": g.scalar("GroundColorWeight", 0.0, "Look"),
    }
    c = g.custom(hlsl.SURFACE, inputs, F3, {"Rough": F1, "Metal": F1, "Spec": F1, "NormalTS": F3,
                                             "Emis": F3, "Alpha": F1}, desc="GTB Surface")
    g.out(c, "", MP.MP_BASE_COLOR)
    g.out(c, "Rough", MP.MP_ROUGHNESS)
    g.out(c, "Metal", MP.MP_METALLIC)
    g.out(c, "Spec", MP.MP_SPECULAR)
    g.out(c, "NormalTS", MP.MP_NORMAL)
    g.out(c, "Emis", MP.MP_EMISSIVE_COLOR)
    if masked:
        g.out(c, "Alpha", MP.MP_OPACITY_MASK)
    g.surface = c
    return g


def build_walker(mat, defaults, mpc):
    """Surface that walks (gtb_hlsl.WALKER_WPO, ue/walkers.py): skinned in the vertex
    shader from the bone and path textures; its normals turned to the current pose."""
    linear = unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
    q = {}

    def vnormal(g):
        wpo = g.custom(hlsl.WALKER_WPO, {
            "I01": g.texcoord(1), "I23": g.texcoord(2), "W01": g.texcoord(3), "W23": g.texcoord(4),
            "BXY": g.texcoord(5), "BZ": g.texcoord(6), "T": g.time(),
            # Per actor (custom primitive data 0-7, see ue/walkers.py). "RGBA": a vector
            # parameter's default output is RGB only.
            "WA": (g.node(unreal.MaterialExpressionVectorParameter, parameter_name="WalkerA",
                          use_custom_primitive_data=True, primitive_data_index=0), "RGBA"),
            "WB": (g.node(unreal.MaterialExpressionVectorParameter, parameter_name="WalkerB",
                          use_custom_primitive_data=True, primitive_data_index=4), "RGBA"),
            "Bones": g.node(unreal.MaterialExpressionTextureObjectParameter, parameter_name="WalkerBones",
                            group="Walkers", texture=defaults["linear"], sampler_type=linear),
            "Paths": g.node(unreal.MaterialExpressionTextureObjectParameter, parameter_name="WalkerPaths",
                            group="Walkers", texture=defaults["linear"], sampler_type=linear),
            "BoneRange": g.scalar("BoneRange", 100.0, "Walkers"),
            "PathRange": (g.vector("PathRange", (1000, 1000, 100, 8), "Walkers"), "RGBA"),
            "Frames": g.scalar("Frames", 48.0, "Walkers"), "Samples": g.scalar("PathSamples", 128.0, "Walkers"),
            "WalkPeriod": g.scalar("WalkPeriod", 1.1, "Look"), "Walk": g.scalar("Walk", 1.0, "Look"),
            "ObjPos": g.node(unreal.MaterialExpressionObjectPositionWS),
            "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
        }, F3, {"QX": F3, "QZ": F3}, desc="GTB Walker")
        q["wpo"] = wpo
        for k in ("QX", "QZ"):           # vertex -> pixel shader
            vi = g.node(unreal.MaterialExpressionVertexInterpolator)
            if not MEL.connect_material_expressions(wpo, k, vi, ""):
                raise RuntimeError(f"could not connect {k} to its vertex interpolator")
            q[k] = vi
        return g.custom(hlsl.WALKER_ROT, {"V": g.node(unreal.MaterialExpressionVertexNormalWS),
                                          "QX": q["QX"], "QZ": q["QZ"]}, F3, desc="GTB Walker vertex normal")

    g = build_surface(mat, defaults, mpc, False, vnormal)
    mat.set_editor_property("tangent_space_normal", False)
    tw = g.node(unreal.MaterialExpressionTransform,
                transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_TANGENT,
                transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_WORLD)
    if not MEL.connect_material_expressions(g.surface, "NormalTS", tw, ""):
        raise RuntimeError("could not connect NormalTS to the tangent -> world transform")
    n = g.custom(hlsl.WALKER_ROT, {"V": tw, "QX": q["QX"], "QZ": q["QZ"]}, F3, desc="GTB Walker normal")
    g.out(n, "", MP.MP_NORMAL)
    g.out(q["wpo"], "", MP.MP_WORLD_POSITION_OFFSET)


def build_cloth(mat, defaults, mpc):
    """Masked surface that flutters in the wind (gtb_hlsl.CLOTH_WPO)."""
    g = build_surface(mat, defaults, mpc, True)
    # Also grows the primitive bounds, so moving cloth isn't culled early.
    mat.set_editor_property("max_world_position_offset_displacement", 600.0)
    # Pre-skinned bounds: ObjectLocalBounds are padded by the max WPO displacement
    # below (PrimitiveSceneProxy::SetTransform), which moved the top 6 m up.
    bounds = g.node(unreal.MaterialExpressionPreSkinnedLocalBounds)
    wpo = g.custom(hlsl.CLOTH_WPO, {
        "Local": g.node(unreal.MaterialExpressionLocalPosition),
        "BMin": (bounds, "Min"), "BMax": (bounds, "Max"),
        "ObjPos": g.node(unreal.MaterialExpressionObjectPositionWS),
        "Normal": g.node(unreal.MaterialExpressionVertexNormalWS), "T": g.time(),
        "Wind": g.vector("Wind", (250, -100, 0, 0), "Look"),
        "Ripple": g.scalar("Ripple", 0.03, "Look"), "Sway": g.scalar("Sway", 0.08, "Look"),
        "WaveLength": g.scalar("WaveLength", 0.75, "Look"), "Speed": g.scalar("FlutterSpeed", 0.5, "Look"),
        # Per actor (custom primitive data 0-11, see ue/export.cloth_room); none = unlimited.
        # "RGBA": a vector parameter's default output is RGB only.
        "Room": (g.node(unreal.MaterialExpressionVectorParameter, parameter_name="ClothRoom",
                        use_custom_primitive_data=True, primitive_data_index=0), "RGBA"),
        "BudA": (g.node(unreal.MaterialExpressionVectorParameter, parameter_name="ClothBudgetA",
                        use_custom_primitive_data=True, primitive_data_index=4), "RGBA"),
        "BudB": (g.node(unreal.MaterialExpressionVectorParameter, parameter_name="ClothBudgetB",
                        use_custom_primitive_data=True, primitive_data_index=8), "RGBA"),
    }, F3, desc="GTB Cloth")
    g.out(wpo, "", MP.MP_WORLD_POSITION_OFFSET)


def build_snowdrift(mat, defaults, mpc):
    _reset(mat, two_sided=True, shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           blend_mode=unreal.BlendMode.BLEND_OPAQUE)
    g = Graph(mat, mpc)
    c = g.custom(hlsl.SNOWDRIFT, {
        "UV": g.texcoord(0), "SnowTex": g.texture("SnowTex", defaults["linear"]),
        "NormalFlipGreen": g.scalar("NormalFlipGreen"),
        "NormalStrength": g.scalar("NormalStrength", 1.0, "Look"),
        "DriftColor": g.vector("DriftColor", (0.71, 0.79, 0.93, 1), "Look"),
    }, F3, {"NormalTS": F3}, desc="GTB Snow drift")
    g.out(c, "", MP.MP_BASE_COLOR)
    g.out(c, "NormalTS", MP.MP_NORMAL)
    rough = g.node(unreal.MaterialExpressionConstant, r=0.6)
    g.out(rough, "", MP.MP_ROUGHNESS)


def _sprite_wpo(g):
    wpo = g.custom(hlsl.SPRITE_WPO, dict({
        "D1": g.texcoord(1), "D2": g.texcoord(2), "Corner": g.texcoord(3),
        "CamR": g.view_axis(1, 0, 0), "CamU": g.view_axis(0, 1, 0), "TSeq": g.sequence_time(),
        "Rise": g.scalar("Rise", 60.0, "Look"), "Billboard": g.scalar("Billboard", 1.0, "Look"),
    }, **g.loop_inputs()), F3, desc="GTB Billboard")
    g.out(wpo, "", MP.MP_WORLD_POSITION_OFFSET)


def build_smoke(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING,
           tangent_space_normal=False)
    g = Graph(mat, mpc)
    vc = g.node(unreal.MaterialExpressionVertexColor)
    c = g.custom(hlsl.SMOKE, dict(g.loop_inputs(), **{
        "UV": g.texcoord(0), "Tan": g.texcoord(4), "Bit": g.texcoord(5), "VColorA": (vc, "A"),
        "Atlas": g.texture("Atlas", defaults["linear"]),
        "CamR": g.view_axis(1, 0, 0), "CamU": g.view_axis(0, 1, 0), "CamF": g.view_axis(0, 0, 1),
        "Opacity": g.scalar("Opacity", 0.7, "Look"), "NormalStrength": g.scalar("NormalStrength", 1.0, "Look"),
        "SmokeColor": g.vector("SmokeColor", (0.45, 0.48, 0.57, 1), "Look"),
        "Ambient": g.scalar("Ambient", 0.25, "Look"),
    }), F3, {"Alpha": F1, "WorldN": F3, "Emis": F3}, desc="GTB Smoke")
    fade = g.custom(hlsl.SOFT_FADE, {
        "Alpha": (c, "Alpha"), "SceneZ": g.node(unreal.MaterialExpressionSceneDepth),
        "PixelZ": g.node(unreal.MaterialExpressionPixelDepth),
        "SoftFade": g.scalar("SoftFade", 100.0, "Look"), "NearFade": g.scalar("NearFade", 1.0, "Look"),
    }, F1, desc="GTB Soft fade")
    g.out(c, "", MP.MP_BASE_COLOR)
    g.out(fade, "", MP.MP_OPACITY)
    g.out(c, "WorldN", MP.MP_NORMAL)
    g.out(c, "Emis", MP.MP_EMISSIVE_COLOR)
    for name, v in (("Roughness", 1.0), ("Specular", 0.0)):
        k = g.node(unreal.MaterialExpressionConstant, r=v)
        g.out(k, "", getattr(MP, f"MP_{name.upper()}"))
    _sprite_wpo(g)


def build_fire(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_UNLIT)
    g = Graph(mat, mpc)
    vc = g.node(unreal.MaterialExpressionVertexColor)
    c = g.custom(hlsl.FIRE, dict(g.loop_inputs(), **{
        "UV": g.texcoord(0), "VColorA": (vc, "A"), "Atlas": g.texture("Atlas", defaults["color"]),
        "Opacity": g.scalar("Opacity", 0.7, "Look"), "FireStrength": g.scalar("FireStrength", 6.0, "Look"),
    }), F3, {"Alpha": F1}, desc="GTB Fire")
    g.out(c, "", MP.MP_EMISSIVE_COLOR)
    g.out(c, "Alpha", MP.MP_OPACITY)
    _sprite_wpo(g)


def build_plume(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING,
           tangent_space_normal=False)
    g = Graph(mat, mpc)
    t = g.time()
    p1, cam_r, cam_u = g.texcoord(1), g.view_axis(1, 0, 0), g.view_axis(0, 1, 0)
    period = g.scalar("Period", 30.0, "Look")
    wpo = g.custom(hlsl.PLUME_WPO, {
        "P1": p1, "P2": g.texcoord(2), "Corner": g.texcoord(3), "CamR": cam_r, "CamU": cam_u, "T": t,
        "Base": g.node(unreal.MaterialExpressionObjectPositionWS),
        "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
        "Period": period, "Height": g.scalar("Height", 8000.0, "Look"), "Grow": g.scalar("Grow", 0.0007, "Look"),
        "Drift": g.vector("Drift", (0.2, 0.0, 0.0, 0.0), "Look"),
        "Radius": g.scalar("Radius", 400.0), "Column": g.scalar("Column", 0.0),
        "PuffSize": g.scalar("PuffSize", 1.6, "Look"), "Spread": g.scalar("Spread", 1.0, "Look"),
    }, F3, desc="GTB Plume motion")
    g.out(wpo, "", MP.MP_WORLD_POSITION_OFFSET)
    vc = g.node(unreal.MaterialExpressionVertexColor)
    c = g.custom(hlsl.PLUME, {
        "UV": g.texcoord(0), "P1": p1, "Corner": g.texcoord(3), "VColorA": (vc, "A"), "T": t, "Period": period,
        "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
        "Base": g.node(unreal.MaterialExpressionObjectPositionWS),
        "Atlas": g.texture("Atlas", defaults["linear"]), "CamR": cam_r, "CamU": cam_u, "CamF": g.view_axis(0, 0, 1),
        "Opacity": g.scalar("Opacity", 0.6, "Look"), "NormalStrength": g.scalar("NormalStrength", 1.0, "Look"),
        "SmokeColor": g.vector("SmokeColor", (0.33, 0.34, 0.38, 1), "Look"), "Ambient": g.scalar("Ambient", 0.25, "Look"),
        "GlowColor": g.vector("GlowColor", (1.0, 0.38, 0.14, 1), "Look"), "Glow": g.scalar("Glow", 1.5, "Look"),
        "ShadowColor": g.vector("ShadowColor", (0.13, 0.11, 0.095, 1), "Look"),
        "Detail": g.scalar("Detail", 1.0, "Look"), "FireReach": g.scalar("FireReach", 2500.0, "Look"),
        "FlameHeight": g.scalar("FlameHeight", 0.16, "Look"), "FlameStrength": g.scalar("FlameStrength", 5.0, "Look"),
    }, F3, {"Alpha": F1, "WorldN": F3, "Emis": F3}, desc="GTB Plume")
    g.out(c, "", MP.MP_BASE_COLOR)
    g.out(c, "Alpha", MP.MP_OPACITY)
    g.out(c, "WorldN", MP.MP_NORMAL)
    g.out(c, "Emis", MP.MP_EMISSIVE_COLOR)
    for name, v in (("Roughness", 1.0), ("Specular", 0.0)):
        k = g.node(unreal.MaterialExpressionConstant, r=v)
        g.out(k, "", getattr(MP, f"MP_{name.upper()}"))


def build_snowfall(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_VOLUMETRIC_NON_DIRECTIONAL)
    g = Graph(mat, mpc)
    wpo = g.custom(hlsl.SNOW_WPO, {
        "Corner": g.texcoord(0), "R1": g.texcoord(1), "R2": g.texcoord(2), "R3": g.texcoord(3),
        "CamPos": g.node(unreal.MaterialExpressionCameraPositionWS),
        "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
        "CamR": g.view_axis(1, 0, 0), "CamU": g.view_axis(0, 1, 0), "T": g.time(),
        "Follow": g.scalar("FollowCamera", 1.0, "Look"), "Extent": g.vector("Extent", (4500, 4500, 3000, 0), "Look"),
        "BoxMin": g.vector("BoxMin", (-5000, -5000, -5000, 0), "Look"),
        "BoxMax": g.vector("BoxMax", (5000, 5000, 0, 0), "Look"),
        "Wind": g.vector("Wind", (250, -100, 0, 0), "Look"), "Fall": g.scalar("FallSpeed", 400.0, "Look"),
        "Flutter": g.scalar("Flutter", 120.0, "Look"), "Size": g.scalar("Size", 10.0, "Look"),
        "Count": g.scalar("CountFraction", 0.6, "Look"),
    }, F3, desc="GTB Snowfall")
    g.out(wpo, "", MP.MP_WORLD_POSITION_OFFSET)
    a = g.custom(hlsl.SNOW_ALPHA, {"UV": g.texcoord(0), "Opacity": g.scalar("Opacity", 0.85, "Look"),
                                   "WorldPos": g.node(unreal.MaterialExpressionWorldPosition),
                                   "CamPos": g.node(unreal.MaterialExpressionCameraPositionWS),
                                   "NearFade": g.scalar("NearFade", 200.0, "Look")},
                 F1, desc="GTB Flake")
    g.out(a, "", MP.MP_OPACITY)
    base = g.node(unreal.MaterialExpressionConstant3Vector, constant=unreal.LinearColor(0.95, 0.97, 1.0, 1.0))
    g.out(base, "", MP.MP_BASE_COLOR)
    glow = g.custom("return float3(0.75, 0.83, 1.0) * 0.25 * Brightness;",
                    {"Brightness": g.scalar("Brightness", 1.0, "Look")}, F3, desc="GTB Flake glow")
    g.out(glow, "", MP.MP_EMISSIVE_COLOR)
    for name, v in (("Roughness", 1.0), ("Specular", 0.0)):
        k = g.node(unreal.MaterialExpressionConstant, r=v)
        g.out(k, "", getattr(MP, f"MP_{name.upper()}"))


def build_sky(mat, defaults, mpc):
    _reset(mat, two_sided=True, shading_model=unreal.MaterialShadingModel.MSM_UNLIT,
           blend_mode=unreal.BlendMode.BLEND_OPAQUE)
    g = Graph(mat, mpc)
    g.out(g.vector("SkyColor", (0.1, 0.17, 0.32, 1), "Look"), "", MP.MP_EMISSIVE_COLOR)


def build_tonemap(mat, defaults, mpc):
    _reset(mat, material_domain=unreal.MaterialDomain.MD_POST_PROCESS,
           blendable_location=unreal.BlendableLocation.BL_SCENE_COLOR_BEFORE_BLOOM)
    g = Graph(mat, mpc)
    scene = g.node(unreal.MaterialExpressionSceneTexture, scene_texture_id=unreal.SceneTextureId.PPI_POST_PROCESS_INPUT0)
    c = g.custom(hlsl.TONEMAP, {
        "Scene": (scene, "Color"), "Exposure": g.node(unreal.MaterialExpressionEyeAdaptation),
        "Lut": g.texture("LUT", defaults["lut"]), "Scale": g.scalar("ExposureScale", 1.0, "Look"),
        "Size": g.scalar("LutSize", 64.0), "Lo": g.scalar("LutLo", -12.0), "Hi": g.scalar("LutHi", 8.0),
    }, F3, desc="GTB Blender view transform")
    g.out(c, "", MP.MP_EMISSIVE_COLOR)


MASTERS = {
    "surface": ("M_GTB_Surface", lambda m, d, p: build_surface(m, d, p, False)),
    "surface_masked": ("M_GTB_SurfaceMasked", lambda m, d, p: build_surface(m, d, p, True)),
    "cloth": ("M_GTB_Cloth", build_cloth),
    "walker": ("M_GTB_Walker", build_walker),
    "snowdrift": ("M_GTB_SnowDrift", build_snowdrift),
    "smoke": ("M_GTB_Smoke", build_smoke),
    "fire": ("M_GTB_Fire", build_fire),
    "plume": ("M_GTB_Plume", build_plume),
    "snowfall": ("M_GTB_Snowfall", build_snowfall),
    "sky": ("M_GTB_Sky", build_sky),
    "tonemap": ("PP_GTB_Tonemap", build_tonemap),
}


def ensure_masters(path, defaults, mpc, log):
    """Create (or rebuild when MASTER_VERSION changed) every master material."""
    at = unreal.AssetToolsHelpers.get_asset_tools()
    EAL = unreal.EditorAssetLibrary
    out = {}
    for key, (name, builder) in MASTERS.items():
        asset = f"{path}/{name}"
        mat = EAL.load_asset(asset) if EAL.does_asset_exist(asset) else None
        if mat is None:
            mat = at.create_asset(name, path, unreal.Material, unreal.MaterialFactoryNew())
        if EAL.get_metadata_tag(mat, "gtb_version") != MASTER_VERSION:
            builder(mat, defaults, mpc)
            MEL.layout_material_expressions(mat)
            MEL.recompile_material(mat)
            EAL.set_metadata_tag(mat, "gtb_version", MASTER_VERSION)
            EAL.save_loaded_asset(mat)
            log(f"master {name} built")
        out[key] = mat
    return out
