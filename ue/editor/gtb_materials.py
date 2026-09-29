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
MASTER_VERSION = "3"


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


def build_surface(mat, defaults, mpc, masked):
    _reset(mat, two_sided=True, shading_model=unreal.MaterialShadingModel.MSM_DEFAULT_LIT,
           blend_mode=unreal.BlendMode.BLEND_MASKED if masked else unreal.BlendMode.BLEND_OPAQUE,
           opacity_mask_clip_value=0.5)
    g = Graph(mat, mpc)
    col, lin = defaults["color"], defaults["linear"]
    inputs = {
        "UV": g.texcoord(0),
        "VColor": g.node(unreal.MaterialExpressionVertexColor),
        "VNormal": g.node(unreal.MaterialExpressionVertexNormalWS),
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
    wpo = g.custom(hlsl.SPRITE_WPO, {
        "D1": g.texcoord(1), "D2": g.texcoord(2), "Corner": g.texcoord(3),
        "CamR": g.view_axis(1, 0, 0), "CamU": g.view_axis(0, 1, 0), "T": g.time(),
        "Rise": g.scalar("Rise", 60.0, "Look"), "Billboard": g.scalar("Billboard", 1.0, "Look"),
    }, F3, desc="GTB Billboard")
    g.out(wpo, "", MP.MP_WORLD_POSITION_OFFSET)


def build_smoke(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE,
           translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING,
           tangent_space_normal=False)
    g = Graph(mat, mpc)
    vc = g.node(unreal.MaterialExpressionVertexColor)
    c = g.custom(hlsl.SMOKE, {
        "UV": g.texcoord(0), "Tan": g.texcoord(4), "Bit": g.texcoord(5), "VColorA": (vc, "A"),
        "Atlas": g.texture("Atlas", defaults["linear"]),
        "CamR": g.view_axis(1, 0, 0), "CamU": g.view_axis(0, 1, 0), "CamF": g.view_axis(0, 0, 1),
        "Opacity": g.scalar("Opacity", 0.7, "Look"), "NormalStrength": g.scalar("NormalStrength", 1.0, "Look"),
        "SmokeColor": g.vector("SmokeColor", (0.45, 0.48, 0.57, 1), "Look"),
        "Ambient": g.scalar("Ambient", 0.25, "Look"), "Translucency": g.scalar("Translucency", 0.6, "Look"),
    }, F3, {"Alpha": F1, "WorldN": F3, "Emis": F3, "Sub": F3}, desc="GTB Smoke")
    g.out(c, "", MP.MP_BASE_COLOR)
    g.out(c, "Alpha", MP.MP_OPACITY)
    g.out(c, "WorldN", MP.MP_NORMAL)
    g.out(c, "Emis", MP.MP_EMISSIVE_COLOR)
    g.out(c, "Sub", MP.MP_SUBSURFACE_COLOR)
    for name, v in (("Roughness", 1.0), ("Specular", 0.0)):
        k = g.node(unreal.MaterialExpressionConstant, r=v)
        g.out(k, "", getattr(MP, f"MP_{name.upper()}"))
    _sprite_wpo(g)


def build_fire(mat, defaults, mpc):
    _reset(mat, two_sided=True, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT,
           shading_model=unreal.MaterialShadingModel.MSM_UNLIT)
    g = Graph(mat, mpc)
    vc = g.node(unreal.MaterialExpressionVertexColor)
    c = g.custom(hlsl.FIRE, {
        "UV": g.texcoord(0), "VColorA": (vc, "A"), "Atlas": g.texture("Atlas", defaults["color"]),
        "Opacity": g.scalar("Opacity", 0.7, "Look"), "FireStrength": g.scalar("FireStrength", 6.0, "Look"),
    }, F3, {"Alpha": F1}, desc="GTB Fire")
    g.out(c, "", MP.MP_EMISSIVE_COLOR)
    g.out(c, "Alpha", MP.MP_OPACITY)
    _sprite_wpo(g)


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
    a = g.custom(hlsl.SNOW_ALPHA, {"UV": g.texcoord(0), "Opacity": g.scalar("Opacity", 0.85, "Look")},
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
    "snowdrift": ("M_GTB_SnowDrift", build_snowdrift),
    "smoke": ("M_GTB_Smoke", build_smoke),
    "fire": ("M_GTB_Fire", build_fire),
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
