"""Push pipeline changes into the Unreal editor the user has open, through
ue.remote (Python Remote Execution), instead of a headless rebuild that needs
the editor closed. Changes go into one undo transaction and stay unsaved.

    python -m ue.live cloth <capture>       banners: M_GTB_Cloth, MI_Look_Cloth, each banner's room
    python -m ue.live materials <capture>   every material instance, texture import kind and actor's material
    python -m ue.live plume <capture>       the smoke column: M_GTB_Plume and MI_Look_Plume from look.json

Both re-export the capture's Unreal plan first, so the editor gets what the
code produces now. The capture's level must be the one open in the editor.
`cloth` rebuilds the master from ue/editor and checks that it compiles:
MaterialEditingLibrary.get_statistics compiles synchronously and reports 0
instructions for a broken shader (a master that fails to compile makes every
banner vanish).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtb import config  # noqa: E402
from gtb.scene_common import load_look  # noqa: E402
from ue.export import export  # noqa: E402
from ue.look import cloth_params, plume_params  # noqa: E402
from ue.remote import run  # noqa: E402

CLOTH = r'''
import sys, importlib, json, unreal
D = json.loads(DATA)
sys.path.insert(0, D["editor_dir"])
import gtb_hlsl, gtb_materials
importlib.reload(gtb_hlsl); importlib.reload(gtb_materials)
MEL, EAL = unreal.MaterialEditingLibrary, unreal.EditorAssetLibrary
SHARED, CAP = "/Game/GTB/Shared", D["content"]
at = unreal.AssetToolsHelpers.get_asset_tools()
with unreal.ScopedEditorTransaction("GTB: banners"):
    defaults = {k: unreal.load_asset(f"{SHARED}/T_GTB_{n}") for k, n in
                (("color", "DefaultColor"), ("linear", "DefaultLinear"), ("lut", "DefaultLut"))}
    path = f"{SHARED}/M_GTB_Cloth"
    mat = EAL.load_asset(path) if EAL.does_asset_exist(path) else \
        at.create_asset("M_GTB_Cloth", SHARED, unreal.Material, unreal.MaterialFactoryNew())
    gtb_materials.build_cloth(mat, defaults, unreal.load_asset(f"{SHARED}/MPC_GTB_Time"))
    MEL.layout_material_expressions(mat)
    MEL.recompile_material(mat)
    EAL.set_metadata_tag(mat, "gtb_version", gtb_materials.MASTER_VERSION)
    st = MEL.get_statistics(mat)
    ok = st.num_vertex_shader_instructions > 0 and st.num_pixel_shader_instructions > 0
    print("GTB live: master built, " + (f"compiles ({st.num_vertex_shader_instructions} VS / "
          f"{st.num_pixel_shader_instructions} PS instructions)" if ok else
          "SHADER COMPILE FAILED: banners using it will vanish; see the editor's Output Log"))
    lp = f"{CAP}/Materials/MI_Look_Cloth"
    look = EAL.load_asset(lp) if EAL.does_asset_exist(lp) else at.create_asset(
        "MI_Look_Cloth", f"{CAP}/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(look, mat)
    src = unreal.load_asset(f"{CAP}/Materials/MI_Look_SurfaceMasked")      # the look's surface values
    for p in src.get_editor_property("scalar_parameter_values"):
        MEL.set_material_instance_scalar_parameter_value(look, p.parameter_info.name, p.parameter_value)
    for p in src.get_editor_property("vector_parameter_values"):
        MEL.set_material_instance_vector_parameter_value(look, p.parameter_info.name, p.parameter_value)
    for k, v in D["params"].items():
        if isinstance(v, list):
            MEL.set_material_instance_vector_parameter_value(look, k, unreal.LinearColor(*v))
        else:
            MEL.set_material_instance_scalar_parameter_value(look, k, float(v))
    for name in D["materials"]:
        mi = EAL.load_asset(f"{CAP}/Materials/{name}")
        mi.modify()
        MEL.set_material_instance_parent(mi, look)
    n = 0
    for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        v = D["room"].get(a.get_actor_label())
        if v and isinstance(a, unreal.StaticMeshActor):
            c = a.static_mesh_component
            c.modify()
            for i in range(0, 12, 4):
                c.set_default_custom_primitive_data_vector4(i, unreal.Vector4(*v[i:i + 4]))
            n += 1
    print(f"GTB live: {len(D['materials'])} banner material(s), room on {n} of {len(D['room'])} banner(s)")
'''

PLUME = r'''
import sys, importlib, json, unreal
D = json.loads(DATA)
sys.path.insert(0, D["editor_dir"])
import gtb_hlsl, gtb_materials
importlib.reload(gtb_hlsl); importlib.reload(gtb_materials)
MEL, EAL = unreal.MaterialEditingLibrary, unreal.EditorAssetLibrary
SHARED, CAP = "/Game/GTB/Shared", D["content"]
def mesh_options():                                  # as gtb_ue.mesh_import_options
    p = unreal.InterchangeGenericAssetsPipeline()
    c = p.get_editor_property("common_meshes_properties")
    for k, v in {"recompute_normals": False, "recompute_tangents": True, "use_mikk_t_space": True,
                 "use_full_precision_u_vs": True, "remove_degenerates": False, "import_lods": False,
                 "bake_meshes": True, "force_all_mesh_as_type": unreal.InterchangeForceMeshType.IFMT_STATIC_MESH}.items():
        c.set_editor_property(k, v)
    m = p.get_editor_property("mesh_pipeline")
    for k, v in {"import_static_meshes": True, "import_skeletal_meshes": False, "generate_lightmap_u_vs": False,
                 "build_nanite": False, "collision": False, "generate_distance_field_as_if_two_sided": True}.items():
        m.set_editor_property(k, v)
    mp = p.get_editor_property("material_pipeline")
    mp.set_editor_property("import_materials", False)
    mp.get_editor_property("texture_pipeline").set_editor_property("import_textures", False)
    stack = unreal.InterchangePipelineStackOverride()
    stack.add_pipeline(p)
    return stack
with unreal.ScopedEditorTransaction("GTB: smoke column"):
    if D.get("glb"):                                 # the puffs (count, frames) come from the export
        t = unreal.AssetImportTask()
        t.set_editor_properties({"filename": D["glb"], "destination_path": f"{CAP}/Meshes", "automated": True,
                                 "save": False, "replace_existing": True, "options": mesh_options()})
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
        pm = EAL.load_asset(f"{CAP}/Meshes/SM_Plume")
        for k in ("positive_bounds_extension", "negative_bounds_extension"):
            pm.set_editor_property(k, unreal.Vector(1e5, 1e5, 1e5))
        print("GTB live: SM_Plume re-imported,", pm.get_num_triangles(0) // 2, "puffs")
    defaults = {k: unreal.load_asset(f"{SHARED}/T_GTB_{n}") for k, n in
                (("color", "DefaultColor"), ("linear", "DefaultLinear"), ("lut", "DefaultLut"))}
    mat = EAL.load_asset(f"{SHARED}/M_GTB_Plume")
    gtb_materials.build_plume(mat, defaults, unreal.load_asset(f"{SHARED}/MPC_GTB_Time"))
    MEL.layout_material_expressions(mat)
    MEL.recompile_material(mat)
    EAL.set_metadata_tag(mat, "gtb_version", gtb_materials.MASTER_VERSION)
    st = MEL.get_statistics(mat)
    ok = st.num_vertex_shader_instructions > 0 and st.num_pixel_shader_instructions > 0
    print("GTB live: plume master built, " + (f"compiles ({st.num_vertex_shader_instructions} VS / "
          f"{st.num_pixel_shader_instructions} PS instructions)" if ok else
          "SHADER COMPILE FAILED: the smoke column will vanish; see the editor's Output Log"))
    look = EAL.load_asset(f"{CAP}/Materials/MI_Look_Plume")
    look.modify()
    for k, v in D["params"].items():
        if isinstance(v, list):
            MEL.set_material_instance_vector_parameter_value(look, k, unreal.LinearColor(*(list(v) + [0.0] * 4)[:4]))
        else:
            MEL.set_material_instance_scalar_parameter_value(look, k, float(v))
    print("GTB live: MI_Look_Plume set:", sorted(D["params"]))
'''

MATERIALS = r'''
import os, json, unreal
D = json.loads(DATA)
MEL, EAL = unreal.MaterialEditingLibrary, unreal.EditorAssetLibrary
CAP = D["content"]
changed = imported = reparented = created = reassigned = 0
def texture(spec):
    global imported
    name = "T_" + os.path.splitext(spec["file"])[0]
    path = f"{CAP}/Textures/{name}"
    if not EAL.does_asset_exist(path):
        t = unreal.AssetImportTask()
        t.set_editor_properties({"filename": os.path.join(D["textures_dir"], spec["file"]),
                                 "destination_path": f"{CAP}/Textures", "destination_name": name,
                                 "automated": True, "save": False, "replace_existing": True})
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
        imported += 1
    return EAL.load_asset(path)
with unreal.ScopedEditorTransaction("GTB: materials"):
    for spec in D["textures"]:                       # sRGB for colour, linear for data (as gtb_ue.configure_texture)
        tex = texture(spec)
        if tex.get_editor_property("srgb") != (spec["kind"] == "albedo"):
            tex.modify()
            tex.set_editor_properties({"srgb": spec["kind"] == "albedo",
                                       "compression_settings": unreal.TextureCompressionSettings.TC_BC7})
            changed += 1
    for m in D["materials"]:
        mi = EAL.load_asset(f"{CAP}/Materials/{m['name']}")
        if mi is None:                               # a material the editor's build didn't have
            mi = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                m["name"], f"{CAP}/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
            created += 1
        mi.modify()
        look = "MI_Look_" + "".join(w.title() for w in m["parent"].split("_"))   # as gtb_ue.build_materials
        parent = EAL.load_asset(f"{CAP}/Materials/{look}") if EAL.does_asset_exist(f"{CAP}/Materials/{look}") else None
        if parent is None:
            print(f"GTB live: {m['name']}: no {look} yet (cloth: run `cloth` first; else a full build)")
        elif mi.parent != parent:
            MEL.set_material_instance_parent(mi, parent)
            reparented += 1
        MEL.clear_all_material_instance_parameters(mi)
        for k, v in (m.get("scalars") or {}).items():
            MEL.set_material_instance_scalar_parameter_value(mi, k, float(v))
        for k, v in (m.get("vectors") or {}).items():
            MEL.set_material_instance_vector_parameter_value(mi, k, unreal.LinearColor(*(list(v) + [0.0] * 4)[:4]))
        for k, spec in (m.get("textures") or {}).items():
            MEL.set_material_instance_texture_parameter_value(mi, k, texture(spec))
    # Material names are numbered in order of first use, so a material added or merged
    # upstream renumbers the ones after it. Point every rip actor at the instance the
    # plan names, so the editor matches a fresh build even after such a shift.
    for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        want = D["actors"].get(a.get_actor_label())
        c = a.get_component_by_class(unreal.StaticMeshComponent) if want else None
        if c is None:
            continue
        cur = c.get_material(0)
        if cur is None or cur.get_name() != want:
            mi = EAL.load_asset(f"{CAP}/Materials/{want}")
            if mi is not None:
                c.modify()
                c.set_material(0, mi)
                reassigned += 1
print(f"GTB live: {len(D['materials'])} material instance(s) reset from the plan ({reparented} re-parented, "
      f"{created} created), {reassigned} actor(s) re-pointed, {changed} texture(s) re-flagged, {imported} imported")
'''


def _send(template, data):
    return run(template.replace("json.loads(DATA)", "json.loads(" + repr(json.dumps(data)) + ")"))


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("cloth", "materials", "plume"):
        sys.exit(__doc__)
    cfg = config.load()
    cap = Path(sys.argv[2])
    if not cap.exists():
        cap = Path(cfg.get("captures_dir") or ROOT / "captures") / sys.argv[2]
    plan = export(cap)
    base = {"content": f"/Game/GTB/{cap.name}", "editor_dir": str(ROOT / "ue" / "editor"),
            "textures_dir": plan["textures_dir"]}
    if sys.argv[1] == "plume":
        pp = plume_params(load_look(cap / "look.json"), plan)
        params = {k: pp[k] for k in ("Period", "Height", "Grow", "Opacity", "Ambient", "NormalStrength", "Glow",
                                     "Detail", "PuffSize", "FireReach", "FlameStrength", "FlameHeight", "Spread",
                                     "Drift", "SmokeColor", "GlowColor", "ShadowColor")}
        print(_send(PLUME, dict(base, params=params, glb=(plan.get("plume") or {}).get("glb"))), end="")
        return
    if sys.argv[1] == "materials":
        print(_send(MATERIALS, dict(base, textures=plan["textures"], materials=plan["materials"],
                                    actors={a["name"]: a["material"] for a in plan["actors"] if a.get("material")})), end="")
        return
    data = dict(base, params=cloth_params(load_look(cap / "look.json")),
                materials=[m["name"] for m in plan["materials"] if m["parent"] == "cloth"],
                room={a["name"]: a["cloth"] for a in plan["actors"] if a.get("cloth")})
    print(_send(CLOTH, data), end="")


if __name__ == "__main__":
    main()
