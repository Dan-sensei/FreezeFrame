"""The generator's smoke as a Niagara Fluids simulation, built in the open Unreal editor.

    python -m ue.fluid_smoke <capture>                      build or update it (one undo step, unsaved)
    python -m ue.fluid_smoke <capture> --suffix _Test --offset 0 1500 0
                                                           a separate copy beside it, for testing
    python -m ue.fluid_smoke <capture> --remove --suffix _Test   delete that copy again
    python -m ue.fluid_smoke <capture> --show sprites       switch to the sprite column (SmokeColumnNN)
    python -m ue.fluid_smoke <capture> --show fluid         and back (visibility only; rebuilds a plume
                                                            whose Niagara system is missing)

There are two generator smokes; offer the user the choice (docs/GENERATOR_SMOKE.md):
  fluid    a Niagara Fluids simulation (this script): light, billowing, rises out of the furnace
  sprites  camera-facing puffs from the game's own smoke flipbook (M_GTB_Plume, made by
           `gtb.py unreal`, tuned with look.unreal.plume and `python -m ue.live plume`)

See docs/GENERATOR_SMOKE.md. Needs the editor open on the capture's level, with Python Remote
Execution and the NiagaraFluids, ToolsetRegistry and NiagaraToolsets plugins (all in
ue/project_template). Settings: look.unreal.fluid_smoke over ue/look.py UNREAL_DEFAULTS.

What it makes (under /Game/GTB/Shared, <S> = suffix):
  M_GTB_FluidGas<S>          copy of NiagaraFluids' M_3DGas_Base with GTB_* controls spliced in
  MI_GTB_GeneratorPlume<S>   the look (density, albedo, height fade, fire glow)
  NS_GTB_GeneratorPlume<S>   copy of the Grid3D_Gas_Fire template, source made smoke-heavy
and the level actors GeneratorPlume<S> and GeneratorFireLight<S> (folder Smoke). Without a
suffix it also hides the sprite smoke column (SmokeColumnNN) and makes sure the sky dome has
no collision (with collision, the fluid treats the whole city as solid and shows nothing).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtb import config  # noqa: E402
from gtb.scene_common import load_look  # noqa: E402
from ue.export import export  # noqa: E402
from ue.look import fluid_smoke_params  # noqa: E402
from ue.remote import run  # noqa: E402

# Bump when the material splice below changes: an older M_GTB_FluidGas is rebuilt.
MASTER_VERSION = "1"

EDITOR = r'''
import json, unreal
D = json.loads(DATA)
EAL, MEL = unreal.EditorAssetLibrary, unreal.MaterialEditingLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
SH, SFX, C = "/Game/GTB/Shared", D["suffix"], D["cfg"]
MASTER, MI, NS = f"{SH}/M_GTB_FluidGas{SFX}", f"{SH}/MI_GTB_GeneratorPlume{SFX}", f"{SH}/NS_GTB_GeneratorPlume{SFX}"
PLUME, LIGHT = f"GeneratorPlume{SFX}", f"GeneratorFireLight{SFX}"
TS = "NiagaraToolsets.NiagaraToolset_System"


def actors(label=None, tag=None):
    return [a for a in sub.get_all_level_actors()
            if (label is None or a.get_actor_label() == label) and (tag is None or tag in [str(t) for t in a.tags])]


if not (hasattr(unreal, "ToolsetRegistry") and unreal.ToolsetRegistry.is_available()):
    raise RuntimeError("enable the ToolsetRegistry and NiagaraToolsets plugins (ue/project_template) and restart the editor")


def tool(name, **args):
    """A NiagaraToolsets call (the only Python route to Niagara module inputs)."""
    for n in (name, f"{TS}.{name}"):
        r = unreal.ToolsetRegistry.execute_tool(TS, n, json.dumps(args))
        if not r.get_editor_property("error"):
            v = r.get_value_as_json_string()
            v = json.loads(v) if v else None
            return json.loads(v) if isinstance(v, str) else v
    raise RuntimeError(f"{name}: {r.get_editor_property('error')}")


def pname(e):
    try:
        return str(e.get_editor_property("parameter_name"))
    except Exception:  # noqa: BLE001
        return None


def build_master(m):
    """Splice the GTB_* controls into a fresh copy of M_3DGas_Base. Niagara keeps driving the
    original parameters through its bindings; these sit after them."""
    def ex():
        return MEL.get_material_expressions(m)

    def find(name):
        return next(e for e in ex() if pname(e) == name)

    def users(src):
        out = []
        for e in ex():
            ins, names = MEL.get_inputs_for_material_expression(m, e), MEL.get_material_expression_input_names(e)
            out += [(e, names[i]) for i, s in enumerate(ins) if s is not None and s == src]
        return out

    def node(cls, x, y, **props):
        e = MEL.create_material_expression(m, cls, x, y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def param(cls, name, default, x, y):
        return node(cls, x, y, parameter_name=name, default_value=default)

    def link(a, b, inp, out=""):
        MEL.connect_material_expressions(a, out, b, inp)

    S, V = unreal.MaterialExpressionScalarParameter, unreal.MaterialExpressionVectorParameter
    # 1. multipliers after the bound parameters: extinction, albedo, fire emission
    for tname, gname, cls, default in (("DensityGain", "GTB_Density", S, 1.0),
                                       ("Albedo", "GTB_SmokeAlbedo", V, unreal.LinearColor(1, 1, 1, 1)),
                                       ("FireGain", "GTB_FireGain", S, 1.0)):
        src = find(tname)
        x, y = MEL.get_material_expression_node_position(src)
        before = users(src)
        p = param(cls, gname, default, x - 250, y + 60)
        mul = node(unreal.MaterialExpressionMultiply, x + 150, y)
        link(src, mul, "A")
        link(p, mul, "B")
        for e, inp in before:
            link(mul, e, inp)
    # 2. GTB_DensityBind: 0 ignores Niagara's DensityGain (the fire template binds it to 0)
    dg, gd = find("DensityGain"), find("GTB_Density")
    dmul = next(e for e, _ in users(gd) if isinstance(e, unreal.MaterialExpressionMultiply))
    x, y = MEL.get_material_expression_node_position(dg)
    one = node(unreal.MaterialExpressionConstant, x - 200, y - 120, r=1.0)
    bind = param(S, "GTB_DensityBind", 1.0, x - 250, y + 140)
    lerp = node(unreal.MaterialExpressionLinearInterpolate, x + 80, y - 40)
    link(one, lerp, "A")
    link(dg, lerp, "B")
    link(bind, lerp, "Alpha")
    link(lerp, dmul, "A")
    # 3. GTB_AlbedoBind: 0 replaces Niagara's smoke colour x albedo (black in the fire template)
    base = MEL.get_material_property_input_node(m, unreal.MaterialProperty.MP_BASE_COLOR)
    base_out = MEL.get_material_property_input_node_output_name(m, unreal.MaterialProperty.MP_BASE_COLOR)
    alb = find("GTB_SmokeAlbedo")
    x, y = MEL.get_material_expression_node_position(base)
    abind = param(S, "GTB_AlbedoBind", 1.0, x + 100, y + 200)
    alerp = node(unreal.MaterialExpressionLinearInterpolate, x + 300, y)
    link(alb, alerp, "A")
    link(base, alerp, "B", base_out)
    link(abind, alerp, "Alpha")
    MEL.connect_material_property(alerp, "", unreal.MaterialProperty.MP_BASE_COLOR)
    # 4. height fade, h = saturate((world z - GTB_BaseZ) / GTB_FadeHeight): thinner and lighter up high
    x, y = MEL.get_material_expression_node_position(gd)
    wp = node(unreal.MaterialExpressionWorldPosition, x - 900, y + 300)
    zmask = node(unreal.MaterialExpressionComponentMask, x - 700, y + 300, r=False, g=False, b=True, a=False)
    link(wp, zmask, "")
    bz = param(S, "GTB_BaseZ", 0.0, x - 700, y + 400)
    fh = param(S, "GTB_FadeHeight", 1.0e9, x - 500, y + 450)
    dz = node(unreal.MaterialExpressionSubtract, x - 500, y + 320)
    link(zmask, dz, "A")
    link(bz, dz, "B")
    div = node(unreal.MaterialExpressionDivide, x - 300, y + 320)
    link(dz, div, "A")
    link(fh, div, "B")
    h = node(unreal.MaterialExpressionSaturate, x - 150, y + 320)
    link(div, h, "")
    top_d = param(S, "GTB_TopDensity", 1.0, x, y + 450)
    one2 = node(unreal.MaterialExpressionConstant, x, y + 380, r=1.0)
    dl = node(unreal.MaterialExpressionLinearInterpolate, x + 150, y + 380)
    link(one2, dl, "A")
    link(top_d, dl, "B")
    link(h, dl, "Alpha")
    before = users(dmul)
    fm = node(unreal.MaterialExpressionMultiply, x + 300, y + 200)
    link(dmul, fm, "A")
    link(dl, fm, "B")
    for e, inp in before:
        link(fm, e, inp)
    ax, ay = MEL.get_material_expression_node_position(alb)
    before = users(alb)
    top_a = param(V, "GTB_TopAlbedo", unreal.LinearColor(1, 1, 1, 1), ax, ay + 150)
    al = node(unreal.MaterialExpressionLinearInterpolate, ax + 200, ay + 60)
    link(alb, al, "A")
    link(top_a, al, "B")
    link(h, al, "Alpha")
    for e, inp in before:
        link(al, e, inp)
    # 5. emissive = Niagara's emission x GTB_EmissiveBind + h x GTB_TopGlow x extinction (a gentle
    #    lift toward the top). The fire template's emission is its orange heat glow.
    ext = MEL.get_material_property_input_node(m, unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
    ext_out = MEL.get_material_property_input_node_output_name(m, unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
    emi = MEL.get_material_property_input_node(m, unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    emi_out = MEL.get_material_property_input_node_output_name(m, unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    x, y = MEL.get_material_expression_node_position(emi)
    ebind = param(S, "GTB_EmissiveBind", 1.0, x - 200, y - 150)
    emul = node(unreal.MaterialExpressionMultiply, x - 60, y - 100)
    link(emi, emul, "A", emi_out)
    link(ebind, emul, "B")
    glow = param(S, "GTB_TopGlow", 0.0, x, y + 250)
    g1 = node(unreal.MaterialExpressionMultiply, x + 150, y + 250)
    link(h, g1, "A")
    link(glow, g1, "B")
    g2 = node(unreal.MaterialExpressionMultiply, x + 300, y + 250)
    link(g1, g2, "A")
    link(ext, g2, "B", ext_out)
    add = node(unreal.MaterialExpressionAdd, x + 450, y + 100)
    link(emul, add, "A")
    link(g2, add, "B")
    MEL.connect_material_property(add, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)


def master():
    if EAL.does_asset_exist(MASTER):
        m = EAL.load_asset(MASTER)
        if EAL.get_metadata_tag(m, "gtb_version") == D["master_version"]:
            return m, False
        EAL.delete_asset(MASTER)
    m = EAL.duplicate_asset("/NiagaraFluids/Materials/Grid3D/M_3DGas_Base", MASTER)
    build_master(m)
    MEL.recompile_material(m)
    EAL.set_metadata_tag(m, "gtb_version", D["master_version"])
    st = MEL.get_statistics(m)
    if not st.num_pixel_shader_instructions:
        raise RuntimeError(f"{MASTER} failed to compile; see the Output Log")
    return m, True


def lc(v):
    return unreal.LinearColor(*(list(v) + [1.0])[:4])


with unreal.ScopedEditorTransaction("GTB: generator fluid smoke" + SFX):
    # The sky dome must not collide: set the profile, the flag alone reverts on load.
    if not SFX:
        for a in actors("SkyDome"):
            c = a.static_mesh_component
            if c.get_collision_profile_name() != "NoCollision":
                c.modify()
                c.set_collision_profile_name("NoCollision")
                c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
                print("GTB fluid smoke: SkyDome collision off")
    m, rebuilt = master()
    mi = EAL.load_asset(MI) if EAL.does_asset_exist(MI) else \
        AT.create_asset(MI.rsplit("/", 1)[1], SH, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mi.modify()
    MEL.set_material_instance_parent(mi, m)
    rim = unreal.Vector(*D["rim"])
    for k, v in {"GTB_DensityBind": 0.0, "GTB_AlbedoBind": 0.0, "GTB_Density": C["density"],
                 "GTB_FireGain": C["fire_gain"], "GTB_TopGlow": C["top_glow"], "GTB_TopDensity": C["top_density"],
                 "GTB_BaseZ": rim.z, "GTB_FadeHeight": C["fade_height"]}.items():
        MEL.set_material_instance_scalar_parameter_value(mi, k, float(v))
    MEL.set_material_instance_vector_parameter_value(mi, "GTB_SmokeAlbedo", lc(C["albedo"]))
    MEL.set_material_instance_vector_parameter_value(mi, "GTB_TopAlbedo", lc(C["top_albedo"]))
    MEL.update_material_instance(mi)

    # The fire template: no sideways bias (the smoke template drifts toward world -x), and its
    # source is an ordinary particle emitter whose inputs the toolset can reach.
    if not EAL.does_asset_exist(NS):
        EAL.duplicate_asset("/NiagaraFluids/Templates/Gas/3D/Systems/Grid3D_Gas_Fire", NS)
    ns = EAL.load_asset(NS)
    sysref = ns.get_path_name()
    for inp, v in (("Density", C["source_density"]), ("Temperature", C["source_temperature"])):
        tool("SetStackInputData",
             stackInputRef={"system": {"refPath": sysref}, "emitterName": "ParticleSourceEmitter",
                            "scriptName": "ParticleUpdateScript", "moduleName": "SetFluidSourceAttributes",
                            "rendererIndex": -1, "inputNameStack": [inp, "B"]},
             inputData={"struct": {"refPath": "/Script/Niagara.NiagaraFloat"}, "value": {"value": float(v)}})
    rp = unreal.find_object(None, sysref + ":Grid3D_Gas_Master_Emitter.NiagaraVolumeRendererProperties_0")
    if rp.get_editor_property("material") != mi:
        rp.modify()
        rp.set_editor_property("material", mi)

    off = unreal.Vector(*D["offset"])
    for a in actors(PLUME):
        sub.destroy_actor(a)
    sx, sy = C["source_shift"]                     # the source mesh is lopsided; the light stays centred
    a = sub.spawn_actor_from_object(ns, rim + off + unreal.Vector(sx, sy, -C["depth"]))   # spawn from the asset: set_asset leaves it inactive
    a.set_actor_label(PLUME)
    a.set_folder_path("Smoke")
    a.tags = ["gtb_fluid_smoke"]
    a.set_actor_scale3d(unreal.Vector(C["scale"], C["scale"], C["scale"]))
    c = a.get_component_by_class(unreal.NiagaraComponent)
    c.set_variable_vec3("WorldSpaceSize", unreal.Vector(*C["grid"]))
    c.set_variable_int("ResolutionMaxAxis", int(C["resolution"]))
    c.set_variable_bool("DrawBounds", False)
    c.reset_system()

    lights = actors(LIGHT)
    la = lights[0] if lights else sub.spawn_actor_from_class(unreal.PointLight, rim)
    la.modify()
    la.set_actor_label(LIGHT)
    la.set_folder_path("Smoke")
    la.tags = ["gtb_fluid_smoke"]
    la.set_actor_location(rim + off + unreal.Vector(0, 0, -C["light_depth"]), False, False)
    lcmp = la.get_component_by_class(unreal.PointLightComponent)
    lcmp.set_mobility(unreal.ComponentMobility.MOVABLE)
    lcmp.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    lcmp.set_intensity(float(C["light_cd"]))
    lcmp.set_light_color(lc(C["light_color"]))
    lcmp.set_editor_property("attenuation_radius", float(C["light_reach"]))
    lcmp.set_editor_property("source_radius", float(C["light_radius"]))

    hidden = []
    if not SFX:
        for s in actors(tag="gtb_plume"):           # the sprite column this replaces
            s.modify()
            s.static_mesh_component.set_visibility(False)
            s.set_actor_hidden_in_game(True)
            hidden.append(s.get_actor_label())

# Heterogeneous Volume settings, also in DefaultEngine.ini so they survive a restart.
for cmd in ("r.HeterogeneousVolumes.IndirectLighting 0.35", "r.HeterogeneousVolumes.IndirectLighting.Mode 2",
            "r.HeterogeneousVolumes.SupportOverlappingVolumes 1"):
    unreal.SystemLibrary.execute_console_command(world, cmd)
# Save the plume's own assets now (not the level). They are new packages: a level saved
# on its own (Ctrl+S) kept the actor while closing the editor dropped the unsaved system,
# and the plume came back with no system, showing nothing.
unsaved = [p for p in (MASTER, MI, NS) if not EAL.save_asset(p, only_if_is_dirty=False)]
loc = a.get_actor_location()
print(f"GTB fluid smoke: {PLUME} at ({loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f}), {MASTER.rsplit('/', 1)[1]} "
      f"{'rebuilt' if rebuilt else 'up to date'}, light {C['light_cd']} cd"
      + (f", hid {', '.join(hidden)}" if hidden else "")
      + (f". COULD NOT SAVE {unsaved}" if unsaved else ". Its assets are saved")
      + "; the level isn't: File > Save All to keep the plume in it. The fluid takes ~30 s to fill."
      + (f" The sprite column is the alternative: python -m ue.fluid_smoke {D['capture']} --show sprites" if not SFX else ""))
'''


REMOVE = r'''
import json, unreal
D = json.loads(DATA)
EAL = unreal.EditorAssetLibrary
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
SH, SFX = "/Game/GTB/Shared", D["suffix"]
labels = (f"GeneratorPlume{SFX}", f"GeneratorFireLight{SFX}")
with unreal.ScopedEditorTransaction("GTB: remove fluid smoke" + SFX):
    for a in sub.get_all_level_actors():
        if a.get_actor_label() in labels:
            sub.destroy_actor(a)
unreal.SystemLibrary.collect_garbage()          # the destroyed actor still holds the system until then
left = [p for p in (f"{SH}/NS_GTB_GeneratorPlume{SFX}", f"{SH}/MI_GTB_GeneratorPlume{SFX}", f"{SH}/M_GTB_FluidGas{SFX}")
        if EAL.does_asset_exist(p) and not EAL.delete_asset(p)]
print(f"GTB fluid smoke: removed {', '.join(labels)} and their assets"
      + (f"; could not delete {left} (run again)" if left else ""))
'''


SHOW = r'''
import json, unreal
D = json.loads(DATA)
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
fluid = D["show"] == "fluid"
found = {"fluid": 0, "sprites": 0}
cap = D["capture"]


def switch():
    with unreal.ScopedEditorTransaction("GTB: generator smoke -> " + D["show"]):
        for a in sub.get_all_level_actors():
            tags, label = [str(t) for t in a.tags], a.get_actor_label()
            kind = ("fluid" if "gtb_fluid_smoke" in tags or label.startswith(("GeneratorPlume", "GeneratorFireLight"))
                    else "sprites" if "gtb_plume" in tags or label.startswith("SmokeColumn") else None)
            if kind is None:
                continue
            found[kind] += 1
            on = (kind == "fluid") == fluid
            a.modify()
            for c in a.get_components_by_class(unreal.SceneComponent):
                c.set_visibility(on)
            nc = a.get_component_by_class(unreal.NiagaraComponent)
            if nc:
                if on:
                    nc.activate(True)                 # restarts empty: ~30 s to fill again
                else:
                    nc.deactivate()                   # no simulation cost while hidden
            a.set_actor_hidden_in_game(not on)
    if not found[D["show"]]:
        print("GTB generator smoke: nothing to show: " + (f"build the fluid first: python -m ue.fluid_smoke {cap}" if fluid
              else f"the sprite column (SmokeColumnNN) comes from the level build: python gtb.py unreal {cap}"))
    else:
        print(f"GTB generator smoke: showing the {'Niagara Fluids plume' if fluid else 'sprite column'} "
              f"({found['fluid']} fluid / {found['sprites']} sprite actor(s)). Switch back: "
              f"python -m ue.fluid_smoke {cap} --show {'sprites' if fluid else 'fluid'}. Unsaved: File > Save All to keep it.")


# A plume whose Niagara system is gone (a level saved without its new assets) shows
# nothing: main() rebuilds it instead of switching.
broken = [a.get_actor_label() for a in sub.get_all_level_actors() if a.get_actor_label() == "GeneratorPlume"
          and (a.get_component_by_class(unreal.NiagaraComponent) is None
               or a.get_component_by_class(unreal.NiagaraComponent).get_asset() is None)]
if fluid and broken:
    print("GTB_REBUILD: " + ", ".join(broken) + " has no Niagara system")
else:
    switch()
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--suffix", default="", help="asset/actor name suffix, for a test copy")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="move the copy (cm) away from the furnace, e.g. 0 1500 0")
    ap.add_argument("--remove", action="store_true", help="delete the copy named by --suffix")
    ap.add_argument("--show", choices=("fluid", "sprites"),
                    help="switch which generator smoke is visible (rebuilds a fluid plume whose system is missing)")
    args = ap.parse_args()
    cfg = config.load()
    cap = Path(args.capture)
    if not cap.exists():
        cap = Path(cfg.get("captures_dir") or ROOT / "captures") / args.capture
    if args.show:
        data = {"show": args.show, "capture": cap.name}
        out = run(SHOW.replace("json.loads(DATA)", "json.loads(" + repr(json.dumps(data)) + ")"), cfg)
        if "GTB_REBUILD" not in out:
            print(out, end="")
            return
        print("GTB generator smoke: " + out.split("GTB_REBUILD: ", 1)[1].strip() + "; rebuilding it")
        args.suffix, args.offset, args.remove = "", (0.0, 0.0, 0.0), False
    params = None
    if not args.remove:
        plan = export(cap)
        params = fluid_smoke_params(load_look(cap / "look.json"), plan)
        if params is None:
            sys.exit("this capture has no smoke column (plan.plume) to put the plume at")
    if args.remove and not args.suffix:
        sys.exit("--remove only deletes a test copy: give its --suffix")
    data = {"suffix": args.suffix, "offset": list(args.offset), "capture": cap.name,
            "master_version": MASTER_VERSION, "cfg": params, "rim": params["rim"] if params else None}
    code = (REMOVE if args.remove else EDITOR).replace("json.loads(DATA)", "json.loads(" + repr(json.dumps(data)) + ")")
    print(run(code, cfg), end="")


if __name__ == "__main__":
    main()
