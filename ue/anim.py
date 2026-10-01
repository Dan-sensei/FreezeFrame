"""Swap the animation a person plays, live in the open editor (docs/WALKING_PEOPLE.md, "Clips").

    python -m ue.anim <capture>                    the clips, and who plays what
    python -m ue.anim <capture> <person> <clip>    swap: person = mesh_655_655, 655 or all;
                                                   clip = walk or a clip in animations/

Every clip is baked for every walker (ue/walkers.py), so a swap only changes the actor's
custom primitive data (bone rows, phase, cycle time, stride): instant, one undo step,
left unsaved. The choice goes into look.json (unreal.walkers.clips), so a rebuild keeps it.

A clip new to animations/ (e.g. a fresh Mixamo download) isn't baked yet: the swap then
runs `python -m ue.live walkers <capture>` first (about 2 minutes).

In the editor, the walkers' clock is the editor's (hours, by now); a person changing
speed would jump along its route, so the swap moves its start along the route to keep
it where it is. A Sequencer render (clock from 0) then starts it elsewhere on its
route; `python -m ue.live walkers <capture>` puts the starts back.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtb import clips as clip_mod, config  # noqa: E402
from ue.remote import run  # noqa: E402

SWAP = r'''
import json, unreal
D = json.loads(DATA)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
ML, MEL = unreal.MaterialLibrary, unreal.MaterialEditingLibrary
mpc = unreal.load_asset("/Game/GTB/Shared/MPC_GTB_Time")
T = ML.get_scalar_parameter_value(world, mpc, "SceneTime") + \
    ML.get_scalar_parameter_value(world, mpc, "EngineTimeWeight") * unreal.GameplayStatics.get_time_seconds(world)
look = unreal.load_asset(D["content"] + "/Materials/MI_Look_Walker")
period = MEL.get_material_instance_scalar_parameter_value(look, "WalkPeriod") if look else 0.0
period = period if period > 0 else D["period"]
bones = unreal.load_asset(D["content"] + "/Textures/T_GTB_WalkerBones")
frames = MEL.get_material_instance_scalar_parameter_value(look, "Frames") if look else 0.0
stale = bones is None or bones.blueprint_get_size_y() < D["rows"] or int(round(frames)) != D["frames"]
done = []
if stale:
    print("GTB anim: STALE (the editor's walker data is older than the plan)")
else:
  with unreal.ScopedEditorTransaction("GTB: animation"):
    for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        name = a.get_actor_label()
        if name not in D["cpd"] or not isinstance(a, unreal.StaticMeshActor):
            continue
        new, secs = D["cpd"][name]
        new = list(new)
        if secs:                                      # a clip: its own length, in WalkPeriods
            new[3] = secs / period
        c = a.static_mesh_component
        old = list(c.get_editor_property("custom_primitive_data").get_editor_property("data"))
        if len(old) >= 8 and old[3] > 0 and new[3] > 0 and new[5] > 0:
            # keep the person where it is: same distance along the route at the editor's time now
            v_old, v_new = old[4] / (old[3] * period), new[4] / (new[3] * period)
            new[6] = (old[6] + T * (v_old - v_new)) % new[5]
        a.modify()
        c.modify()
        for i in range(0, 8, 4):
            c.set_default_custom_primitive_data_vector4(i, unreal.Vector4(*new[i:i + 4]))
        done.append(name)
  print(f"GTB anim: {len(done)} actor(s) set: {', '.join(sorted(done))} (editor time {T:.0f} s, walk period {period:g} s)")
'''


def person_names(plan, who):
    walkers = [a["name"] for a in plan["actors"] if a.get("walker")]
    if who == "all":
        return walkers
    hit = [n for n in walkers if n == who or n.split("_")[1] == who.replace("mesh_", "").split("_")[0]]
    if not hit:
        sys.exit(f"{who} isn't a walker. Walkers: {', '.join(walkers)}")
    return hit[:1]


def main():
    if len(sys.argv) not in (2, 4):
        sys.exit(__doc__)
    cfg = config.load()
    cap = Path(sys.argv[1])
    if not cap.exists():
        cap = Path(cfg["captures_dir"]) / sys.argv[1]
    plan_path = cap / "unreal" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    look_path = cap / "look.json"
    look = json.loads(look_path.read_text(encoding="utf-8")) if look_path.exists() else {}
    chosen = ((look.get("unreal") or {}).get("walkers") or {}).get("clips") or {}
    baked = (plan.get("walkers") or {}).get("clips") or {}
    files = sorted(f.stem for f in clip_mod.CLIP_DIR.glob("*.fbx")) if clip_mod.CLIP_DIR.exists() else []
    if len(sys.argv) == 2:
        print("clips: walk (the game's walk)")
        for n in sorted(set(baked) | set(files)):
            b = baked.get(n)
            print(f"  {n}: " + (f"{b['seconds']:.2f} s{' loop' if b['loop'] else ''}" if b else
                                "in animations/, not baked yet (the first swap to it bakes it)"))
        for a in plan["actors"]:
            if a.get("walker"):
                print(f"{a['name']}: {chosen.get(a['name'], 'walk')}" + ("" if a["walker"][3] > 0 else " (keeps its pose)"))
        return
    who, clip = sys.argv[2], sys.argv[3]
    if clip != "walk" and clip not in baked and clip not in files:
        sys.exit(f"no clip {clip!r}: put {clip}.fbx in {clip_mod.CLIP_DIR} (Mixamo: FBX Binary, Without Skin)")
    names = person_names(plan, who)
    walkers = (look.setdefault("unreal", {}).setdefault("walkers", {}))
    sel = walkers.setdefault("clips", {})
    for n in names:
        if clip == "walk":
            sel.pop(n, None)
        else:
            sel[n] = clip
    look_path.write_text(json.dumps(look, indent=2), encoding="utf-8")
    if clip != "walk" and clip not in baked:
        print(f"{clip} isn't baked yet: re-exporting the walkers and pushing them (about 2 minutes)")
        r = subprocess.run([sys.executable, "-m", "ue.live", "walkers", str(cap)], cwd=ROOT)
        if r.returncode:
            sys.exit(r.returncode)
        return                                    # the push sets every walker's chosen clip
    period = float(walkers.get("period", 1.1))
    cpd = {}
    for a in plan["actors"]:
        if a["name"] in names:
            clips = a.get("walker_clips") or {"walk": a["walker"]}
            if clip not in clips:
                print(f"{a['name']} keeps its pose (no room to walk), so it plays nothing")
                continue
            cpd[a["name"]] = (clips[clip], baked[clip]["seconds"] if clip != "walk" else None)
            a["walker"] = clips[clip]
    plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    info = plan["walkers"]
    rows = info["bone_rows"] * info["count"] * (1 + len(baked))
    out = run(SWAP.replace("json.loads(DATA)", "json.loads(" + repr(json.dumps(
        {"content": f"/Game/GTB/{cap.name}", "cpd": cpd, "period": period, "rows": rows,
         "frames": info["frames"]})) + ")"))
    print(out, end="")
    if "STALE" in out:
        print("pushing the walkers first (about 2 minutes)")
        r = subprocess.run([sys.executable, "-m", "ue.live", "walkers", str(cap)], cwd=ROOT)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
