"""Check the walking people (M_GTB_Walker, docs/WALKING_PEOPLE.md).

Without Unreal (about 30 s): for every walker in the capture's unreal/plan.json,
its loop, stride and cycle time; how fast a planted foot slips on a straight
walk (from the baked cycle); how far its lowest point gets from the path's
ground over a whole loop, and how far its pose at time 0 is from the game's
(both through walkers.replay, which mirrors the shader and reads the same
textures Unreal does); and which walkers pass through each other in a minute
(they don't avoid each other yet).

    python -m ue.walker_check <capture>

With the editor open on the capture's level: photographs one walker with a
temporary SceneCapture2D at fixed times (the MPC's SceneTime pinned and
EngineTimeWeight 0, both restored afterwards; the capture actor is deleted) and
draws where walkers.replay puts it over each picture, in green. The camera is
placed where ray casts through the rip see the walker's whole loop. Pictures go
to captures/<name>/unreal/walker_check/.

    python -m ue.walker_check <capture> --capture mesh_646_646 --times 0,0.5,1,1.5
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtb.scene_common import load_look  # noqa: E402
from ue import walkers as W  # noqa: E402
from ue.export import TriSoup  # noqa: E402
from ue.look import walker_params  # noqa: E402

CAPTURE = r'''
import unreal
D = DATA
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
mpc = unreal.load_asset("/Game/GTB/Shared/MPC_GTB_Time")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
ML = unreal.MaterialLibrary
rt = unreal.RenderingLibrary.create_render_target2d(world, D["w"], D["h"], unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(*D["loc"]), unreal.Rotator(*D["rot"]))
c = cap.capture_component2d
c.set_editor_property("texture_target", rt)
c.set_editor_property("fov_angle", D["fov"])
c.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
c.set_editor_property("capture_every_frame", False)
old = (ML.get_scalar_parameter_value(world, mpc, "SceneTime"), ML.get_scalar_parameter_value(world, mpc, "EngineTimeWeight"))
try:
    ML.set_scalar_parameter_value(world, mpc, "EngineTimeWeight", 0.0)
    for i, t in enumerate(D["times"]):
        ML.set_scalar_parameter_value(world, mpc, "SceneTime", float(t))
        c.capture_scene()
        c.capture_scene()
        unreal.RenderingLibrary.export_render_target(world, rt, D["out"], f"{D['name']}_{i:02d}.png")
finally:
    ML.set_scalar_parameter_value(world, mpc, "SceneTime", old[0])
    ML.set_scalar_parameter_value(world, mpc, "EngineTimeWeight", old[1])
    eas.destroy_actor(cap)
print(f"captured {len(D['times'])}; clock back to SceneTime {old[0]}, EngineTimeWeight {old[1]}")
'''


def load(cap):
    plan = json.loads((cap / "unreal" / "plan.json").read_text(encoding="utf-8"))
    if not plan.get("walkers"):
        sys.exit("no walkers in this capture's plan (run the export: python gtb.py unreal, or python -m ue.live walkers)")
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    walkers, walk, rig = W.find_walkers(cap, manifest, log=lambda *a: None)
    people = {p.name: p for p in walkers}
    acts = {a["name"]: a for a in plan["actors"] if a.get("walker")}
    period = walker_params(load_look(cap / "look.json"))["WalkPeriod"]
    return plan, manifest, people, walk, acts, period


def slip(p, walk, frames=96):
    """Planted-foot slip on a straight walk (cm/s, per frame pair with a foot on the
    ground): the slowest ground vertex, in the walking frame plus the steady stride."""
    R, t, stride, _, _ = W.bake_cycle(p, walk, frames=frames)
    out = []
    for k in range(frames):
        a, b = p.skin(R[k], t[k]), p.skin(R[(k + 1) % frames], t[(k + 1) % frames])
        a[:, 0] += stride * k / frames
        b[:, 0] += stride * (k + 1) / frames
        on = (a[:, 2] < 0.01) & (b[:, 2] < 0.01)
        if on.any():
            out.append(np.linalg.norm((b[on] - a[on])[:, :2], axis=1).min() * frames)
    return np.array(out) * 100.0, stride


def check(cap):
    plan, manifest, people, walk, acts, period = load(cap)
    info = plan["walkers"]
    paths = W.read_png16(info["paths_png"], info["path_range"])
    rows = []
    for name, a in sorted(acts.items()):
        cpd, loc = a["walker"], np.array(a["location"])
        if cpd[3] <= 0:
            rows.append(f"{name}: keeps its pose (no room to walk)")
            continue
        p = people[name]
        uvs = W.vertex_uvs(p)
        cycle = cpd[3] * period
        sp, stride = slip(p, walk)
        sp = sp / cycle                      # per cycle -> per second
        V0 = W.replay(info, uvs, cpd, 0.0, loc, period)
        start = float(np.median(np.linalg.norm(V0 - p.P * [100.0, -100.0, 100.0], axis=1)))
        gaps = []
        loop_s = cpd[5] / cpd[4] * cycle     # one round of the loop
        for t in np.linspace(0, loop_s, 120, endpoint=False):
            V = W.replay(info, uvs, cpd, t, loc, period)
            s = ((cpd[6] + t / cycle * cpd[4]) / cpd[5] % 1.0) * info["path_samples"]
            i0 = int(s)
            g = (1 - (s - i0)) * paths[int(cpd[1]), i0, 2] + (s - i0) * paths[int(cpd[1]), i0 + 1, 2] + loc[2]
            gaps.append(V[:, 2].min() - g)
        rows.append(f"{name}: loop {cpd[5] / 100:.1f} m round, stride {stride:.2f} m, {cycle:.2f} s per cycle "
                    f"({stride / cycle:.1f} m/s) | planted foot slips {np.median(sp):.1f} cm/s median, "
                    f"{np.percentile(sp, 90):.1f} p90 | lowest point {min(gaps):+.1f}..{max(gaps):+.1f} cm from "
                    f"the ground | start {start:.0f} cm from the game's pose")
    # Walkers don't avoid each other yet: pairs that come within a body width.
    moving = [n for n, a in acts.items() if a["walker"][3] > 0]
    uvs = {n: W.vertex_uvs(people[n]) for n in moving}
    ts = np.arange(0.0, 60.0, 0.1)
    track = {n: np.array([W.replay(info, uvs[n], acts[n]["walker"], t, acts[n]["location"], period)[:, :2].mean(0)
                          for t in ts]) for n in moving}
    crossing = 0
    for i, a in enumerate(moving):
        for b in moving[i + 1:]:
            d = np.linalg.norm(track[a] - track[b], axis=1) / 100.0
            if d.min() < 0.6:
                crossing += 1
                rows.append(f"{a} and {b} walk through each other: {d.min():.2f} m apart at closest, "
                            f"under 0.6 m for {(d < 0.6).mean() * 100:.0f}% of a minute")
    rows.append(f"{len(moving)} walk, {len(acts) - len(moving)} keep their pose, {crossing} pair(s) walk through each other")
    return rows


def camera_for(cap, manifest, info, a, acts, dist=650.0, heights=(300.0, 450.0, 600.0)):
    """A spot that sees the walker's whole loop (rays through the rip to its ends and middle)."""
    loc = np.array(a["location"])
    paths = W.read_png16(info["paths_png"], info["path_range"])
    pts = paths[int(a["walker"][1]), :info["path_samples"], :3] + loc
    c = pts.mean(0)
    d = pts[:, :2] - c[:2]
    axis = np.linalg.eigh(d.T @ d)[1][:, 1]
    targets = [pts[np.argmax(d @ axis)], pts[np.argmin(d @ axis)], c]
    cb = c / [100.0, -100.0, 100.0]
    solids = []
    for m in manifest["meshes"]:
        if m.get("category") != "surface" or m["name"] in acts:
            continue
        data = np.load(cap / m["file"])
        P = data["positions"].astype(np.float64)
        if len(P) and np.all(P.min(0) <= cb + 25) and np.all(P.max(0) >= cb - 25):
            solids.append((P, data["indices"].reshape(-1, 3)))
    soup = TriSoup(solids)
    best = None
    for h in heights:
        for k in range(16):
            side = np.array([math.cos(k * math.pi / 8), math.sin(k * math.pi / 8), 0.0])
            for r in (dist, dist * 1.5):
                cam = c + side * r + [0.0, 0.0, h]
                seen = 0
                for e in targets:
                    v = (e + [0.0, 0.0, 100.0] - cam) / [100.0, -100.0, 100.0]
                    n = np.linalg.norm(v)
                    seen += soup.cast((cam / [100.0, -100.0, 100.0])[None], (v / n)[None])[0] > n - 0.3
                score = seen * 10 + abs(side[:2] @ [-axis[1], axis[0]]) * 2 - h / 1000.0
                if best is None or score > best[0]:
                    best = (score, cam, seen)
    _, cam, seen = best
    f = c + [0.0, 0.0, 90.0] - cam
    rot = (0.0, math.degrees(math.atan2(f[2], np.linalg.norm(f[:2]))), math.degrees(math.atan2(f[1], f[0])))
    return cam, rot, f / np.linalg.norm(f), seen


def project(P, cam, fwd, fov, w, h):
    right = np.cross([0.0, 0.0, 1.0], fwd)        # Unreal is left-handed: right = up x forward
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    d = P - cam
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    z = d @ fwd
    return np.stack([w / 2 + f * (d @ right) / z, h / 2 - f * (d @ up) / z], 1)


def capture(cap, name, times, w=960, h=540, fov=50.0):
    import cv2
    from ue.remote import run
    plan, manifest, people, walk, acts, period = load(cap)
    if name not in acts:
        sys.exit(f"{name} is not a walker; walkers: {sorted(acts)}")
    out = cap / "unreal" / "walker_check"
    out.mkdir(parents=True, exist_ok=True)
    info, a = plan["walkers"], acts[name]
    cam, rot, fwd, seen = camera_for(cap, manifest, info, a, acts)
    print(f"camera sees {seen} of 3 points of the loop")
    data = {"name": name, "times": times, "loc": cam.tolist(), "rot": rot, "fov": fov, "w": w, "h": h,
            "out": str(out.resolve())}
    print(run(CAPTURE.replace("DATA", repr(data))), end="")
    p = people[name]
    uvs = W.vertex_uvs(p)
    tris = np.load(cap / next(m for m in manifest["meshes"] if m["name"] == name)["file"])["indices"].reshape(-1, 3)
    for i, t in enumerate(times):
        img = cv2.imread(str(out / f"{name}_{i:02d}.png"))
        if img is None:
            print(f"no picture for t={t}")
            continue
        V = W.replay(info, uvs, a["walker"], t, a["location"], period)
        if V is not None:
            xy = project(V, cam, fwd, fov, w, h)
            for tri in tris[::3]:
                cv2.polylines(img, [xy[tri].astype(np.int32)], True, (0, 255, 0), 1)
        cv2.imwrite(str(out / f"{name}_{i:02d}_overlay.png"), img)
    print(f"pictures and overlays: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--capture", dest="walker", help="editor open: photograph this walker")
    ap.add_argument("--times", default="0,0.5,1,1.5,2,2.5", help="seconds, comma separated (with --capture)")
    args = ap.parse_args()
    cap = Path(args.capture)
    if not cap.exists():
        cap = ROOT / "captures" / args.capture
    if args.walker:
        capture(cap, args.walker, [float(x) for x in args.times.split(",")])
    else:
        for row in check(cap):
            print(row)
