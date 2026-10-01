"""Check the walking people (M_GTB_Walker, docs/WALKING_PEOPLE.md).

Without Unreal (about 30 s): for every walker in the capture's unreal/plan.json,
its loop, stride and cycle time; how fast a planted foot slips on a straight
walk (from the baked cycle); how far its lowest point gets from the path's
ground over a whole loop, and how far its pose at time 0 is from the game's
(both through walkers.replay, which mirrors the shader and reads the same
textures Unreal does); and which walkers pass through each other in a minute
(they don't avoid each other yet).

    python -m ue.walker_check <capture>          (also writes walker_check/routes.png: every route on a map)

With the editor open on the capture's level: photographs one walker with a
temporary SceneCapture2D at fixed times (the MPC's SceneTime pinned and
EngineTimeWeight 0, both restored afterwards; the capture actor is deleted) and
draws where walkers.replay puts it over each picture, in green. The camera is
placed where ray casts through the rip see the walker's whole loop. Pictures go
to captures/<name>/unreal/walker_check/.

    python -m ue.walker_check <capture> --capture mesh_646_646 --times 0,0.5,1,1.5
    python -m ue.walker_check <capture> --capture mesh_655_655 --times 0,10,20,30 --follow
                                          (a camera beside the walker each time: long routes)
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
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(*D["cams"][0][0]), unreal.Rotator(*D["cams"][0][1]))
c = cap.capture_component2d
c.set_editor_property("texture_target", rt)
c.set_editor_property("fov_angle", D["fov"])
c.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
c.set_editor_property("capture_every_frame", False)
old = (ML.get_scalar_parameter_value(world, mpc, "SceneTime"), ML.get_scalar_parameter_value(world, mpc, "EngineTimeWeight"))
try:
    ML.set_scalar_parameter_value(world, mpc, "EngineTimeWeight", 0.0)
    for i, t in enumerate(D["times"]):
        cap.set_actor_location_and_rotation(unreal.Vector(*D["cams"][i][0]), unreal.Rotator(*D["cams"][i][1]), False, False)
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
    """Planted-foot slip on a straight walk (cm/s): per foot and frame pair where it carries
    the person (walkers._planted), the mean motion of its vertices on the ground, in the
    walking frame plus the steady stride. Also the body's forward speed range (it should be
    steady: an uneven one is a surge in every step) and its rise and fall."""
    R, t, stride, _, _ = W.bake_cycle(p, walk, frames=frames)
    feet = [W._leg_vertices(p, leg) for leg in p.rig.legs]
    pel = p.rig.bone["pelvis"]
    out, root = [], []
    for k in range(frames):
        a, b = p.skin(R[k], t[k]), p.skin(R[(k + 1) % frames], t[(k + 1) % frames])
        a[:, 0] += stride * k / frames
        b[:, 0] += stride * (k + 1) / frames
        root.append(R[k][pel] @ p.joints[pel] + t[k][pel] + [stride * k / frames, 0.0, 0.0])
        for sel in feet:
            if sel is None:
                continue
            on = W._planted(a[sel], b[sel])
            if on.any():
                out.append(np.linalg.norm((b[sel][on] - a[sel][on])[:, :2], axis=1).mean() * frames)
    root = np.array(root)
    speed = np.diff(np.r_[root[:, 0], root[0, 0] + stride]) * frames      # m per cycle
    return np.array(out) * 100.0, stride, (speed.min() / stride, speed.max() / stride), np.ptp(root[:, 2])


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
        sp, stride, surge, bob = slip(p, walk)
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
                    f"({stride / cycle:.1f} m/s, body speed {surge[0] * 100:.0f}-{surge[1] * 100:.0f}% of it, "
                    f"bob {bob * 100:.1f} cm) | planted foot slips {np.median(sp):.1f} cm/s median, "
                    f"{np.percentile(sp, 90):.1f} p90 | lowest point {min(gaps):+.1f}..{max(gaps):+.1f} cm from "
                    f"the ground | start {start:.0f} cm from the game's pose")
    # Routes into geometry: rays along the path at knee, chest and head height.
    hits = route_hits(cap, manifest, info, acts)
    for name, n, where in hits:
        rows.append(f"{name}: its path runs into geometry at {n} point(s), e.g. {where}")
    # Pairs that come within a body width (lanes keep people coming the other way apart).
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


def route_hits(cap, manifest, info, acts, heights=(0.5, 1.0, 1.7)):
    """Per walker: how many path steps run into solid geometry at body height (rays from
    each path point to the next, through the rip's surfaces without the walkers)."""
    paths = W.read_png16(info["paths_png"], info["path_range"])
    lines = {}
    for name, a in acts.items():
        cpd = a["walker"]
        if cpd[3] > 0:
            p = paths[int(cpd[1]), :info["path_samples"] + 1, :3] + a["location"]
            lines[name] = p / [100.0, -100.0, 100.0]                   # Blender m
    if not lines:
        return []
    allp = np.concatenate(list(lines.values()))
    lo, hi = allp.min(0) - 2.0, allp.max(0) + 3.0
    solids = []
    for m in manifest["meshes"]:
        if m.get("category") != "surface" or m["name"] in acts:
            continue
        data = np.load(cap / m["file"])
        P = data["positions"].astype(np.float64)
        if len(P) and np.all(P.min(0) <= hi) and np.all(P.max(0) >= lo):
            solids.append((P, data["indices"].reshape(-1, 3)))
    soup = TriSoup(solids)
    out = []
    for name, p in lines.items():
        d = np.diff(p, axis=0)
        n = np.linalg.norm(d[:, :2], axis=1)
        dirs = np.column_stack([d[:, :2] / np.maximum(n, 1e-9)[:, None], np.zeros(len(d))])
        bad = np.zeros(len(d), bool)
        for i in range(0, len(d), 16):                         # short stretches: few triangles each
            j = slice(i, i + 16)
            q = p[i:i + 17]
            sub = soup.near(q.min(0) - [1.0, 1.0, 0.5], q.max(0) + [1.0, 1.0, 2.5])
            for h in heights:
                bad[j] |= sub.cast(p[j] + [0.0, 0.0, h], dirs[j]) < n[j]
        if bad.any():
            i = int(np.argmax(bad))
            out.append((name, int(bad.sum()), f"({p[i, 0]:.1f}, {p[i, 1]:.1f}) m"))
    return out


def solids_near(cap, manifest, acts, lo, hi):
    """The rip's surfaces (without the walkers) in a box, Blender m."""
    solids = []
    for m in manifest["meshes"]:
        if m.get("category") != "surface" or m["name"] in acts:
            continue
        data = np.load(cap / m["file"])
        P = data["positions"].astype(np.float64)
        if len(P) and np.all(P.min(0) <= hi) and np.all(P.max(0) >= lo):
            solids.append((P, data["indices"].reshape(-1, 3)))
    return TriSoup(solids)


def camera_for(cap, manifest, info, a, acts, dist=650.0, heights=(300.0, 450.0, 600.0), targets=None, soup=None):
    """A spot that sees the walker's whole loop (rays through the rip to its ends and
    middle), or the given target points (Unreal cm). None if no spot sees them."""
    loc = np.array(a["location"])
    if targets is None:
        paths = W.read_png16(info["paths_png"], info["path_range"])
        pts = paths[int(a["walker"][1]), :info["path_samples"], :3] + loc
    else:
        pts = np.asarray(targets, dtype=np.float64)
    c = pts.mean(0)
    d = pts[:, :2] - c[:2]
    axis = np.linalg.eigh(d.T @ d + np.eye(2) * 1e-6)[1][:, 1]
    targets = [pts[np.argmax(d @ axis)], pts[np.argmin(d @ axis)], c]
    cb = c / [100.0, -100.0, 100.0]
    if soup is None:
        soup = solids_near(cap, manifest, acts, cb - 25, cb + 25)
    best = None
    for h in heights:
        for k in range(16):
            side = np.array([math.cos(k * math.pi / 8), math.sin(k * math.pi / 8), 0.0])
            for r in (dist, dist * 1.5):
                cam = c + side * r + [0.0, 0.0, h]
                to_cam = (cam - c) / [100.0, -100.0, 100.0]
                if soup.cast((cb + [0.0, 0.0, 1.0])[None], (to_cam / np.linalg.norm(to_cam))[None])[0] \
                        < np.linalg.norm(to_cam) - 0.3:
                    continue                                   # the camera would be behind a wall
                seen = 0
                for e in targets:
                    v = (e + [0.0, 0.0, 100.0] - cam) / [100.0, -100.0, 100.0]
                    n = np.linalg.norm(v)
                    seen += soup.cast((cam / [100.0, -100.0, 100.0])[None], (v / n)[None])[0] > n - 0.3
                score = seen * 10 + abs(side[:2] @ [-axis[1], axis[0]]) * 2 - h / 1000.0
                if best is None or score > best[0]:
                    best = (score, cam, seen)
    if best is None:
        return None
    _, cam, seen = best
    f = c + [0.0, 0.0, 90.0] - cam
    rot = (0.0, math.degrees(math.atan2(f[2], np.linalg.norm(f[:2]))), math.degrees(math.atan2(f[1], f[0])))
    return cam, rot, f / np.linalg.norm(f), seen


def route_map(cap, plan, manifest, acts, px=4.0):
    """captures/<name>/unreal/walker_check/routes.png: a top view of the walkers' part of
    the city (roads light, everything else dark, open ground white) with each route."""
    import cv2
    info = plan["walkers"]
    paths = W.read_png16(info["paths_png"], info["path_range"])
    lines = {}
    for name, a in sorted(acts.items()):
        cpd = a["walker"]
        if cpd[3] > 0:
            lines[name] = (paths[int(cpd[1]), :info["path_samples"], :3] + a["location"])[:, :2] * [1, -1] / 100.0
    allp = np.concatenate(list(lines.values()) + [np.array([a["location"][:2]]) * [1, -1] / 100.0 for a in acts.values()])
    lo, hi = allp.min(0) - 15.0, allp.max(0) + 15.0
    img = np.full((int((hi[1] - lo[1]) * px), int((hi[0] - lo[0]) * px), 3), 245, np.uint8)
    to = lambda xy: np.stack([(xy[..., 0] - lo[0]) * px, (hi[1] - xy[..., 1]) * px], -1).astype(np.int32)
    road = info.get("road_material")
    actor = {a["name"]: a for a in plan["actors"]}
    for m in manifest["meshes"]:
        a = actor.get(m["name"])
        if not a or a.get("hidden") or a.get("walker") or a.get("folder") not in ("Geometry", "People"):
            continue
        data = np.load(cap / m["file"])
        if "uv0" not in data.files:              # the bare terrain: open ground
            continue
        P = data["positions"]
        if not len(P) or (P[:, :2].max(0) < lo).any() or (P[:, :2].min(0) > hi).any():
            continue
        T = data["indices"].reshape(-1, 3)
        t = P[T]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        keep = np.abs(n[:, 2]) * 0.5 * px * px > 0.5            # what covers at least half a pixel
        col = (150, 195, 240) if a["material"] == road else (95, 95, 95)
        cv2.fillPoly(img, list(to(t[keep][:, :, :2])), col)
    cols = [(200, 0, 0), (0, 150, 0), (0, 0, 220), (180, 0, 180), (0, 140, 140), (210, 110, 0), (90, 0, 160),
            (0, 90, 200), (150, 150, 0), (0, 0, 0), (120, 60, 0)]
    for i, (name, xy) in enumerate(lines.items()):
        q = to(xy)
        cv2.polylines(img, [q], True, cols[i % len(cols)], 2)
        cv2.circle(img, tuple(int(v) for v in q[0]), 6, cols[i % len(cols)], -1)
    for i, (name, a) in enumerate(sorted(acts.items())):
        if a["walker"][3] > 0:
            # start = the walker's position at time 0 (the path's start offset)
            xy = lines[name]
            k = int(round(a["walker"][6] / a["walker"][5] * len(xy))) % len(xy)
            q = to(xy[k])
            cv2.circle(img, tuple(int(v) for v in q), 7, (255, 255, 255), 2)
            cv2.putText(img, name[5:].split("_")[0], (int(q[0]) + 9, int(q[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        cols[list(lines).index(name) % len(cols)], 2)
        else:
            q = to(np.array(a["location"][:2]) * [1, -1] / 100.0)
            cv2.drawMarker(img, (int(q[0]), int(q[1])), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    out = cap / "unreal" / "walker_check" / "routes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def project(P, cam, fwd, fov, w, h):
    right = np.cross([0.0, 0.0, 1.0], fwd)        # Unreal is left-handed: right = up x forward
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    d = P - cam
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    z = d @ fwd
    return np.stack([w / 2 + f * (d @ right) / z, h / 2 - f * (d @ up) / z], 1)


def capture(cap, name, times, w=960, h=540, fov=50.0, follow=False):
    import cv2
    from ue.remote import run
    plan, manifest, people, walk, acts, period = load(cap)
    if name not in acts:
        sys.exit(f"{name} is not a walker; walkers: {sorted(acts)}")
    out = cap / "unreal" / "walker_check"
    out.mkdir(parents=True, exist_ok=True)
    info, a = plan["walkers"], acts[name]
    p = people[name]
    uvs = W.vertex_uvs(p)
    if follow:          # a camera beside the walker at each time (long routes)
        where = [W.replay(info, uvs, a["walker"], t, a["location"], period).mean(0) for t in times]
        allp = np.array(where) / [100.0, -100.0, 100.0]
        soup = solids_near(cap, manifest, acts, allp.min(0) - 15, allp.max(0) + 15)
        cams = [camera_for(cap, manifest, info, a, acts, 550.0, (150.0, 250.0, 400.0), targets=[c], soup=soup)
                for c in where]
        print(f"a camera beside the walker at each of {len(times)} times")
    else:
        cam = camera_for(cap, manifest, info, a, acts)
        print(f"camera sees {cam[3]} of 3 points of the loop")
        cams = [cam] * len(times)
    fallback = next((c for c in cams if c is not None), None)
    if fallback is None:
        sys.exit("no camera spot sees the walker")
    cams = [c if c is not None else fallback for c in cams]
    data = {"name": name, "times": times, "cams": [(c[0].tolist(), c[1]) for c in cams], "fov": fov, "w": w, "h": h,
            "out": str(out.resolve())}
    print(run(CAPTURE.replace("DATA", repr(data))), end="")
    tris = np.load(cap / next(m for m in manifest["meshes"] if m["name"] == name)["file"])["indices"].reshape(-1, 3)
    for i, t in enumerate(times):
        img = cv2.imread(str(out / f"{name}_{i:02d}.png"))
        if img is None:
            print(f"no picture for t={t}")
            continue
        V = W.replay(info, uvs, a["walker"], t, a["location"], period)
        if V is not None:
            xy = project(V, cams[i][0], cams[i][2], fov, w, h)
            for tri in tris[::3]:
                cv2.polylines(img, [xy[tri].astype(np.int32)], True, (0, 255, 0), 1)
        cv2.imwrite(str(out / f"{name}_{i:02d}_overlay.png"), img)
    print(f"pictures and overlays: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--capture", dest="walker", help="editor open: photograph this walker")
    ap.add_argument("--times", default="0,0.5,1,1.5,2,2.5", help="seconds, comma separated (with --capture)")
    ap.add_argument("--follow", action="store_true", help="with --capture: a camera beside the walker at each "
                    "time instead of one that sees its whole loop (long routes)")
    args = ap.parse_args()
    cap = Path(args.capture)
    if not cap.exists():
        cap = ROOT / "captures" / args.capture
    if args.walker:
        capture(cap, args.walker, [float(x) for x in args.times.split(",")], follow=args.follow)
    else:
        for row in check(cap):
            print(row)
        plan, manifest, people, walk, acts, period = load(cap)
        print(f"map of the routes: {route_map(cap, plan, manifest, acts)}")
