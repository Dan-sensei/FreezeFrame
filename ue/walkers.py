"""Walking people in Unreal (M_GTB_Walker): the people a rip caught upright walk
loops through the streets, with the walk cycle gtb/characters.py fits to them.
The ones lying down stay as they are.

Everything moves in the material (World Position Offset) on MPC_GTB_Time, like
the other effects, so they walk in the editor viewport and in Sequencer renders
alike, and at time 0 every walker stands where the game had it, in its stride.

Per vertex (UV1-6): its 4 bone indices and weights, and its bind position (Unreal
cm, standing at the origin facing +X). Two 16-bit textures (unreal/walkers/):
  bones.png  one row per walker and bone, 3 texels per frame: the rows of
             [R | t] taking the bind position to the walking frame (X forward,
             ground at z 0, the stance foot planted), RGB = R (-1..1), A = t
             (-BoneRange..BoneRange cm). The last frame holds the captured pose's
             rotations: the mesh's own normals and tangents are the captured
             ones, and the material turns them by R_now R_captured^T.
  paths.png  one row per walker: (x, y, z, yaw) at N+1 points evenly spread
             along its closed loop, relative to the actor (PathRange per
             channel); the last point is the first with yaw + 2 pi.
Custom primitive data 0-7 per walker actor: bone row, path row, phase at time 0,
cycle time (x the look's WalkPeriod, seconds for a typical adult), stride (cm per
cycle), loop length (cm), start (cm along the loop), 1. A walker with no room to
walk gets cycle time 0 and keeps its pose.

The loops: from where the game had the person, straight ahead and back as far
as the ground is clear (no wall, beam or crate at knee, chest or head height,
no step over 30 cm, at most MAX_RUN m each way), a second lane beside it, and
half circles between them. Other headings up to 90 degrees off are tried when
the way ahead is short.
"""
import functools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from gtb import characters as ch

FRAMES = 48           # cycle frames in bones.png (linearly blended)
PATH_SAMPLES = 128    # points along each loop
MAX_RUN = 15.0        # m each way from the start
TURN_RADIUS = 0.5     # m: lanes 1 m apart
MIN_LOOP = 2.0        # m of straight path, or the person stays put
STEP = 0.25
BODY = ((0.35, 1.0, 1.7), 0.3)     # ray heights over the ground and half the shoulder width (m)
TOUCH = 0.15
UE = np.diag([1.0, -1.0, 1.0])     # Blender axes -> Unreal axes (and back)
CM = 100.0


def rig_for(manifest):
    """The profile's rig; captures processed before it existed read the current profile file."""
    prof = manifest.get("profile", {})
    rig = ch.rig_from_profile(prof)
    if rig is None:
        f = Path(__file__).resolve().parent.parent / "profiles" / f"{prof.get('_name', 'default')}.json"
        if f.exists():
            rig = ch.rig_from_profile(json.loads(f.read_text(encoding="utf-8")))
    return rig


def find_walkers(capture, manifest, log=print):
    """(walkers, Walk, rig) for a capture, or ([], None, None). Fills in the skin data of
    captures processed before process.py kept it, from the rip if it's still there."""
    rig = rig_for(manifest)
    if rig is None:
        return [], None, None
    skinned = [m for m in manifest["meshes"] if any(a.startswith("BLENDINDICES") for a in m.get("layout_pre", []))]
    if skinned and "skin_bind" not in np.load(Path(capture) / skinned[0]["file"]).files:
        info = Path(capture) / "capture.json"
        rip = json.loads(info.read_text(encoding="utf-8")).get("rip_dir") if info.exists() else None
        if rip and Path(rip).exists():
            n = ch.backfill(capture, manifest, rip)
            log(f"[people] skin data of {n} skinned meshes read from the rip")
        else:
            log("[people] no skin data and the rip is gone: nobody walks")
            return [], None, None
    people = ch.find_people(capture, manifest, rig, log)
    if not people:
        return [], None, rig
    upright = [p for p in people if p.upright()]
    # Frostpunk's amputees (no calf or foot on one leg) walk on a crutch, which a
    # two-legged cycle can't do: they keep their pose, like the people lying down.
    walkers = [p for p in upright if all(b in p.bones for leg in rig.legs for b in leg)]
    log(f"[people] {len(people)} people: {len(walkers)} walk, {len(upright) - len(walkers)} upright on a crutch "
        f"and {len(people) - len(upright)} lying keep their pose")
    if not walkers:
        return [], None, rig
    walk = ch.fit_walk(walkers, rig, log) if len(walkers) >= 3 else ch.default_walk(rig)
    return walkers, walk, rig


# --------------------------------------------------------------------------- cycle bake

def _leg_vertices(p, leg):
    """Vertices bound mostly to a leg's lowest bone there is (foot, else calf, else thigh)."""
    dom = p.idx[np.arange(len(p.idx)), p.w.argmax(1)]
    for b in reversed(leg):
        if b in p.bones:
            sel = dom == b
            if sel.sum() >= 3:
                return sel
    return None


def bake_cycle(p, walk, frames=FRAMES, fine=192):
    """Bone transforms of one walker over the cycle, in the walking frame (Blender axes,
    m: x forward, z up, ground 0). The root rises and falls so the lowest vertex touches
    the ground, and moves forward by delta(phi) on top of the steady stride, so the
    stance foot stays put. Returns (R (F,nb,3,3), t (F,nb,3), stride m, hip height m, H):
    H takes the bind axes to the walking frame."""
    H = ch.frame(np.array([1.0, 0.0, 0.0]), p.rig)
    legs = [_leg_vertices(p, leg) for leg in p.rig.legs]
    legs = [l for l in legs if l is not None]
    phis = np.arange(fine) / fine
    Vs, hz = [], []
    for phi in phis:
        R, t = p.fk(walk.pose(phi), H, np.zeros(3))
        V = p.skin(R, t)
        hz.append(-V[:, 2].min())
        Vs.append([V[l] - [0.0, 0.0, V[:, 2].min()] for l in legs])    # each foot, on a ground at z 0
    hz = np.array(hz)
    # Root advance per step: how far back the foot vertices on the ground in both frames
    # moved (the heel at first, the toes at the end of a stance), so they stay put. With
    # both feet down, the one moving back is the planted one; the other is still landing.
    ds = np.full(fine, np.nan)
    dl = np.zeros(fine)                    # the same sideways: the body sways over the planted foot
    for i in range(fine):
        for f in range(len(legs)):
            a, b = Vs[i][f], Vs[(i + 1) % fine][f]
            on = (a[:, 2] < 0.015) & (b[:, 2] < 0.015)
            if on.any():
                d = -(b[on, 0] - a[on, 0]).mean()
                if np.isnan(ds[i]) or d > ds[i]:
                    ds[i], dl[i] = d, -(b[on, 1] - a[on, 1]).mean()
    if np.isnan(ds).all():
        ds[:] = 0.0
    ds = np.where(np.isnan(ds), np.nanmean(ds), ds)
    sway = np.concatenate([[0.0], np.cumsum(dl)[:-1]])
    sway = sway - phis * dl.sum()          # periodic
    sway -= sway.mean()
    stride = float(ds.sum())
    if stride < 0.2:                       # no clear stance: the feet's spread
        stride = 2.0 * float(np.ptp(np.array([v[0][:, 0].mean() for v in Vs])))
        ds = np.full(fine, stride / fine)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    delta = s - stride * phis
    Rs, ts = [], []
    for k in range(frames):
        phi = k / frames
        R, t = p.fk(walk.pose(phi), H, np.zeros(3))
        lift = float(np.interp(phi, phis, hz, period=1.0))
        t = t + np.array([float(np.interp(phi, phis, delta, period=1.0)),
                          float(np.interp(phi, phis, sway, period=1.0)), lift])
        Rs.append(R)
        ts.append(t)
    return np.array(Rs), np.array(ts), stride, float(hz.mean()), H


# --------------------------------------------------------------------------- paths

def clear_run(soup, p0, f, max_len=MAX_RUN):
    """How far a person can walk from p0 (on the ground) along f: no geometry at knee,
    chest or head height within a shoulder width, ground all the way, no step over 30 cm."""
    left = np.array([-f[1], f[0], 0.0])
    heights, half = BODY
    end = p0 + f * max_len
    box = soup.near(np.minimum(p0, end) - [half + 0.1, half + 0.1, 0.5], np.maximum(p0, end) + [half + 0.1, half + 0.1, 2.0])
    # Rays start TOUCH m ahead: geometry the person already brushes where the game put
    # them (Frostpunk: mesh_1519's head under a ledge) doesn't count.
    o = np.array([p0 + np.array([0, 0, h]) + left * s + f * TOUCH for h in heights for s in (-half, 0.0, half)])
    obstacle = float(box.cast(o, np.broadcast_to(f, o.shape)).min()) + TOUCH - 0.6
    n = int(max_len / STEP)
    pts = p0 + np.outer(np.arange(1, n + 1) * STEP, f) + np.array([0, 0, 1.2])
    d = box.cast(pts, np.broadcast_to([0.0, 0.0, -1.0], pts.shape))
    with np.errstate(invalid="ignore"):
        gz = p0[2] + 1.2 - d
        ok = np.isfinite(d) & (np.abs(np.diff(np.concatenate([[p0[2]], gz]))) < 0.3)
    ground = (np.argmin(ok) if not ok.all() else n) * STEP
    return max(0.0, min(obstacle, ground, max_len))


def plan_loop(soup, p0, f0):
    """(forward, a, c, lane offset) of the longest loop around p0: a m ahead, c m back,
    the return lane `offset` beside it (Blender m, xy)."""
    best = None
    for ang in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90):
        r = math.radians(ang)
        f = np.array([f0[0] * math.cos(r) - f0[1] * math.sin(r), f0[0] * math.sin(r) + f0[1] * math.cos(r), 0.0])
        a, c = clear_run(soup, p0, f), clear_run(soup, p0, -f)
        score = a + c - 0.05 * abs(ang)
        if best is None or score > best[0]:
            best = (score, f, a, c)
    _, f, a, c = best
    if a + c < MIN_LOOP:
        return None
    left = np.array([-f[1], f[0], 0.0])
    lane = None
    for side in (1.0, -1.0):
        q = p0 - f * c + left * side * 2 * TURN_RADIUS
        if clear_run(soup, q, f, a + c + 0.5) >= a + c - 0.1:
            lane = left * side * 2 * TURN_RADIUS
            break
    if lane is None:                       # no room beside: turn about on the spot
        lane = left * 0.04
    return f, a, c, lane


def loop_points(p0, f, a, c, lane):
    """Dense closed polyline (xy) and its segment lengths: lane 1 from the back end to the
    front end through p0, a half circle ahead, lane 2 back beside it, a half circle behind.
    p0 is c m along it."""
    B, A, ln, fw = p0[:2] - f[:2] * c, p0[:2] + f[:2] * a, lane[:2], f[:2]
    r = np.linalg.norm(ln) / 2

    def line(P, Q):
        n = max(int(np.linalg.norm(Q - P) / 0.02), 2)
        return list(P + np.outer(np.linspace(0, 1, n, endpoint=False), Q - P))

    def half_circle(S, bulge):
        C = S + ln / 2 if bulge @ fw > 0 else S - ln / 2       # ahead: to lane 2; behind: back to lane 1
        v0 = S - C
        n = max(int(math.pi * r / 0.02), 8)
        th = np.linspace(0, math.pi, n, endpoint=False)
        return list(C + np.outer(np.cos(th), v0) + np.outer(np.sin(th), bulge * r))

    pts = line(B, A) + half_circle(A, fw) + line(A + ln, B + ln) + half_circle(B + ln, -fw)
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    return pts, seg


def resample(pts, seg, n):
    """n points evenly spread along the closed polyline, their tangent angles (unwrapped,
    n + 1 values: the last is the first after one round) and the loop's length."""
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    closed = np.vstack([pts, pts[:1]])
    q = np.linspace(0, L, n, endpoint=False)
    xy = np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], 1)
    d = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)          # central differences
    yaw = np.unwrap(np.concatenate([np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[:1, 1], d[:1, 0])]))
    turn = yaw[-2] - yaw[0] + (yaw[1] - yaw[0])          # about +-2 pi after one round
    yaw[-1] = yaw[0] + 2 * math.pi * np.sign(turn)
    return xy, yaw, L


# --------------------------------------------------------------------------- plan

def plan(capture, manifest, walkers, walk, soup, actors, log=print):
    """Bake every walker: (per actor name: custom primitive data and mesh bounds extension,
    textures info for plan.json). `actors`: {name: actor dict with location (Unreal cm)}."""
    out_dir = Path(capture) / "unreal" / "walkers"
    out_dir.mkdir(parents=True, exist_ok=True)
    hips = []
    baked = []
    for p in walkers:
        R, t, stride, hip, H = bake_cycle(p, walk)
        baked.append((p, R, t, stride, hip, H))
        hips.append(hip)
    hip_ref = float(np.median(hips)) if hips else 1.0
    nb = max(p.nb for p in walkers)
    bones = np.zeros((nb * len(walkers), 3 * (FRAMES + 1), 4))
    paths = np.zeros((len(walkers), PATH_SAMPLES + 1, 4))
    per_actor, stuck = {}, []
    for row, (p, R, t, stride, hip, H) in enumerate(baked):
        # bind (Blender, scaled) -> walking frame -> Unreal: R' = M R H^T M, t' = 100 M t
        Rl = np.einsum("ij,fbjk,kl->fbil", UE, R, H.T @ UE)
        tl = np.einsum("ij,fbj->fbi", UE, t) * CM
        Rc = np.einsum("ij,bjk,kl->bil", UE, p.R, H.T @ UE)
        for b in range(p.nb):
            r0 = row * nb + b
            for k in range(FRAMES):
                bones[r0, 3 * k:3 * k + 3, :3] = Rl[k, b]
                bones[r0, 3 * k:3 * k + 3, 3] = tl[k, b]
            bones[r0, 3 * FRAMES:3 * FRAMES + 3, :3] = Rc[b]
        for b in range(p.nb, nb):
            for k in range(FRAMES + 1):
                bones[row * nb + b, 3 * k:3 * k + 3, :3] = np.eye(3)
        # path
        pelvis = p.rig.bone["pelvis"]
        p0 = p.joint(pelvis).copy()
        p0[2] = float(p.P[:, 2].min())
        f0 = p.heading() @ p.rig.forward
        f0[2] = 0.0
        f0 /= np.linalg.norm(f0)
        sub = soup.near(p0 - [MAX_RUN + 3, MAX_RUN + 3, 3.0], p0 + [MAX_RUN + 3, MAX_RUN + 3, 3.0])
        loop = plan_loop(sub, p0, f0)
        name = p.name
        loc = np.array(actors[name]["location"], dtype=np.float64)
        # Cycle time relative to look.unreal.walkers.period (a typical adult's):
        # a pendulum's, sqrt of the hip height (children step faster).
        period = math.sqrt(hip / hip_ref)
        phase0 = walk.phases.get(name, walk.phase_for(p))
        if loop is None:
            stuck.append(name)
            per_actor[name] = {"walker": [row * nb, row, phase0, 0.0, 0.0, 1.0, 0.0, 0.0], "extent_cm": 300.0}
            continue
        f, a, c, lane = loop
        pts, seg = loop_points(p0, f, a, c, lane)
        xy, yaw_b, L = resample(pts, seg, PATH_SAMPLES)
        hit = sub.cast(np.column_stack([xy, np.full(len(xy), p0[2] + 1.2)]),
                       np.broadcast_to([0.0, 0.0, -1.0], (len(xy), 3)))
        z = np.where(np.isfinite(hit), p0[2] + 1.2 - hit, p0[2])
        rel = np.column_stack([xy, z]) * [CM, -CM, CM] - loc     # Unreal cm, relative to the actor
        paths[row, :PATH_SAMPLES, :3] = rel
        paths[row, PATH_SAMPLES, :3] = rel[0]
        paths[row, :, 3] = -yaw_b                                 # Unreal's y flip mirrors angles
        ext = float(np.abs(rel).max()) + 250.0
        per_actor[name] = {"walker": [row * nb, row, round(phase0, 4), round(period, 4), round(stride * CM, 2),
                                      round(L * CM, 2), round(c * CM, 2), 1.0],
                           "extent_cm": round(ext, 1)}
        log(f"[people] {name}: {a + c:.1f} m loop, stride {stride:.2f} m, cycle x{period:.2f}")
    if stuck:
        log(f"[people] {len(stuck)} with no room to walk keep their pose: {stuck}")
    bone_range = float(math.ceil(np.abs(bones[..., 3]).max() / 50.0 + 1e-9) * 50.0) or 50.0
    rng = np.abs(paths).max((0, 1))
    path_range = [float(math.ceil(max(rng[0], rng[1]) / 100.0 + 1e-9) * 100.0)] * 2 + \
                 [float(math.ceil(rng[2] / 100.0 + 1e-9) * 100.0) or 100.0, float(math.ceil(rng[3] + 1e-9))]
    write_png16(out_dir / "bones.png", bones, [1.0, 1.0, 1.0, bone_range])
    write_png16(out_dir / "paths.png", paths, path_range)
    info = {"bones_png": str((out_dir / "bones.png").resolve()), "paths_png": str((out_dir / "paths.png").resolve()),
            "frames": FRAMES, "path_samples": PATH_SAMPLES, "bone_range": bone_range, "path_range": path_range,
            "count": len(walkers), "bone_rows": nb}
    return per_actor, info


def write_png16(path, data, ranges):
    """Signed values -> 16-bit RGBA PNG: (v / range + 1) / 2 per channel."""
    r = np.asarray(ranges, dtype=np.float64)
    u = np.clip((data / r + 1.0) / 2.0, 0.0, 1.0)
    img = np.round(u * 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), img[..., [2, 1, 0, 3]])


def read_png16(path, ranges):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float64)
    return (img / 65535.0 * 2.0 - 1.0) * np.asarray(ranges, dtype=np.float64)


@functools.lru_cache(maxsize=4)
def _texture(path, ranges, mtime):
    return read_png16(path, ranges)


def vertex_uvs(p, H=None):
    """UV1-6 of a walker's mesh: bone indices (1, 2), weights (3, 4), bind position in
    Unreal cm standing at the origin facing +X (5, 6)."""
    H = ch.frame(np.array([1.0, 0.0, 0.0]), p.rig) if H is None else H
    idx = np.zeros((len(p.idx), 4))
    w = np.zeros((len(p.idx), 4))
    k = min(4, p.idx.shape[1])
    idx[:, :k], w[:, :k] = p.idx[:, :k], p.w[:, :k]
    b = (UE @ H @ p.bind.T).T * CM
    return [idx[:, :2], idx[:, 2:], w[:, :2], w[:, 2:], b[:, :2], np.column_stack([b[:, 2], np.zeros(len(b))])]


# --------------------------------------------------------------------------- shader mirror

def replay(info, uvs, cpd, t, location, walk_period=1.1):
    """World positions (Unreal cm) of a walker's vertices at time t, as M_GTB_Walker
    computes them from the textures (keep in sync with gtb_hlsl.WALKER_WPO)."""
    bones = _texture(info["bones_png"], (1, 1, 1, info["bone_range"]), Path(info["bones_png"]).stat().st_mtime)
    paths = _texture(info["paths_png"], tuple(info["path_range"]), Path(info["paths_png"]).stat().st_mtime)
    row, prow, phase0, period, stride, loop, s0, _ = cpd
    F, N = info["frames"], info["path_samples"]
    idx = np.concatenate([uvs[0], uvs[1]], 1).astype(int)
    w = np.concatenate([uvs[2], uvs[3]], 1)
    b = np.column_stack([uvs[4], uvs[5][:, 0]])
    if period <= 0:
        return None
    prog = t / (period * walk_period)
    fk = ((phase0 + prog) % 1.0) * F
    f0 = int(np.floor(fk)) % F
    f1 = (f0 + 1) % F
    al = fk - np.floor(fk)
    p = np.zeros_like(b)
    for i in range(4):
        rows = (row + idx[:, i]).astype(int)
        M = (1 - al) * bones[rows, 3 * f0:3 * f0 + 3] + al * bones[rows, 3 * f1:3 * f1 + 3]   # (n, 3, 4)
        p += w[:, i, None] * (np.einsum("nij,nj->ni", M[..., :3], b) + M[..., 3])
    s = ((s0 + prog * stride) / loop % 1.0) * N
    i0 = int(np.floor(s))
    a2 = s - i0
    q = (1 - a2) * paths[int(prow), i0] + a2 * paths[int(prow), i0 + 1]
    c, sn = math.cos(q[3]), math.sin(q[3])
    rot = np.column_stack([c * p[:, 0] - sn * p[:, 1], sn * p[:, 0] + c * p[:, 1], p[:, 2]])
    return np.asarray(location) + q[:3] + rot
