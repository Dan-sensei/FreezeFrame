"""Walking people in Unreal (M_GTB_Walker, docs/WALKING_PEOPLE.md): the people a rip
caught upright walk up and down the city's streets (ue/routes.py), with the walk cycle
gtb/characters.py fits to them. The ones lying down stay as they are.

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

Where they walk: a street of the city, up and back on two lanes (ue/routes.py: a
walkable map from the rip's surfaces, keeping to the game's roads). The fallback, for
a person who can't reach the streets or a plan without the scene's surfaces, is a
straight lane: from where the game had the person, ahead and back as far as the ground
is clear (no wall, beam or crate at knee, chest or head height, no step over 30 cm, at
most MAX_RUN m each way), a second lane beside it, and half circles between them. Other
headings up to 90 degrees off are tried when the way ahead is short.
"""
import functools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from gtb import characters as ch
from ue import routes

FRAMES = 48           # cycle frames in bones.png (linearly blended)
PATH_SAMPLES = 512    # points along each loop (routes are up to ~150 m round)
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


def _turn(a, b):
    """The smallest rotation taking direction a to direction b."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    sn = np.linalg.norm(v)
    if sn < 1e-9:
        return np.eye(3)
    return ch.rodrigues(v / sn * math.atan2(sn, float(a @ b)))


def leg_ik(p, R, t, leg, target):
    """Two-bone IK, in place: the leg's ankle (its foot bone's joint) reaches `target`,
    the hip stays, the knee bends in the plane it had, the foot keeps its orientation."""
    th, ca, fo = leg
    J = p.joints
    hip = R[th] @ J[th] + t[th]
    knee = R[ca] @ J[ca] + t[ca]
    ank = R[fo] @ J[fo] + t[fo]
    l1, l2 = np.linalg.norm(knee - hip), np.linalg.norm(ank - knee)
    v = target - hip
    d = float(np.clip(np.linalg.norm(v), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    u = v / max(np.linalg.norm(v), 1e-12)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole = (knee - hip) - ((knee - hip) @ u) * u
    w = pole / max(np.linalg.norm(pole), 1e-12)
    knee2, ank2 = hip + a * u + h * w, hip + d * u
    Q1 = _turn(knee - hip, knee2 - hip)
    R[th] = Q1 @ R[th]
    t[th] = hip - R[th] @ J[th]
    Q2 = _turn(Q1 @ (ank - knee), ank2 - knee2)
    R[ca] = Q2 @ Q1 @ R[ca]
    t[ca] = knee2 - R[ca] @ J[ca]
    t[fo] = ank2 - R[fo] @ J[fo]


def foot_pivot(p, R, t, leg, sel, lock, reach):
    """A locked foot whose ankle is out of the leg's reach rolls about the part of it on the
    ground (the toes at push-off, the heel at landing), as real feet do. Returns the ankle
    target and the foot's new world rotation (in place in R; t follows in leg_ik)."""
    th, ca, fo = leg
    hip = R[th] @ p.joints[th] + t[th]
    ank = R[fo] @ p.joints[fo] + t[fo] + lock
    if np.linalg.norm(ank - hip) <= reach:
        return ank
    V = p.skin(R, t)[sel] + lock
    piv = V[V[:, 2] < V[:, 2].min() + 0.01].mean(0)           # the part of the foot on the ground
    best = None
    for sign in (1.0, -1.0):
        for deg in range(2, 61, 2):
            Q = ch.rodrigues(np.array([0.0, sign * math.radians(deg), 0.0]))   # about the walking frame's sideways axis
            a2 = piv + Q @ (ank - piv)
            if np.linalg.norm(a2 - hip) <= reach:
                if best is None or deg < best[0]:
                    best = (deg, Q, a2)
                break
    if best is None:
        return ank
    _, Q, a2 = best
    R[fo] = Q @ R[fo]
    return a2


def _planted(a, b, ground_a=0.0, ground_b=0.0):
    """The foot vertices that carry the person between two frames (a, b: the foot's
    vertices; ground_*: the ground's height in each): the foot's lowest point must be
    within 4 mm of the ground in both, and then its vertices within 1.5 cm of it count.
    Right after toe-off the game's toe stays within 1.5 cm of the ground; a looser test
    took that swinging foot for a planted one."""
    za, zb = a[:, 2] - ground_a, b[:, 2] - ground_b
    if za.min() > 0.004 or zb.min() > 0.004:
        return np.zeros(len(a), bool)
    return (za < 0.015) & (zb < 0.015)


def leg_timing(p, walk, H, legs, fine=192):
    """Per leg, how much its phase is shifted at each fine frame (re-timed): while the foot is
    on the ground, the leg's poses play faster or slower so the foot moves back evenly.
    The game's cycle (and our gait curves fitted to it) moves a planted foot back at
    1.4-4.1 m per cycle within one step: slow mid-step, fast before it lifts. A body at a
    steady speed then needed foot locks of 11-20 cm, and moving the body unevenly instead
    surged it. Re-timed, the poses are the same, only their timing within the step."""
    phis = np.arange(fine) / fine
    V = [p.skin(*p.fk(walk.pose(phi), H, np.zeros(3))) for phi in phis]
    low = np.array([-v[:, 2].min() for v in V])
    out = []
    for leg, sel in legs:
        step = np.zeros(fine)
        down = np.zeros(fine, bool)
        for i in range(fine):
            j = (i + 1) % fine
            a, b = V[i][sel] + [0, 0, low[i]], V[j][sel] + [0, 0, low[j]]
            on = _planted(a, b)
            if on.any():
                step[i] = max(-(b[on, 0] - a[on, 0]).mean(), 1e-5)
                down[i] = True
        timing = phis.copy()
        if down.any() and not down.all():
            # the longest run of frames down (cyclic), from heel strike to toe-off
            start = int(np.argmax(down & ~np.roll(down, 1)))
            idx = (start + np.arange(fine)) % fine
            n = int(np.argmin(down[idx])) or fine
            run = idx[:n]
            travel = np.concatenate([[0.0], np.cumsum(step[run])])      # template position -> travel
            even = np.linspace(0.0, travel[-1], n + 1)                 # time -> even travel
            pos = np.interp(even, travel, np.arange(n + 1))            # time -> template frame
            warped = (start + pos[:n]) / fine
            # ease the re-timing in and out over 8 frames, so the leg doesn't jerk at
            # heel strike and toe-off
            ease = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / 8.0, 0.0, 1.0)
            base = (start + np.arange(n)) / fine
            timing[run] = base + ease * (warped - base)
        out.append(((timing - phis + 0.5) % 1.0) - 0.5)          # as a small shift of the phase
    return out


SWAY = 0.025      # m each way: the body over the stance foot
BOB_HARMONICS = 4  # the body's rise and fall: the feet's, smoothed to this many harmonics


SWING_CLEAR = 0.05   # m: the swinging foot's lowest point at least this high mid-swing


def _foot_locks(p, legs, V, low, lift, sway, stride, phis):
    """Per leg and fine frame: how far its foot must move (world) so that its vertices on
    the ground stay put (eased back to 0 while the foot is up), and whether it is down.
    A swinging foot is also lifted to clear the ground by SWING_CLEAR * sin(pi * progress):
    the game's own swing passes 1-5 cm over the ground, and with the body a little lower
    than it expects, the toe skimmed along it."""
    fine = len(phis)
    body = np.column_stack([stride * phis, sway, lift])        # the root's way through the world
    lock = np.zeros((len(legs), fine, 3))
    down_at = np.zeros((len(legs), fine), bool)
    for f, (leg, sel) in enumerate(legs):
        world = [V[i][sel] + body[i] for i in range(fine)]
        for i in range(fine):
            j = (i + 1) % fine
            b = world[j] + ([stride, 0, 0] if j == 0 else 0)
            down_at[f, j] = _planted(world[i], b, lift[i] - low[i], lift[j] - low[j]).any()
        # swing progress 0..1 over each run of frames up
        prog = np.zeros(fine)
        up = ~down_at[f]
        if up.any() and not up.all():
            start = int(np.argmax(up & ~np.roll(up, 1)))
            idx = (start + np.arange(fine)) % fine
            k = 0
            while k < fine:
                if not up[idx[k]]:
                    k += 1
                    continue
                n = 1
                while k + n < fine and up[idx[k + n]]:
                    n += 1
                prog[idx[k:k + n]] = (np.arange(n) + 0.5) / n
                k += n
        corr = np.zeros(3)
        for _ in range(2):
            for i in range(fine):
                j = (i + 1) % fine
                a = world[i]
                b = world[j] + ([stride, 0, 0] if j == 0 else 0)
                down = _planted(a, b, lift[i] - low[i], lift[j] - low[j])
                if down.any():
                    corr = corr - (b[down] - a[down]).mean(0)
                    corr[2] = -float(b[down, 2].min())          # on the ground, not above or in it
                else:
                    corr = corr * 0.93                          # eased off over the swing
                    clear = SWING_CLEAR * math.sin(math.pi * prog[j]) if up[j] else 0.0
                    corr[2] = max(corr[2], clear - float(b[:, 2].min()))
                lock[f, j] = corr
        # A little time smoothing (+-2 frames of 192) of the height only, so locking on and
        # off doesn't pop; smoothing the pin along the ground too let the landing foot slip.
        lock[f, :, 2] = sum(np.roll(lock[f, :, 2], d) for d in range(-2, 3)) / 5.0
    return lock, down_at


def bake_cycle(p, walk, frames=FRAMES, fine=192, stride_scale=1.0):
    """Bone transforms of one walker over the cycle, in the walking frame (Blender axes,
    m: x forward, z up, ground 0). Returns (R (F,nb,3,3), t (F,nb,3), stride m, hip
    height m, H); H takes the bind axes to the walking frame.

    The body moves at a steady speed (the shader carries it along its path), sways SWAY m
    over the stance foot and rises and falls smoothly. Each foot is locked where it
    touched the ground, by two-bone IK on its leg, and eased back to the cycle's own
    motion while it swings. An earlier version planted the feet by moving the whole body
    unevenly instead: it surged between 1.3 and 3.8 m/s within every step."""
    H = ch.frame(np.array([1.0, 0.0, 0.0]), p.rig)
    legs = [(leg, _leg_vertices(p, leg)) for leg in p.rig.legs]
    legs = [(leg, v) for leg, v in legs if v is not None and leg[2] in p.joints]
    phis = np.arange(fine) / fine
    timing = leg_timing(p, walk, H, legs, fine)

    def pose_at(phi):
        return walk.pose(phi, [phi + float(np.interp(phi, phis, shift, period=1.0)) for shift in timing])

    poses = [p.fk(pose_at(phi), H, np.zeros(3)) for phi in phis]
    V = [p.skin(R, t) for R, t in poses]
    low = np.array([-v[:, 2].min() for v in V])                 # root height for the lowest vertex at 0
    # Stride: how fast a planted foot travels back. Per leg, the travel of its vertices on
    # the ground in both frames (the heel first, then the toes), over the frames it is down;
    # the body must move that fast for the foot to stay put. (Summing, frame by frame, the
    # faster of the two feet down counted the double-support frames twice: 2.35 m per cycle
    # instead of 1.5, and the body surged or the feet dragged to make up for it.)
    rates = []
    for leg, sel in legs:
        travel, frames_down = 0.0, 0
        for i in range(fine):
            j = (i + 1) % fine
            a, b = V[i][sel] + [0, 0, low[i]], V[j][sel] + [0, 0, low[j]]
            on = _planted(a, b)
            if on.any():
                travel += -(b[on, 0] - a[on, 0]).mean()
                frames_down += 1
        if frames_down:
            rates.append(travel / frames_down)
    stride = float(np.mean(rates)) * fine * stride_scale if rates else 0.0
    if stride < 0.2:
        stride = 2.0 * float(np.ptp([v[legs[0][1], 0].mean() for v in V]))
    # Smooth rise and fall (the feet's, a few harmonics), and a sway over the stance foot:
    # towards the first leg's side at its mid-stance (phase 0.3).
    c = np.linalg.lstsq(ch.basis(phis, BOB_HARMONICS), low, rcond=None)[0]
    lift = ch.basis(phis, BOB_HARMONICS) @ c
    side = np.sign(np.mean([v[legs[0][1], 1].mean() for v in V]) or 1.0)
    sway = side * SWAY * np.cos(2 * np.pi * (phis - 0.3))
    reach = []
    for leg, _ in legs:                                         # leg length, hip to ankle
        th, ca, fo = leg
        R0, t0 = poses[0]
        hip0, knee0, ank0 = (R0[b] @ p.joints[b] + t0[b] for b in (th, ca, fo))
        reach.append(np.linalg.norm(knee0 - hip0) + np.linalg.norm(ank0 - knee0) - 0.005)
    for _ in range(4):
        lock, down_at = _foot_locks(p, legs, V, low, lift, sway, stride, phis)
        # Where a locked foot is further than its leg reaches, lower the body (the legs bend
        # a little more as the stride opens); otherwise the feet lifted off: a hop each step.
        over = np.zeros(fine)
        for f, (leg, sel) in enumerate(legs):
            th, ca, fo = leg
            for i in range(fine):
                if not down_at[f, i]:                           # only a foot on the ground must reach it
                    continue
                R0, t0 = poses[i]
                R0, t0 = R0.copy(), t0 + [0.0, sway[i], lift[i]]
                hip = R0[th] @ p.joints[th] + t0[th]
                ank = foot_pivot(p, R0, t0, leg, sel, lock[f, i], reach[f])
                over[i] = max(over[i], np.linalg.norm(ank - hip) - reach[f])
        if over.max() <= 0.003:
            break
        need = np.clip(over + 0.005, 0.0, None)
        lift = lift - sum(np.roll(need, d) for d in range(-4, 5)) / 9.0 * 1.5
    lock, _ = _foot_locks(p, legs, V, low, lift, sway, stride, phis)
    Rs, ts = [], []
    for k in range(frames):
        phi = k / frames
        R, t = p.fk(pose_at(phi), H, np.zeros(3))
        off = np.array([0.0, float(np.interp(phi, phis, sway, period=1.0)),
                        float(np.interp(phi, phis, lift, period=1.0))])
        t = t + off
        for f, (leg, sel) in enumerate(legs):
            cf = np.array([np.interp(phi, phis, lock[f, :, d], period=1.0) for d in range(3)])
            if np.abs(cf).max() > 1e-4:
                leg_ik(p, R, t, leg, foot_pivot(p, R, t, leg, sel, cf, reach[f]))
        Rs.append(R)
        ts.append(t)
    return np.array(Rs), np.array(ts), stride, float(lift.mean()), H


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
    # Unwrapped round the closed loop: the last value is the first plus the loop's whole
    # turn (+-2 pi; a guessed sign spun walkers round once at the seam).
    yaw = np.unwrap(np.concatenate([np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[:1, 1], d[:1, 0])]))
    return xy, yaw, L


# --------------------------------------------------------------------------- plan

def start_of(p):
    """Where a person stands (pelvis over its lowest point) and its heading (xy), Blender m."""
    p0 = p.joint(p.rig.bone["pelvis"]).copy()
    p0[2] = float(p.P[:, 2].min())
    f0 = p.heading() @ p.rig.forward
    return p0, f0[:2] / np.linalg.norm(f0[:2])


def walk_map(walkers, ground, log=print):
    """The city's walkable map around the walkers (ue/routes.py), or None without ground.
    ground: the export's solid surfaces [(positions, tris, material, has_uv)]."""
    if not ground:
        return None
    t0 = time.time()
    starts = np.array([start_of(p)[0] for p in walkers])
    feet = [p.P[p.P[:, 2] < p.P[:, 2].min() + 0.05].mean(0) for p in walkers]
    roads, road_material = routes.road_meshes(ground, feet, log)
    lo = starts.min(0) - [routes.MARGIN, routes.MARGIN, 15.0]
    hi = starts.max(0) + [routes.MARGIN, routes.MARGIN, 15.0]
    wm = routes.WalkMap([(P, T, i in roads) for i, (P, T, *_) in enumerate(ground)], lo, hi)
    log(f"[people] walkable map: {len(wm.col)} floors ({wm.main.sum()} in the main network, "
        f"{(wm.main & wm.road).sum()} on roads), {time.time() - t0:.0f} s")
    wm.road_material = road_material
    return wm


def route_path(wm, soup, p0, f0, log=print):
    """A route through the city as PATH_SAMPLES points (xy, ground z) and tangent angles,
    or None (ue/routes.py)."""
    route = routes.plan_route(wm, p0, f0)
    if route is None:
        return None
    P, L, s0 = routes.route_points(wm, route, p0)
    seg = np.linalg.norm(np.diff(np.vstack([P[:, :2], P[:1, :2]]), axis=0), axis=1)
    xy, yaw, L = resample(P[:, :2], seg, PATH_SAMPLES)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    zf = np.interp(np.linspace(0, s[-1], PATH_SAMPLES, endpoint=False), s, np.r_[P[:, 2], P[0, 2]])
    sub = soup.near(np.r_[xy.min(0), zf.min()] - [1.0, 1.0, 2.0], np.r_[xy.max(0), zf.max()] + [1.0, 1.0, 2.0])
    hit = sub.cast(np.column_stack([xy, zf + 0.6]), np.broadcast_to([0.0, 0.0, -1.0], (len(xy), 3)))
    z = np.where(np.isfinite(hit) & (hit < 1.2), zf + 0.6 - hit, zf)
    k = max(1, int(round(0.5 / (L / PATH_SAMPLES))))             # +-0.5 m: no bobbing on ruts
    z = sum(np.roll(z, d) for d in range(-k, k + 1)) / (2 * k + 1)
    road = float(wm.road[route[0]].mean())
    return xy, z, yaw, L, s0, road


def plan(capture, manifest, walkers, walk, soup, actors, ground=None, log=print):
    """Bake every walker: (per actor name: custom primitive data and mesh bounds extension,
    textures info for plan.json). `actors`: {name: actor dict with location (Unreal cm)};
    `ground`: the solid surfaces with their materials, for routes through the city
    (without it, every walker gets a straight lane)."""
    out_dir = Path(capture) / "unreal" / "walkers"
    out_dir.mkdir(parents=True, exist_ok=True)
    wm = walk_map(walkers, ground, log)
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
        # path: a route through the city, else a straight lane
        p0, f0 = start_of(p)
        name = p.name
        loc = np.array(actors[name]["location"], dtype=np.float64)
        # Cycle time relative to look.unreal.walkers.period (a typical adult's):
        # a pendulum's, sqrt of the hip height (children step faster).
        period = math.sqrt(hip / hip_ref)
        phase0 = walk.phases.get(name, walk.phase_for(p))
        route = route_path(wm, soup, p0, f0, log) if wm is not None else None
        if route is not None:
            xy, z, yaw_b, L, s0, road = route
            what = f"{L / 2:.0f} m street up and back ({road * 100:.0f}% on roads)"
        else:
            f3 = np.array([f0[0], f0[1], 0.0])
            sub = soup.near(p0 - [MAX_RUN + 3, MAX_RUN + 3, 3.0], p0 + [MAX_RUN + 3, MAX_RUN + 3, 3.0])
            loop = plan_loop(sub, p0, f3)
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
            s0 = c                                                # the lane starts c m behind the person
            what = f"{a + c:.1f} m lane"
        rel = np.column_stack([xy, z]) * [CM, -CM, CM] - loc     # Unreal cm, relative to the actor
        paths[row, :PATH_SAMPLES, :3] = rel
        paths[row, PATH_SAMPLES, :3] = rel[0]
        paths[row, :, 3] = -yaw_b                                 # Unreal's y flip mirrors angles
        ext = float(np.abs(rel).max()) + 250.0
        per_actor[name] = {"walker": [row * nb, row, round(phase0, 4), round(period, 4), round(stride * CM, 2),
                                      round(L * CM, 2), round(s0 * CM, 2), 1.0],
                           "extent_cm": round(ext, 1)}
        log(f"[people] {name}: {what}, stride {stride:.2f} m, cycle x{period:.2f}")
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
            "count": len(walkers), "bone_rows": nb,
            "road_material": getattr(wm, "road_material", None) if wm is not None else None}
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
