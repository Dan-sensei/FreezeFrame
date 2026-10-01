"""People in a frame rip: their skeletons, recovered from the skinned draws, and a
walk cycle fitted to the ones the rip caught mid-stride.

A skinned draw keeps its bind pose (pre-VS POSITION, BLENDINDICES, BLENDWEIGHT)
and its posed result (post-VS). Ninja Ripper saves no bone matrices, but linear
blend skinning, P = sum_k w_k (R_k b + t_k), is linear in them, so each bone's
rigid transform is solved from the vertices (Gauss-Newton, robust weights).
Frostpunk: 25 people, every vertex within 1 cm.

Pitfalls (Frostpunk, Ninja Ripper 2.18):
- The bind pose is D3D (left-handed): mirror z first, or every bone comes out as a
  reflection and no rotation fits.
- The 8-bit weights sum to 0.996-1.000. The game skins in model space, where that
  doesn't matter; solved around world positions 150 m from the origin it moved
  vertices by 0.5 m. Weights are normalised.
- The game scales each person (1.3 for most, 1.0 for some): solved per person.

The rig (bone names, parents, legs) is in the game profile ("characters"); the
cycle's shape comes from normative gait curves (hip, knee, ankle) whose timing,
scale and offset are fitted to the walkers, and the rest of the body (arms,
spine, head) is a first-harmonic fit to them. On Frostpunk the 11 walkers sit at
four points of the stride; a free two-harmonic fit placed them wrongly.
"""
import numpy as np

# --------------------------------------------------------------------------- rotations


def skew(v):
    v = np.asarray(v, dtype=np.float64)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return K - np.swapaxes(K, -1, -2)


def rodrigues(r):
    r = np.asarray(r, dtype=np.float64)
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    K = skew(r / np.maximum(th, 1e-12))
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def logrot(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    if th > np.pi - 1e-4:                       # near 180 degrees: axis from the symmetric part
        w, v = np.linalg.eigh((R + R.T) / 2)
        return v[:, np.argmax(w)] * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th


def kabsch(A, B, w):
    """Rotation R, translation t with B ~ A @ R.T + t, weighted."""
    ca = (A * w[:, None]).sum(0) / w.sum()
    cb = (B * w[:, None]).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((A - ca) * w[:, None]).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def lbs(bind, idx, w, R, t):
    out = np.zeros_like(bind)
    for k in range(idx.shape[1]):
        out += w[:, k, None] * (np.einsum("nij,nj->ni", R[idx[:, k]], bind) + t[idx[:, k]])
    return out


# --------------------------------------------------------------------------- skin data

def skin_arrays(pre, rows):
    """Bind-pose data of a skinned draw for its npz (process.py), or {}.
    pre: the pre-VS stream, rows: its row for each output vertex."""
    if pre is None or rows is None:
        return {}
    a_p, a_i, a_w = pre.find("POSITION"), pre.find("BLENDINDICES"), pre.find("BLENDWEIGHT")
    if a_p is None or a_i is None or a_w is None or a_p.va_type != 0:
        return {}
    idx = pre.read(a_i)
    w = pre.read(a_w).astype(np.float64)
    if a_w.va_type == 3:                         # UNORM bytes
        w = w / 255.0
    if idx.shape[1] != w.shape[1]:
        return {}
    return {"skin_bind": pre.read(a_p)[rows, :3].astype(np.float32),
            "skin_index": idx[rows].astype(np.int16), "skin_weight": w[rows].astype(np.float32)}


def backfill(capture, manifest, rip_dir):
    """Add the skin arrays to the npz files of a capture processed before process.py
    kept them. Needs the rip's .nr files. Returns the number of meshes updated."""
    from pathlib import Path
    from gtb import nr
    from gtb.process import DrawData
    n = 0
    for m in manifest["meshes"]:
        if not any(a.startswith("BLENDINDICES") for a in m.get("layout_pre", [])):
            continue
        f = Path(capture) / m["file"]
        data = dict(np.load(f))
        if "skin_bind" in data:
            continue
        src = Path(rip_dir) / m["source"]
        if not src.exists():
            continue
        draws = [d for d in nr.read_nr(src) if f"{src.stem}_{d.draw_call}" == m["name"]] or nr.read_nr(src)[-1:]
        dd = DrawData(draws[-1])
        skin = skin_arrays(dd.pre, dd.rows)
        if not skin or len(skin["skin_bind"]) != len(data["positions"]):
            continue
        data.update(skin)
        np.savez_compressed(f, **data)
        n += 1
    return n


# --------------------------------------------------------------------------- rig

class Rig:
    """profile["characters"]: bone names and parents, the two legs (thigh, calf, foot),
    left/right pairs, and the bind pose's up and forward axes (D3D object space)."""

    def __init__(self, spec):
        self.spec = spec
        self.names = {int(k): v[0] for k, v in spec["bones"].items()}
        self.parent = {int(k): int(v[1]) for k, v in spec["bones"].items() if int(v[1]) >= 0}
        self.root = next(b for b in self.names if b not in self.parent)
        order, todo = [self.root], [self.root]
        while todo:
            p = todo.pop(0)
            kids = sorted(b for b, q in self.parent.items() if q == p)
            order += kids
            todo += kids
        self.order = order
        self.legs = [tuple(l) for l in spec["legs"]]
        self.mirror = {}
        for a, b in spec.get("mirror", []):
            self.mirror[a], self.mirror[b] = b, a
        self.bone = {v: k for k, v in self.names.items()}
        flip = np.diag([1.0, 1.0, -1.0])          # D3D left-handed -> right-handed
        self.up = flip @ np.asarray(spec.get("bind_up", [0, 1, 0]), dtype=np.float64)
        self.forward = flip @ np.asarray(spec.get("bind_forward", [0, 0, 1]), dtype=np.float64)
        self.right = np.cross(self.forward, self.up)
        # Mirror plane: left-right axis.
        self.lateral = self.right / np.linalg.norm(self.right)

    def mirror_rotvec(self, r):
        """Rotation vector of the mirrored joint (reflection across the body's mid-plane)."""
        n = self.lateral
        M = np.eye(3) - 2 * np.outer(n, n)
        return -(M @ r)


def rig_from_profile(profile):
    spec = (profile or {}).get("characters")
    return Rig(spec) if spec and spec.get("bones") else None


# --------------------------------------------------------------------------- skeleton

def solve_bones(bind, idx, w, P, iters=25):
    """Per-bone rigid (R, t) with P ~ sum_k w_k (R_k bind + t_k). bind is right-handed and
    already scaled. Kabsch per bone (weights^4) to start, then Gauss-Newton with
    Cauchy-style robust weights. Solved around the mesh's centroid."""
    o = P.mean(0)
    Q = P - o
    nb = int(idx.max()) + 1
    n = len(bind)
    wb = np.zeros((n, nb))
    for k in range(idx.shape[1]):
        np.add.at(wb, (np.arange(n), idx[:, k]), w[:, k])
    R = np.tile(np.eye(3), (nb, 1, 1))
    t = np.zeros((nb, 3))
    used = np.where((wb > 0).sum(0) >= 3)[0]
    for b in used:
        R[b], t[b] = kabsch(bind, Q, wb[:, b] ** 4 + 1e-12)
    cols = {b: i for i, b in enumerate(used)}
    m = len(used)
    for it in range(iters):
        pred = lbs(bind, idx, w, R, t)
        r = (pred - Q)
        res = np.linalg.norm(r, axis=1)
        scale = max(np.median(res) * 3.0, 1e-4)
        rw = 1.0 / (1.0 + (res / scale) ** 2)
        J = np.zeros((n, 3, m * 6))
        for k in range(idx.shape[1]):
            for b in used:
                sel = (idx[:, k] == b) & (w[:, k] > 0)
                if not sel.any():
                    continue
                c = cols[b]
                Rb = np.einsum("ij,nj->ni", R[b], bind[sel])
                J[sel, :, c * 6:c * 6 + 3] += -w[sel, k, None, None] * skew(Rb)
                J[sel, :, c * 6 + 3:c * 6 + 6] += w[sel, k, None, None] * np.eye(3)
        Jf = J.reshape(n * 3, m * 6)
        W = np.repeat(rw, 3)
        A = Jf.T @ (Jf * W[:, None]) + 1e-6 * np.eye(m * 6)
        g = Jf.T @ (W * r.ravel())
        dx = -np.linalg.solve(A, g).reshape(m, 6)
        for b in used:
            c = cols[b]
            R[b] = rodrigues(dx[c, :3]) @ R[b]
            t[b] += dx[c, 3:]
        if np.abs(dx).max() < 1e-7:
            break
    t = t + o
    res = np.linalg.norm(lbs(bind, idx, w, R, t) - P, axis=1)
    return R, t, res


def person_scale(bind, idx, w, P):
    """The game's scale for this person, from the bone with the most vertices bound to it alone."""
    best = None
    for b in np.unique(idx[:, 0]):
        s = (idx[:, 0] == b) & (w[:, 0] > 0.95)
        if best is None or s.sum() > best.sum():
            best = s
    if best is None or best.sum() < 4:
        return 1.0
    A, B = bind[best], P[best]
    return float(np.sqrt(((B - B.mean(0)) ** 2).sum() / max(((A - A.mean(0)) ** 2).sum(), 1e-12)))


class Person:
    """One skinned person: bind pose (right-handed, scaled), weights, solved bones."""

    def __init__(self, name, data, rig):
        self.name, self.rig = name, rig
        w = data["skin_weight"].astype(np.float64)
        self.w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
        self.idx = data["skin_index"].astype(np.int64)
        self.w[self.idx < 0] = 0.0
        self.idx = np.maximum(self.idx, 0)
        raw = data["skin_bind"].astype(np.float64) * [1.0, 1.0, -1.0]
        self.P = data["positions"].astype(np.float64)
        self.scale = person_scale(raw, self.idx, self.w, self.P)
        self.bind = raw * self.scale
        self.R, self.t, self.res = solve_bones(self.bind, self.idx, self.w, self.P)
        self.nb = len(self.R)
        self.bones = {int(b) for b in np.unique(self.idx[self.w > 0])}
        self.joints = self._bind_joints()

    def _bind_joints(self):
        """Bind-space position of each bone's joint with its parent: the point both bones
        move to the same place in the captured pose, (R_p - R_b) j = t_b - t_p. One pose
        leaves it free along the joint's turning axis, so it is pulled towards the
        centroid of the vertices the bones share. The centroid alone was up to 7 cm off
        at the hips, and the feet of a re-posed walker hung in the air."""
        J = {}
        n = len(self.bind)
        wb = {}
        for b in set(self.rig.names) | self.bones:
            wb[b] = np.zeros(n)
            for k in range(self.idx.shape[1]):
                wb[b] += self.w[:, k] * (self.idx[:, k] == b)
        for b in self.rig.order:
            if b not in self.bones:
                continue
            p = self.rig.parent.get(b)
            sh = np.minimum(wb[b], wb[p]) if p is not None and p in wb else np.zeros(n)
            if sh.sum() > 1e-6:
                j0 = (self.bind * sh[:, None]).sum(0) / sh.sum()
            else:
                ww = wb[b] ** 4
                j0 = (self.bind * ww[:, None]).sum(0) / max(ww.sum(), 1e-12)
            if p is not None and p in self.bones:
                A = self.R[p] - self.R[b]
                J[b] = np.linalg.solve(A.T @ A + 0.01 * np.eye(3), A.T @ (self.t[b] - self.t[p]) + 0.01 * j0)
            else:
                J[b] = j0
        return J

    def joint(self, b, R=None, t=None):
        R = self.R if R is None else R
        t = self.t if t is None else t
        return R[b] @ self.joints[b] + t[b]

    def has(self, *names):
        return all(self.rig.bone.get(n) in self.bones for n in names)

    def upright(self):
        """Torso within ~45 degrees of vertical (the others lie on the ground)."""
        top = next((n for n in ("neck", "head", "chest") if self.has(n)), None)
        if top is None or not self.has("pelvis"):
            return False
        v = self.joint(self.rig.bone[top]) - self.joint(self.rig.bone["pelvis"])
        return bool(v[2] / max(np.linalg.norm(v), 1e-9) > 0.7)

    def heading(self):
        """World rotation H taking the bind axes to the walking frame: forward
        horizontal (pelvis and chest forward averaged), up = world z."""
        f = np.zeros(3)
        for n in ("pelvis", "chest", "spine"):
            b = self.rig.bone.get(n)
            if b in self.bones:
                f += self.R[b] @ self.rig.forward
        f[2] = 0.0
        f /= max(np.linalg.norm(f), 1e-9)
        return frame(f, self.rig)

    def local_pose(self, H=None):
        """{bone: rotation vector}: the root relative to H, every other bone to its parent."""
        H = self.heading() if H is None else H
        out = {}
        for b in self.rig.order:
            if b not in self.bones:
                continue
            p = self.rig.parent.get(b)
            if p is None:
                out[b] = logrot(H.T @ self.R[b])
            elif p in self.bones:
                out[b] = logrot(self.R[p].T @ self.R[b])
        return out

    def fk(self, pose, H, root):
        """Bone transforms for a local pose (as local_pose), with the root joint at `root`."""
        R = np.tile(np.eye(3), (self.nb, 1, 1))
        t = np.zeros((self.nb, 3))
        for b in self.rig.order:
            if b not in self.joints:
                continue
            L = rodrigues(pose[b]) if b in pose else np.eye(3)
            p = self.rig.parent.get(b)
            if p is None or p not in self.joints:
                R[b] = H @ L
                t[b] = root - R[b] @ self.joints[b]
            else:
                R[b] = R[p] @ L
                t[b] = R[p] @ self.joints[b] + t[p] - R[b] @ self.joints[b]
        # Bones outside the rig follow their nearest rigged ancestor (none on Frostpunk).
        return R, t

    def skin(self, R, t):
        return lbs(self.bind, self.idx, self.w, R, t)


def frame(forward, rig):
    """Rotation taking the rig's bind axes (forward, up, right) to (forward, world up, right)."""
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    src = np.stack([rig.forward, rig.up, rig.right], 1)
    dst = np.stack([forward, up, right], 1)
    return dst @ np.linalg.inv(src)


def find_people(capture, manifest, rig, log=print):
    """Every skinned, human-sized, non-rigid mesh with skin data, solved."""
    from pathlib import Path
    spec = rig.spec
    lo, hi = spec.get("bind_height", [0.8, 3.0])
    people = []
    for m in manifest["meshes"]:
        if not any(a.startswith("BLENDINDICES") for a in m.get("layout_pre", [])):
            continue
        if (m.get("rigid_fit") or 0.0) < spec.get("min_rigid_fit", 0.01) or m.get("category") != "surface":
            continue
        data = dict(np.load(Path(capture) / m["file"]))
        if "skin_bind" not in data:
            continue
        h = float(np.ptp(data["skin_bind"][:, 1]))
        nb = len(np.unique(data["skin_index"][data["skin_weight"] > 0]))
        if not lo <= h <= hi or nb < spec.get("min_bones", 16) or int(data["skin_index"].max()) >= 64:
            continue
        if np.ptp(data["positions"], 0).max() > hi * 2:
            continue
        if not {rig.bone.get("pelvis")} <= set(np.unique(data["skin_index"]).tolist()):
            continue
        p = Person(m["name"], data, rig)
        if np.percentile(p.res, 99) > spec.get("max_fit_error", 0.03):
            log(f"[people] {m['name']}: skinning fit off by {np.percentile(p.res, 99):.3f} m, skipped")
            continue
        people.append(p)
    return people


# --------------------------------------------------------------------------- walk cycle

# Normative sagittal gait (degrees vs. fraction of the cycle from this leg's heel
# strike; Perry/Winter-style averages): hip flexion, knee flexion, ankle plantarflexion.
# The loading knee bump is a little higher than the textbook's 18: Frostpunk's
# walkers land with 21-28.
GAIT_POINTS = {
    "hip": [(0, 30), (.1, 28), (.2, 20), (.3, 10), (.4, 0), (.5, -9), (.55, -10), (.6, -5), (.7, 10), (.8, 25),
            (.87, 32), (.95, 31)],
    "knee": [(0, 4), (.05, 14), (.12, 22), (.2, 19), (.3, 9), (.38, 5), (.45, 7), (.55, 25), (.62, 42), (.7, 58),
             (.75, 60), (.8, 52), (.9, 22), (.97, 5)],
    "ankle": [(0, 0), (.07, 7), (.15, 1), (.3, -6), (.45, -10), (.55, 0), (.62, 15), (.7, 10), (.8, 0), (.9, -2),
              (.97, 0)],
}


def basis(phi, K):
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    cols = [np.ones_like(phi)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * phi), np.sin(2 * np.pi * k * phi)]
    return np.stack(cols, 1)


def _periodic(points, K=6):
    x = np.array([p for p, _ in points])
    y = np.array([v for _, v in points], dtype=np.float64)
    xx = np.linspace(0, 1, 400, endpoint=False)
    yy = np.interp(xx, np.concatenate([x, [1 + x[0]]]), np.concatenate([y, [y[0]]]))
    c = np.linalg.lstsq(basis(xx, K), yy, rcond=None)[0]
    return lambda phi: basis(np.asarray(phi, dtype=np.float64) % 1.0, K) @ c


GAIT = {k: _periodic(v) for k, v in GAIT_POINTS.items()}
# Leg joint flexion -> the local rotation about the lateral axis (sign per joint).
FLEX_SIGN = {"hip": 1.0, "knee": -1.0, "ankle": -1.0}


class Walk:
    """The fitted cycle: per-curve scale/offset of the gait template (leg flexion about
    the lateral axis), a first-harmonic fit of everything else, and each walker's phase
    (of the first leg) at capture time."""

    def __init__(self, rig, a, b, body, bones, phases, rms):
        self.rig, self.a, self.b, self.body, self.bones = rig, a, b, body, bones
        self.phases, self.rms = phases, rms

    def phase_for(self, person):
        """Phase of the cycle closest to a person's legs (for walkers outside the fit)."""
        ax = self.rig.lateral
        q = person.local_pose()
        grid = np.linspace(0, 1, 200, endpoint=False)
        e = np.zeros(len(grid))
        for leg, off in zip(self.rig.legs, (0.0, 0.5)):
            for bone, curve, k in zip(leg, ("hip", "knee", "ankle"), range(3)):
                if bone in q:
                    e += (self.a[k] * GAIT[curve](grid + off) + self.b[k] - _flex(q, bone, ax)) ** 2
        return float(grid[np.argmin(e)])

    def pose(self, phi, legs=None):
        """The pose at cycle phase phi. legs: each leg's own phase in the cycle (default phi;
        ue/walkers.py re-times them so a planted foot moves back evenly)."""
        K = (self.body.shape[0] - 1) // 2
        v = (basis(phi, K) @ self.body)[0]
        p = {b: v[i * 3:i * 3 + 3].copy() for i, b in enumerate(self.bones)}
        ax = self.rig.lateral
        mean = {b: self.body[0, i * 3:i * 3 + 3] for i, b in enumerate(self.bones)}
        for n, (leg, off) in enumerate(zip(self.rig.legs, (0.0, 0.5))):
            u = phi if legs is None else legs[n]
            for bone, curve, k in zip(leg, ("hip", "knee", "ankle"), range(3)):
                if bone not in p:
                    continue
                ang = np.radians(self.a[k] * GAIT[curve](u + off)[0] + self.b[k])
                # Flexion from the template; the leg's twist and sideways swing stay at the
                # walkers' average: fitted, they slid the planted foot sideways at 80 cm/s.
                p[bone] = mean[bone] - ax * (mean[bone] @ ax) + ax * ang
        return p


def _flex(pose, bone, ax):
    return np.degrees(pose[bone] @ ax) if bone in pose else np.nan


def fit_walk(walkers, rig, log=print):
    """Phases of the walkers and the cycle's shape (see the module doc)."""
    ax = rig.lateral
    # People missing a leg bone (Frostpunk's amputees, a peg leg on the thigh) limp:
    # they walk with the cycle but stay out of the fit.
    full = [p for p in walkers if all(b in p.bones for leg in rig.legs for b in leg)]
    walkers = full if len(full) >= 3 else walkers
    poses = [p.local_pose() for p in walkers]
    bones = [b for b in rig.order if any(b in q for q in poses)]
    # Leg flexion observations (degrees): leg 0 then leg 1, (hip, knee, ankle).
    obs = np.array([[_flex(q, b, ax) for leg in rig.legs for b in leg] for q in poses])
    S = len(poses)
    grid = np.linspace(0, 1, 400, endpoint=False)
    curves = ("hip", "knee", "ankle")
    G = {c: [GAIT[c](grid + off) for off in (0.0, 0.5)] for c in curves}
    best = None
    for start in np.linspace(0, 1, 8, endpoint=False):
        phi = np.full(S, start)
        a, b = np.array([FLEX_SIGN[c] for c in curves]), np.zeros(3)
        for _ in range(30):
            for s in range(S):
                e = np.zeros(len(grid))
                for leg in range(2):
                    for k, c in enumerate(curves):
                        o = obs[s, leg * 3 + k]
                        if not np.isnan(o):
                            e += (a[k] * G[c][leg] + b[k] - o) ** 2 * (0.5 if c == "ankle" else 1.0)
                phi[s] = grid[np.argmin(e)]
            for k, c in enumerate(curves):
                X, Y = [], []
                for s in range(S):
                    for leg, off in ((0, 0.0), (1, 0.5)):
                        o = obs[s, leg * 3 + k]
                        if not np.isnan(o):
                            X.append([GAIT[c](phi[s] + off)[0], 1.0])
                            Y.append(o)
                if len(X) >= 3:
                    a[k], b[k] = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]
        pred = np.array([[a[k] * GAIT[c](phi[s] + off)[0] + b[k] for off in (0.0, 0.5) for k, c in enumerate(curves)]
                         for s in range(S)])
        rms = float(np.sqrt(np.nanmean((pred - obs) ** 2)))
        if best is None or rms < best[3]:
            best = (phi.copy(), a.copy(), b.copy(), rms)
    phi, a, b, rms = best
    # Phase 0 = heel strike of the first leg (template phase 0).
    # Everything else: first harmonic, mirrored samples half a cycle later.
    D = len(bones) * 3

    def vec(q):
        v = np.full(D, np.nan)
        for i, bn in enumerate(bones):
            if bn in q:
                v[i * 3:i * 3 + 3] = q[bn]
        return v

    def mirrored(q):
        return {rig.mirror.get(bn, bn): rig.mirror_rotvec(r) for bn, r in q.items()}

    Y = np.array([vec(q) for q in poses] + [vec(mirrored(q)) for q in poses])
    Bm = basis(np.concatenate([phi, phi + 0.5]), 1)
    body = np.zeros((3, D))
    for d in range(D):
        ok = ~np.isnan(Y[:, d])
        if ok.sum() >= 3:
            body[:, d] = np.linalg.lstsq(Bm[ok], Y[ok, d], rcond=None)[0]
        elif ok.any():
            body[0, d] = Y[ok, d].mean()
    log(f"[people] walk cycle from {S} walkers: legs within {rms:.1f} deg rms "
        f"(hip x{a[0]:.2f} {b[0]:+.0f}, knee x{a[1]:.2f} {b[1]:+.0f}, ankle x{a[2]:.2f} {b[2]:+.0f})")
    return Walk(rig, a, b, body, bones, {p.name: float(f) for p, f in zip(walkers, phi)}, rms)


def default_walk(rig):
    """The plain template, when a rip has fewer than 3 walkers to fit."""
    return Walk(rig, np.array([FLEX_SIGN[c] for c in ("hip", "knee", "ankle")]), np.zeros(3),
                np.zeros((3, len(rig.order) * 3)), list(rig.order), {}, float("nan"))
