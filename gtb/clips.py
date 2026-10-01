"""Animation clips for the people (docs/WALKING_PEOPLE.md, "Clips"): a humanoid FBX with
Mixamo's bone names (a Mixamo download, "Without Skin") retargeted onto a person's
solved bones (gtb/characters.py), in the same form as the baked walk cycle, so the
walker material plays it along the person's route.

The clips live in animations/ (<name>.fbx, out of git: Mixamo's licence doesn't allow
sharing the files). Blender reads each FBX once (blender/fbx_clip.py) into <name>.npz.

Retargeting: each rig bone copies its Mixamo bone's turn away from Mixamo's rest pose,
in world space (both rests stand upright, facing forward). The rests differ in the
limbs (Frostpunk binds in an A-pose with the forearms forward, Mixamo in a T-pose), so
the upper arms, forearms, thighs and calves are first turned to point where Mixamo's
do at rest, and the hands, fingers and feet follow their parent's turn. The body is
scaled by hip height. The clip's travel over a cycle becomes the stride the shader
carries the person along its route by; what's left of the hips' motion stays in the
bones."""
import math
import subprocess
from pathlib import Path

import numpy as np

from gtb import characters as ch

ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = ROOT / "animations"

# rig bone name -> Mixamo bone (the names the profile's rig uses, docs/WALKING_PEOPLE.md)
MIXAMO = {"pelvis": "Hips", "spine": "Spine", "chest": "Spine2", "neck": "Neck", "head": "Head"}
for _s, _m in (("l", "Left"), ("r", "Right")):
    MIXAMO.update({f"clavicle_{_s}": f"{_m}Shoulder", f"upperarm_{_s}": f"{_m}Arm",
                   f"forearm_{_s}": f"{_m}ForeArm", f"hand_{_s}": f"{_m}Hand", f"fingers_{_s}": f"{_m}HandMiddle1",
                   f"thigh_{_s}": f"{_m}UpLeg", f"calf_{_s}": f"{_m}Leg", f"foot_{_s}": f"{_m}Foot"})
# Limb bones whose rest direction is matched first: bone -> the joint it points to.
AIM = {"upperarm": "forearm", "forearm": "hand", "thigh": "calf", "calf": "foot"}
LOOP_DEG = 5.0      # the last frame repeats the first (within this many degrees): a loop


def turn(a, b):
    """The smallest rotation taking direction a to direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-9:
        return np.eye(3)
    return ch.rodrigues(v / s * math.atan2(s, c))


def slerp(R0, R1, al):
    return R0 @ ch.rodrigues(al * ch.logrot(R0.T @ R1))


class Clip:
    """One clip in the walking frame (x the way it travels, z up, metres)."""

    def __init__(self, name, d):
        self.name = name
        self.bones = [str(n).split(":")[-1] for n in d["names"]]
        self.i = {n: k for k, n in enumerate(self.bones)}
        self.fps = float(d["fps"])
        rest_head, rest_rot = d["rest_head"], d["rest_rot"]
        head, rot = d["head"], d["rot"]
        # Facing: up is Blender's z, left from the hips; forward = left x up.
        up = np.array([0.0, 0.0, 1.0])
        left = rest_head[self.i["LeftUpLeg"]] - rest_head[self.i["RightUpLeg"]]
        left = left - (left @ up) * up
        left /= np.linalg.norm(left)
        S = np.stack([np.cross(left, up), left, up])
        # A loop's last frame repeats its first; travel is measured over the whole cycle.
        mapped = [self.i[m] for m in MIXAMO.values() if m in self.i]
        diff = max(math.degrees(np.linalg.norm(ch.logrot(rot[0, k].T @ rot[-1, k]))) for k in mapped)
        self.loop = len(rot) > 2 and diff < LOOP_DEG
        hips = self.i["Hips"]
        if self.loop:
            adv = head[-1, hips] - head[0, hips]
            head, rot = head[:-1], rot[:-1]
        else:
            adv = (head[-1, hips] - head[0, hips]) * len(head) / max(len(head) - 1, 1)
        # Travel along x: turn the clip so its travel over a cycle points forward.
        a = S @ adv
        if math.hypot(a[0], a[1]) > 0.2:
            c, s = a[0] / math.hypot(a[0], a[1]), a[1] / math.hypot(a[0], a[1])
            S = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ S
        self.advance = S @ adv
        self.frames = len(rot)
        self.duration = self.frames / self.fps
        self.rest_head = rest_head @ S.T
        self.rest_rot = np.einsum("ij,njk->nik", S, rest_rot)
        self.head = head @ S.T
        self.rot = np.einsum("ij,fnjk->fnik", S, rot)

    def at(self, u):
        """(rotation per bone index (dict, mapped bones only), hips position) at frame u
        (fractional, wraps for a loop), the cycle's steady travel taken out."""
        F = self.frames
        if self.loop:
            u = u % F
            f0 = int(math.floor(u))
            f1, al = (f0 + 1) % F, u - f0
        else:
            u = min(max(u, 0.0), F - 1.0)
            f0 = min(int(math.floor(u)), F - 2)
            f1, al = f0 + 1, u - f0
        R = {k: slerp(self.rot[f0, k], self.rot[f1, k], al) for k in {self.i[m] for m in MIXAMO.values()
                                                                    if m in self.i}}
        hips = self.i["Hips"]
        h0, h1 = self.head[f0, hips], self.head[f1, hips]
        if self.loop and f1 == 0:
            h1 = h1 + self.advance
        h = h0 + al * (h1 - h0) - self.head[0, hips] - self.advance * (u / F)
        return R, h + [0.0, 0.0, self.head[0, hips][2]]


def read(fbx, blender_exe):
    """The clip in an FBX, read by Blender once (cached next to it as .npz)."""
    fbx = Path(fbx)
    npz = fbx.with_suffix(".npz")
    if not npz.exists() or npz.stat().st_mtime < fbx.stat().st_mtime:
        cmd = [blender_exe, "-b", "--factory-startup", "--python", str(ROOT / "blender" / "fbx_clip.py"), "--",
               str(fbx), str(npz)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not npz.exists() or "Traceback" in r.stdout + r.stderr:
            raise RuntimeError(f"Blender couldn't read {fbx.name}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return Clip(fbx.stem, dict(np.load(npz)))


def library(blender_exe, clip_dir=CLIP_DIR, log=print):
    """Every clip in animations/ (sorted by name), skipping files that aren't Mixamo-like."""
    out = []
    for f in sorted(Path(clip_dir).glob("*.fbx")):
        try:
            c = read(f, blender_exe)
        except (RuntimeError, KeyError) as e:
            log(f"[clips] {f.name} skipped: {e}")
            continue
        out.append(c)
    return out


def hip_height(p, H):
    """The pelvis joint's height over the lowest point of the bind pose (m)."""
    pel = p.rig.bone["pelvis"]
    return float((H @ p.joints[pel])[2] - (p.bind @ H.T)[:, 2].min())


def retarget(p, clip, frames):
    """Bone transforms of person p playing the clip, in the walking frame (as
    walkers.bake_cycle): (R (F,nb,3,3), t (F,nb,3), stride m per cycle, cycle s, H)."""
    rig = p.rig
    H = ch.frame(np.array([1.0, 0.0, 0.0]), rig)
    k = hip_height(p, H) / float(clip.rest_head[clip.i["Hips"]][2] - clip.rest_head[:, 2].min())
    src, A = {}, {}
    for b in rig.order:
        name = rig.names[b]
        par = rig.parent.get(b)
        m = MIXAMO.get(name)
        if m in clip.i and b in p.joints:
            src[b] = clip.i[m]
        kind = name.rsplit("_", 1)[0]
        child = rig.bone.get(name.replace(kind, AIM[kind], 1)) if kind in AIM else None
        cm = MIXAMO.get(rig.names.get(child, ""))
        if b in src and child in p.joints and cm in clip.i:
            A[b] = turn(H @ (p.joints[child] - p.joints[b]), clip.rest_head[clip.i[cm]] - clip.rest_head[src[b]])
        else:
            A[b] = A.get(par, np.eye(3))
    Rs, ts = [], []
    for f in range(frames):
        Q, hips = clip.at(f / frames * clip.frames)
        R = np.tile(np.eye(3), (p.nb, 1, 1))
        t = np.zeros((p.nb, 3))
        for b in rig.order:
            if b not in p.joints:
                continue
            par = rig.parent.get(b)
            if b in src:
                i = src[b]
                R[b] = Q[i] @ clip.rest_rot[i].T @ A[b] @ H
            elif par is not None and par in p.joints:
                R[b] = R[par]
            else:
                R[b] = H
            if par is None or par not in p.joints:
                t[b] = hips * k - R[b] @ p.joints[b]
            else:
                t[b] = R[par] @ p.joints[b] + t[par] - R[b] @ p.joints[b]
        Rs.append(R)
        ts.append(t)
    R, t = np.array(Rs), np.array(ts)
    # Ground: the lowest point on it in every frame. Scaled by hip height alone, the
    # body's other proportions differ: Frostpunk's bulkier people floated up to 12 cm
    # in the flat part of Mixamo's crawl when its push-up frames touched the ground.
    t[:, :, 2] -= np.array([p.skin(R[f], t[f])[:, 2].min() for f in range(frames)])[:, None]
    stride = float(np.linalg.norm(clip.advance[:2])) * k
    return R, t, stride, clip.duration, H
