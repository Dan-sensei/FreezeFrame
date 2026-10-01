"""A small reader for binary FBX skeleton animation (Mixamo downloads), without Blender.

    clip = fbx.skeleton_clip("animations/zombie_crawl.fbx")

Returns what gtb/clips.py reads: per bone its name and parent, and its rest and posed
head position and rotation in world space, Blender axes (z up, metres; FBX's Y-up and
centimetres are converted the way Blender's importer does), plus the frame rate.
Checked against Blender 5.2's import of Mixamo's zombie crawl: heads within 1e-5 m and
rotations within 1e-3 degrees of their rest (the difference clips.py uses), every frame.

Only binary FBX (7.x) is read; ASCII FBX is not. The node transform is FBX's own:
    T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
with R from Euler angles in the node's rotation order, and the animation curves sampled
linearly (Mixamo keys every frame).
"""
import struct
import zlib
from pathlib import Path

import numpy as np

TICKS = 46186158000                     # FBX time units per second
TIME_MODES = {1: 120, 2: 100, 3: 60, 4: 50, 5: 48, 6: 30, 7: 30, 8: 29.97, 9: 29.97, 10: 25, 11: 24, 12: 1000,
              13: 23.976, 15: 96, 16: 72, 17: 59.94, 18: 119.88}
# FBX (x, y, z) Y-up -> Blender (x, -z, y) Z-up, as Blender's importer (forward -Z, up Y)
AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


# --------------------------------------------------------------------------- binary records

class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        return next((c for c in self.children if c.name == name), None)

    def all(self, name):
        return [c for c in self.children if c.name == name]


def _prop(data, o):
    t = chr(data[o])
    o += 1
    scalar = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}
    if t in scalar:
        fmt = scalar[t]
        return struct.unpack_from(fmt, data, o)[0], o + struct.calcsize(fmt)
    if t in "fdlib":
        n, enc, size = struct.unpack_from("<III", data, o)
        o += 12
        raw = data[o:o + size]
        if enc == 1:
            raw = zlib.decompress(raw)
        dtype = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
        return np.frombuffer(raw, dtype=dtype, count=n).copy(), o + size
    if t in "SR":
        n = struct.unpack_from("<I", data, o)[0]
        o += 4
        raw = bytes(data[o:o + n])
        return (raw.decode("utf-8", "replace") if t == "S" else raw), o + n
    raise ValueError(f"unknown FBX property type {t!r} at {o - 1}")


def _node(data, o, wide):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", data, o)
        o += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", data, o)
        o += 12
    nlen = data[o]
    o += 1
    if end == 0:
        return None, o
    name = bytes(data[o:o + nlen]).decode("ascii", "replace")
    o += nlen
    props = []
    for _ in range(nprops):
        p, o = _prop(data, o)
        props.append(p)
    children = []
    while o < end:
        c, o = _node(data, o, wide)
        if c is None:
            break
        children.append(c)
    return Node(name, props, children), end


def parse(path):
    """The file's top-level records (a Node whose children they are)."""
    data = memoryview(Path(path).read_bytes())
    if bytes(data[:21]) != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{Path(path).name}: not a binary FBX (ASCII FBX isn't supported; download FBX Binary)")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    o, top = 27, []
    while o < len(data) - 13:
        n, o = _node(data, o, wide)
        if n is None:
            break
        top.append(n)
    return Node("", [], top)


def _props70(node):
    """{name: values} of a node's Properties70."""
    out = {}
    p70 = node.find("Properties70") if node is not None else None
    for p in (p70.all("P") if p70 is not None else []):
        out[p.props[0]] = p.props[4:]
    return out


# --------------------------------------------------------------------------- transforms

def _euler(deg, order=0):
    """FBX Euler angles (degrees) -> matrix. order: 0 XYZ, 1 XZY, 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX
    (the first axis is applied first)."""
    x, y, z = np.radians(np.asarray(deg, float))
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    R = {"X": Rx, "Y": Ry, "Z": Rz}
    seq = ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX"][int(order)]
    M = np.eye(3)
    for a in seq:                      # first axis applied first: M = R_last ... R_first
        M = R[a] @ M
    return M


def _affine(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def _local(p, T, R, S):
    """FBX node transform (4x4) from its properties and the current T, R (degrees), S."""
    v = lambda k, d: np.asarray(p.get(k, d)[:3], float)     # noqa: E731
    Roff, Rp = v("RotationOffset", (0, 0, 0)), v("RotationPivot", (0, 0, 0))
    Soff, Sp = v("ScalingOffset", (0, 0, 0)), v("ScalingPivot", (0, 0, 0))
    # Pre/post rotation and the rotation order count only with RotationActive (as in FBX
    # SDK and Blender's importer).
    active = bool(p.get("RotationActive", [0])[0])
    order = int(p.get("RotationOrder", [0])[0]) if active else 0
    Rpre = _euler(v("PreRotation", (0, 0, 0))) if active else np.eye(3)
    Rpost = _euler(v("PostRotation", (0, 0, 0))) if active else np.eye(3)
    M = _affine(t=T) @ _affine(t=Roff) @ _affine(t=Rp) @ _affine(Rpre) @ _affine(_euler(R, order)) \
        @ _affine(Rpost.T) @ _affine(t=-Rp) @ _affine(t=Soff) @ _affine(t=Sp) @ _affine(np.diag(S)) @ _affine(t=-Sp)
    return M


# --------------------------------------------------------------------------- scene

def skeleton_clip(path):
    """Bones (LimbNode/Null models) and their animation, as gtb/clips.py's data dict."""
    root = parse(path)
    objects, conns = root.find("Objects"), root.find("Connections")
    gs = _props70(root.find("GlobalSettings"))
    unit = float(gs.get("UnitScaleFactor", [1.0])[0]) / 100.0            # file units -> metres
    up = int(gs.get("UpAxis", [1])[0])
    if up != 1:
        raise ValueError(f"{Path(path).name}: up axis {up}; only Y-up FBX (Mixamo) is supported")
    # Property defaults for models (Definitions -> ObjectType Model -> PropertyTemplate).
    defaults = {}
    defs = root.find("Definitions")
    for ot in (defs.all("ObjectType") if defs is not None else []):
        if ot.props and ot.props[0] == "Model":
            tpl = ot.find("PropertyTemplate")
            defaults = _props70(tpl)
    models, curves, cnodes = {}, {}, {}
    for o in objects.children:
        oid = o.props[0]
        if o.name == "Model" and len(o.props) > 2 and o.props[2] in ("LimbNode", "Null", "Root"):
            name = o.props[1].split("\x00")[0]
            models[oid] = {"name": name, "p": dict(defaults, **_props70(o)), "parent": None, "anim": {}}
        elif o.name == "AnimationCurveNode":
            cnodes[oid] = {"name": o.props[1].split("\x00")[0], "default": _props70(o), "curves": {}}
        elif o.name == "AnimationCurve":
            t = o.find("KeyTime").props[0].astype(np.float64) / TICKS
            v = o.find("KeyValueFloat").props[0].astype(np.float64)
            curves[oid] = (t, v)
    for c in conns.all("C"):
        kind, child, parent = c.props[0], c.props[1], c.props[2]
        if kind == "OO" and child in models and parent in models:
            models[child]["parent"] = parent
        elif kind == "OP" and child in curves and parent in cnodes:
            cnodes[parent]["curves"][c.props[3]] = curves[child]
        elif kind == "OP" and child in cnodes and parent in models:
            models[parent]["anim"][c.props[3]] = cnodes[child]
    # Time range: the animation stack's, else the keys'.
    stack = next((o for o in objects.children if o.name == "AnimationStack"), None)
    sp = _props70(stack)
    keys = [t for t, _ in curves.values() if len(t)]
    t0 = sp["LocalStart"][0] / TICKS if "LocalStart" in sp else min(k[0] for k in keys)
    t1 = sp["LocalStop"][0] / TICKS if "LocalStop" in sp else max(k[-1] for k in keys)
    mode = int(gs.get("TimeMode", [0])[0])
    fps = float(gs["CustomFrameRate"][0]) if mode == 14 and "CustomFrameRate" in gs else TIME_MODES.get(mode)
    if not fps:
        dt = np.median(np.concatenate([np.diff(k) for k in keys if len(k) > 1]))
        fps = round(1.0 / dt, 3)
    n = int(round((t1 - t0) * fps)) + 1
    times = t0 + np.arange(n) / fps
    # Bones in parent-first order.
    ids = list(models)
    order, seen = [], set()

    def visit(i):
        if i in seen:
            return
        par = models[i]["parent"]
        if par is not None:
            visit(par)
        seen.add(i)
        order.append(i)
    for i in ids:
        visit(i)
    index = {i: k for k, i in enumerate(order)}

    def channel(m, prop, comp, t, base):
        cn = m["anim"].get(prop)
        if cn is None:
            return base
        cv = cn["curves"].get(f"d|{comp}")
        if cv is None:
            d = cn["default"].get(f"d|{comp}")
            return float(d[0]) if d else base
        kt, kv = cv
        return float(np.interp(t, kt, kv))

    def world(t):
        W = {}
        for i in order:
            m = models[i]
            p = m["p"]
            vals = {}
            for prop, d in (("Lcl Translation", (0, 0, 0)), ("Lcl Rotation", (0, 0, 0)), ("Lcl Scaling", (1, 1, 1))):
                base = [float(x) for x in p.get(prop, d)[:3]]
                vals[prop] = base if t is None else [channel(m, prop, c, t, b) for c, b in zip("XYZ", base)]
            L = _local(p, vals["Lcl Translation"], vals["Lcl Rotation"], vals["Lcl Scaling"])
            W[i] = W[m["parent"]] @ L if m["parent"] is not None else L
        return W

    def pose(t):
        W = world(t)
        heads = np.array([AXES @ W[i][:3, 3] * unit for i in order])
        rots = []
        for i in order:
            R = W[i][:3, :3]
            U, _, Vt = np.linalg.svd(R)                     # drop any scale
            rots.append(AXES @ (U @ Vt))
        return heads, np.array(rots)
    rest_head, rest_rot = pose(None)
    frames = [pose(t) for t in times]
    return {"names": np.array([models[i]["name"] for i in order]),
            "parents": np.array([index[models[i]["parent"]] if models[i]["parent"] in index else -1 for i in order]),
            "rest_head": rest_head, "rest_rot": rest_rot,
            "head": np.array([f[0] for f in frames]), "rot": np.array([f[1] for f in frames]), "fps": float(fps)}
