"""Writes a synthetic Ninja Ripper 2 frame rip (NRIP v3) with a known camera, plus
a matching capture folder, so the whole pipeline can be tested without a game.

  python tests/make_fake_rip.py <out_dir>

Scene (D3D left-handed, Y up): ground plane, rotated/scaled boxes, a "skinned"
(non-rigid) box, plus junk a real rip contains: a UI quad, a shadow-map pass and
a depth prepass duplicate. Camera: fov_y 45 deg, 16:9, pitched down 30 deg.
"""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1280, 720
FOV_Y = 45.0
PITCH = 30.0
CAM_POS = np.array([0.0, 8.0, -18.0])


def tag(s):
    return struct.unpack("<I", s.encode())[0]


def cstr(s):
    return s.encode() + b"\0"


class NrWriter:
    def __init__(self):
        self.chunks = []      # (tag, idx, body)
        self.counts = {}

    def add(self, t, body):
        idx = self.counts.get(t, 0)
        self.counts[t] = idx + 1
        self.chunks.append((tag(t), idx, body))
        return (tag(t), idx)

    def vert(self, arrays):
        """arrays: list of (semantic, sem_idx, np array (n,c), va_type)."""
        n = len(arrays[0][2])
        cols, attrs, off = [], [], 0
        for sem, si, a, vt in arrays:
            b = np.ascontiguousarray(a).view(np.uint8).reshape(n, -1)
            cols.append(b)
            attrs.append(cstr(sem) + struct.pack("<4I", si, vt, off, a.shape[1]))
            off += b.shape[1]
        data = np.hstack(cols).tobytes()
        v = self.add("VERT", struct.pack("<II", n, off) + data)
        a = self.add("VATR", struct.pack("<I", len(attrs)) + b"".join(attrs))
        return v, a

    def geom(self, stage, refs, draw_id):
        flags = 1 | (stage << 2)
        body = struct.pack("<8I", flags, len(refs), draw_id, 1, 0, 0, 0, 0)
        body += b"".join(struct.pack("<II", *r) for r in refs)
        self.add("GEOM", body)

    def save(self, path):
        out = struct.pack("<4I", 0x5049524E, 3, 0, 0)
        for t, i, body in self.chunks:
            out += struct.pack("<3I", 12 + len(body), t, i) + body
        Path(path).write_bytes(out)


def props(sub, values):
    body = struct.pack("<II", tag(sub), len(values))
    for k, v in values.items():
        if isinstance(v, int):
            body += cstr(k) + struct.pack("<HHI", 1, 4, v)
        else:
            b = v.encode()
            body += cstr(k) + struct.pack("<HH", 2, len(b)) + b
    return body


def view_matrix():
    p = math.radians(PITCH)
    fwd = np.array([0, -math.sin(p), math.cos(p)])
    right = np.array([1.0, 0, 0])
    up = np.cross(fwd, right)
    up = -up if up[1] < 0 else up
    Rc = np.stack([right, up, fwd], 1)  # camera axes in world (columns)
    return Rc


def project(world):
    Rc = view_matrix()
    v = (world - CAM_POS) @ Rc  # row vectors -> view space
    p11 = 1 / math.tan(math.radians(FOV_Y) / 2)
    p00 = p11 / (W / H)
    n, f = 0.1, 1000.0
    return np.stack([v[:, 0] * p00, v[:, 1] * p11, v[:, 2] * f / (f - n) - n * f / (f - n), v[:, 2]], 1)


def box():
    P, N, U, I = [], [], [], []
    faces = [((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)),
             ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0))]
    for n, u in faces:
        n, u = np.array(n, float), np.array(u, float)
        t = np.cross(n, u)
        base = len(P)
        for s, (a, b) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            P.append(n + a * t + b * u)
            N.append(n)
            U.append(((a + 1) / 2, (1 - b) / 2))
        I += [base, base + 2, base + 1, base, base + 3, base + 2]  # clockwise front (D3D)
    return np.array(P), np.array(N), np.array(U), np.array(I, np.uint32)


def _blocks(ch):
    h, w = ch.shape[:2]
    return ch.reshape(h // 4, 4, w // 4, 4, *ch.shape[2:]).swapaxes(1, 2).reshape(h // 4 * (w // 4), 16, *ch.shape[2:])


def _bc4(ch):
    """ch: (H,W) uint8 -> BC4 blocks (8 bytes each), 8-value interpolation mode."""
    out = bytearray()
    for b in _blocks(ch):
        e0, e1 = int(b.max()), int(b.min())
        if e0 == e1:
            e0 = min(e1 + 1, 255) if e1 < 255 else e0
            e1 = e0 - 1 if e0 == e1 else e1
        pal = [e0, e1] + [((7 - i) * e0 + i * e1) // 7 for i in range(1, 7)]
        idx = np.abs(b[:, None].astype(int) - np.array(pal)[None]).argmin(1)
        bits = 0
        for i, v in enumerate(idx):
            bits |= int(v) << (3 * i)
        out += bytes([e0, e1]) + bits.to_bytes(6, "little")
    return out


def _bc1(rgb):
    out = bytearray()
    for b in _blocks(rgb).astype(int):
        lum = b @ np.array([3, 6, 1])
        c0, c1 = b[lum.argmax()], b[lum.argmin()]
        q = lambda c: ((c[0] >> 3) << 11) | ((c[1] >> 2) << 5) | (c[2] >> 3)
        v0, v1 = q(c0), q(c1)
        if v0 == v1:
            v0 = min(v0 + 1, 0xFFFF) if v0 < 0xFFFF else v0
            v1 = v0 - 1 if v0 == v1 else v1
        if v0 < v1:
            v0, v1, c0, c1 = v1, v0, c1, c0
        pal = np.array([c0, c1, (2 * c0 + c1) // 3, (c0 + 2 * c1) // 3])
        idx = ((b[:, None] - pal[None]) ** 2).sum(-1).argmin(1)
        bits = 0
        for i, v in enumerate(idx):
            bits |= int(v) << (2 * i)
        out += struct.pack("<HHI", v0, v1, bits)
    return out


def write_dds(path, img, fmt):
    """Minimal DDS writer: fmt 'BC1' (DXT1 fourcc) or 'BC5' (DX10 header)."""
    h, w = img.shape[:2]
    img = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    if fmt == "BC1":
        data, fourcc, pitch = _bc1(img[..., :3]), b"DXT1", w * h // 2
    else:
        r, g = _bc4(img[..., 0]), _bc4(img[..., 1])
        data = b"".join(r[i:i + 8] + g[i:i + 8] for i in range(0, len(r), 8))
        fourcc, pitch = b"DX10", w * h
    hdr = struct.pack("<7I", 124, 0x81007, h, w, pitch, 0, 1) + bytes(44)
    hdr += struct.pack("<2I4s5I", 32, 0x4, fourcc, 0, 0, 0, 0, 0) + struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    if fourcc == b"DX10":
        hdr += struct.pack("<5I", 83, 3, 0, 1, 0)
    Path(path).write_bytes(b"DDS " + hdr + bytes(data))


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])


SHOT_TRIS = []  # (world tri (3,3), rgb) for the reference "screenshot"


def write_draw(path, draw_id, local, normals, uv, idx, model, textures, rstt, post_fn=None, color=None):
    w = NrWriter()
    world = local @ model[:3, :3] + model[3, :3]
    if color is not None:
        SHOT_TRIS.extend((world[idx[i:i + 3]], color) for i in range(0, len(idx), 3))
    clip = project(world)
    if post_fn:
        clip = post_fn(clip)
    n8 = np.hstack([(normals * 0.5 + 0.5) * 255, np.zeros((len(normals), 1))]).round().astype(np.uint8)
    uvh = uv.astype(np.float16).view(np.uint16)
    post = w.vert([("SV_Position", 0, clip.astype(np.float32), 0), ("TEXCOORD", 0, uv.astype(np.float32), 0)])
    pre = w.vert([("POSITION", 0, local.astype(np.float32), 0), ("NORMAL", 0, n8, 3), ("TEXCOORD", 0, uvh, 2)])
    ind = w.add("INDX", struct.pack("<II", len(idx), 1) + idx.astype(np.uint32).tobytes())
    tx = w.add("TXTR", struct.pack("<I", len(textures)) + b"".join(struct.pack("<II", i, 3) + cstr(t) for i, t in enumerate(textures)))
    sh = w.add("SHD1", struct.pack("<I", 2) + cstr("vs_1234") + cstr("ps_abcd"))
    pr = w.add("PROP", props("RSTT", rstt))
    w.geom(2, [post[0], post[1], pre[0], pre[1], ind, tx, sh, pr], draw_id)
    w.save(path)


def main(out):
    out = Path(out)
    rip = out / "ripper" / "2026-09-29_12-00-00_Frostpunk"
    rip.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:128, 0:128] / 128.0
    base = np.stack([.6 + .1 * np.sin(xx * 30), .45 + .1 * np.sin(xx * 30), .35 + .05 * np.sin(xx * 30)], -1)
    bricks = np.clip(base * (0.8 + 0.2 * (np.sin(yy * 50) > 0))[..., None] + rng.normal(0, .02, base.shape), 0, 1)
    Image.fromarray((np.dstack([bricks, np.ones((128, 128))]) * 255).astype(np.uint8)).save(rip / "tex_0000.dds", pixel_format="BC7")
    nx, ny = .25 * np.cos(xx * 40), .25 * np.cos(yy * 40)
    nrm = np.dstack([nx * .5 + .5, ny * .5 + .5, np.zeros_like(nx), np.ones_like(nx)])
    write_dds(rip / "tex_0001.dds", nrm, "BC5")
    snow = np.clip(.88 + rng.normal(0, .03, (128, 128, 3)), 0, 1)
    write_dds(rip / "tex_0002.dds", snow, "BC1")
    write_dds(rip / "tex_0003.dds", np.full((4, 4, 3), .5), "BC1")

    solid = {"DEPTH_ENABLE": 1, "DEPTH_WRITE_ENABLE": 1, "RENDERED_TO_RENDERTARGET": 1}
    P, N, U, I = box()
    draw = 0
    placements = [((0, 1.5, 0), 20, 1.5), ((5, 2.5, 6), -35, 2.5), ((-6, 1, 4), 60, 1.0), ((2, 0.75, -6), 10, 0.75)]
    for (t, ang, s) in placements:
        M = np.eye(4)
        M[:3, :3] = rot_y(ang) * s
        M[3, :3] = t
        write_draw(rip / f"mesh_{draw:04d}.nr", draw, P, N, U, I, M, ["tex_0000.dds", "tex_0001.dds", "tex_0003.dds"], solid,
                   color=(.6, .45, .35))
        draw += 1
    # Depth prepass duplicate of box 0 (no textures) -> should be deduplicated.
    M = np.eye(4)
    M[:3, :3] = rot_y(20) * 1.5
    M[3, :3] = (0, 1.5, 0)
    write_draw(rip / f"mesh_{draw:04d}.nr", draw, P, N, U, I, M, [], solid)
    draw += 1
    # Shadow map pass (colour writes disabled) -> skipped.
    write_draw(rip / f"mesh_{draw:04d}.nr", draw, P, N, U, I, M, [], dict(solid, RGBWRITE_DISABLED=1))
    draw += 1
    # "Skinned" box: post-VS positions bent, so it is not an affine image of pre-VS.
    M2 = np.eye(4)
    M2[:3, :3] = np.eye(3) * 1.2
    M2[3, :3] = (-3, 1.2, -2)
    bend = lambda c: c + np.stack([np.sin(c[:, 1] * 3) * .3, 0 * c[:, 0], 0 * c[:, 0], 0 * c[:, 0]], 1)
    write_draw(rip / f"mesh_{draw:04d}.nr", draw, P, N, U, I, M2, ["tex_0000.dds"], solid, bend)
    draw += 1
    # Ground: 60x60 plane made of a grid (flat -> not used for the FOV solve).
    g = np.linspace(-30, 30, 11)
    gx, gz = np.meshgrid(g, g)
    GP = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    GU = np.stack([gx.ravel() / 6, gz.ravel() / 6], 1)
    GI = []
    for r in range(10):
        for c in range(10):
            a = r * 11 + c
            GI += [a, a + 11, a + 1, a + 1, a + 11, a + 12]
    write_draw(rip / f"mesh_{draw:04d}.nr", draw, GP, np.tile([0, 1, 0], (len(GP), 1)).astype(float), GU,
               np.array(GI, np.uint32), np.eye(4), ["tex_0002.dds"], solid, color=(.88, .88, .88))
    draw += 1
    # UI quad: already in clip space with w == 1 -> skipped.
    w = NrWriter()
    q = np.array([[-.9, .9, 0, 1], [-.5, .9, 0, 1], [-.5, .7, 0, 1], [-.9, .7, 0, 1]], np.float32)
    v = w.vert([("SV_Position", 0, q, 0)])
    ind = w.add("INDX", struct.pack("<II", 6, 1) + np.array([0, 1, 2, 0, 2, 3], np.uint32).tobytes())
    w.geom(2, [v[0], v[1], ind], draw)
    w.save(rip / f"mesh_{draw:04d}.nr")

    rd = NrWriter()
    rd.add("PROP", props("PROP", {"rippingType": 1, "gapi": 1, "gapi_majorVer": 11, "gapi_minorVer": 0,
                                  "executable": "Frostpunk.exe", "width": W, "height": H}))
    rd.add("GEOM", struct.pack("<8I", 0, 0, 0, 0, 0, 0, 0, 0))
    rd.save(rip / "ripdesc")

    cap = out / "capture"
    cap.mkdir(exist_ok=True)
    render_reference(cap / "screenshot.png")
    (cap / "capture.json").write_text(json.dumps({"game_exe": "Frostpunk.exe", "resolution": [W, H],
                                                  "rip_dir": str(rip)}, indent=1))
    print(f"fake rip: {rip}\ncapture: {cap}")


def render_reference(path):
    """Painter's-algorithm flat-shaded raster of the true scene from the true camera."""
    import cv2
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (0.62, 0.7, 0.8)
    light = np.array([0.4, 0.8, -0.45])
    light /= np.linalg.norm(light)
    items = []
    for tri, col in SHOT_TRIS:
        c = project(tri)
        if (c[:, 3] <= 0.1).any():
            continue
        px = np.stack([(c[:, 0] / c[:, 3] * .5 + .5) * W, (.5 - c[:, 1] / c[:, 3] * .5) * H], 1)
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n /= np.linalg.norm(n) + 1e-9
        if n @ (CAM_POS - tri.mean(0)) < 0:
            n = -n
        shade = 0.35 + 0.65 * max(n @ light, 0)
        items.append((c[:, 3].mean(), px, np.array(col) * shade))
    for _, px, col in sorted(items, key=lambda x: -x[0]):
        cv2.fillPoly(img, [np.round(px * 16).astype(np.int32)], tuple(float(x) for x in col), cv2.LINE_AA, 4)
    cv2.imwrite(str(path), (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)[..., ::-1])


if __name__ == "__main__":
    main(sys.argv[1])
