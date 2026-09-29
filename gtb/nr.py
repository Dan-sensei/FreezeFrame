"""Reader for Ninja Ripper 2 files (.nr meshes and the per-rip `ripdesc`).

Format (NRIP v3, little-endian): 16-byte header (magic 'NRIP', version, 2x
reserved), then chunks {u32 rawSize incl. 12-byte header, u32 tag, u32 idx}.
A GEOM chunk describes one draw call and references VERT/VATR/INDX/TXTR/SHD1/
PROP chunks. VERT/VATR index 0 is the draw's base stage, index 1 (if present)
the "extra" stage: pre-VS input for a post-VS mesh and vice versa.
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x5049524E


def tag(s):
    return struct.unpack("<I", s.encode("ascii"))[0]


T_GEOM, T_VERT, T_VATR, T_INDX = tag("GEOM"), tag("VERT"), tag("VATR"), tag("INDX")
T_TXTR, T_SHD1, T_PROP = tag("TXTR"), tag("SHD1"), tag("PROP")
P_PROP, P_RSTT = tag("PROP"), tag("RSTT")

TOPOLOGY = {1: "triangles", 2: "points", 3: "lines"}
STAGE = {1: "pre_vs", 2: "vs", 3: "ds_gs"}
# vaType -> numpy dtype
VA_DTYPE = {0: "<f4", 1: "<u4", 2: "<u2", 3: "u1", 4: "<i4", 5: "<i2", 6: "i1"}


@dataclass
class Attribute:
    semantic: str
    index: int
    va_type: int
    offset: int
    comps: int

    @property
    def name(self):
        return f"{self.semantic}{self.index}"


@dataclass
class Stream:
    """One vertex stream (a stage's vertex data plus its layout)."""
    count: int
    stride: int
    raw: bytes
    attrs: list

    def find(self, semantic, index=None):
        for a in self.attrs:
            if a.semantic.upper() == semantic.upper() and (index is None or a.index == index):
                return a
        return None

    def read(self, attr: Attribute):
        """Raw values as (count, comps) array in the stored type."""
        dt = np.dtype(VA_DTYPE[attr.va_type])
        view = np.frombuffer(self.raw, dtype=np.uint8).reshape(self.count, self.stride)
        cols = view[:, attr.offset:attr.offset + dt.itemsize * attr.comps]
        return np.ascontiguousarray(cols).view(dt).reshape(self.count, attr.comps)


@dataclass
class Draw:
    file: Path
    topology: str
    stage: str
    draw_call: int
    instances: int
    rip_id: int
    group0: int
    group1: int
    frame: int
    streams: list = field(default_factory=list)   # [base, extra?]
    indices: np.ndarray | None = None
    textures: list = field(default_factory=list)  # [(u0, u1, filename)]
    shaders: list = field(default_factory=list)
    props: dict = field(default_factory=dict)     # {subtag_str: {name: value}}

    @property
    def render_state(self):
        return self.props.get("RSTT", {})

    def stream_for(self, stage):
        """Stream holding data for 'pre_vs' or post-transform ('vs'/'ds_gs')."""
        if not self.streams:
            return None
        base_is_pre = self.stage == "pre_vs"
        want_pre = stage == "pre_vs"
        idx = 0 if base_is_pre == want_pre else 1
        return self.streams[idx] if idx < len(self.streams) else None


class _Reader:
    def __init__(self, data: bytes):
        self.d = data

    def u32(self, off):
        return struct.unpack_from("<I", self.d, off)[0]

    def cstr(self, off):
        end = self.d.index(b"\0", off)
        return self.d[off:end].decode("utf-8", "replace"), end + 1


def _parse_props(r, off):
    sub, count = struct.unpack_from("<II", r.d, off)
    off += 8
    values = {}
    for _ in range(count):
        name, off = r.cstr(off)
        typ, ln = struct.unpack_from("<HH", r.d, off)
        off += 4
        raw = r.d[off:off + ln]
        off += ln
        values[name] = struct.unpack("<I", raw)[0] if typ == 1 and ln == 4 else raw.decode("utf-8", "replace").rstrip("\0")
    return struct.pack("<I", sub).decode("ascii", "replace"), values


def read_nr(path: Path):
    data = Path(path).read_bytes()
    r = _Reader(data)
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an NRIP file")
    if version < 3:
        raise ValueError(f"{path}: NRIP v{version} is too old (need Ninja Ripper >= 2.0.5)")

    chunks = {}  # tag -> [body offset]
    pos = 16
    while pos + 12 <= len(data):
        size, t, _idx = struct.unpack_from("<III", data, pos)
        if size < 12:
            raise ValueError(f"{path}: corrupt chunk at {pos}")
        chunks.setdefault(t, []).append((pos + 12, pos + size))
        pos += size

    draws = []
    for body, _end in chunks.get(T_GEOM, []):
        flags, n, dc, inst, rip, g0, g1, frame = struct.unpack_from("<8I", data, body)
        refs = [struct.unpack_from("<II", data, body + 32 + 8 * i) for i in range(n)]
        d = Draw(Path(path), TOPOLOGY.get(flags & 3, "unknown"), STAGE.get((flags >> 2) & 7, "unknown"),
                 dc, inst, rip, g0, g1, frame)
        verts, layouts = [], []
        for t, i in refs:
            off, end = chunks[t][i]
            if t == T_VERT:
                count, stride = struct.unpack_from("<II", data, off)
                verts.append((count, stride, data[off + 8:off + 8 + count * stride]))
            elif t == T_VATR:
                cnt = r.u32(off)
                o = off + 4
                attrs = []
                for _ in range(cnt):
                    sem, o = r.cstr(o)
                    si, vt, ao, cc = struct.unpack_from("<4I", data, o)
                    o += 16
                    attrs.append(Attribute(sem, si, vt, ao, cc))
                layouts.append(attrs)
            elif t == T_INDX:
                count = r.u32(off)
                d.indices = np.frombuffer(data, "<u4", count, off + 8).astype(np.int64)
            elif t == T_TXTR:
                cnt = r.u32(off)
                o = off + 4
                for _ in range(cnt):
                    a, b = struct.unpack_from("<II", data, o)
                    name, o = r.cstr(o + 8)
                    d.textures.append((a, b, name))
            elif t == T_SHD1:
                cnt = r.u32(off)
                o = off + 4
                for _ in range(cnt):
                    h, o = r.cstr(o)
                    d.shaders.append(h)
            elif t == T_PROP:
                sub, vals = _parse_props(r, off)
                d.props[sub] = vals
        d.streams = [Stream(c, s, raw, a) for (c, s, raw), a in zip(verts, layouts)]
        draws.append(d)
    return draws


def read_ripdesc(rip_dir: Path):
    """{'executable', 'width', 'height', 'gapi', ...} or {} if missing."""
    p = Path(rip_dir) / "ripdesc"
    if not p.exists():
        return {}
    data = p.read_bytes()
    r = _Reader(data)
    pos = 16
    out = {}
    while pos + 12 <= len(data):
        size, t, _ = struct.unpack_from("<III", data, pos)
        if t == T_PROP:
            _, vals = _parse_props(r, pos + 12)
            out.update(vals)
        pos += max(size, 12)
    return out


def default_output_dir():
    """Ninja Ripper's configured output folder, if it can be found."""
    import os
    import re
    xml = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "ninjaripper" / "nrcommon.xml"
    if xml.exists():
        m = re.search(r'<output_dir\s+val\s*=\s*"([^"]*)"', xml.read_text("utf-8", "replace"))
        if m and m.group(1):
            return m.group(1)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\black_ninja\NR2common") as k:
            return winreg.QueryValueEx(k, "outDir")[0]
    except OSError:
        return None
