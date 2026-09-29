"""Texture conversion (DDS -> PNG/EXR) and role classification.

Games only bind textures to numbered slots; nothing says "this is the albedo".
We guess from the compression format and the pixel statistics, and game
profiles (profiles/*.json) can override the guess per slot.
"""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# DXGI format ids we care about (subset of the enum).
DXGI_NAMES = {
    2: "R32G32B32A32_FLOAT", 10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM",
    24: "R10G10B10A2_UNORM", 26: "R11G11B10_FLOAT", 28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB",
    49: "R8G8_UNORM", 61: "R8_UNORM", 65: "A8_UNORM",
    71: "BC1_UNORM", 72: "BC1_UNORM_SRGB", 74: "BC2_UNORM", 75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM", 78: "BC3_UNORM_SRGB", 80: "BC4_UNORM", 81: "BC4_SNORM",
    83: "BC5_UNORM", 84: "BC5_SNORM", 95: "BC6H_UF16", 96: "BC6H_SF16",
    87: "B8G8R8A8_UNORM", 91: "B8G8R8A8_UNORM_SRGB", 98: "BC7_UNORM", 99: "BC7_UNORM_SRGB",
}
FOURCC_NAMES = {b"DXT1": "BC1_UNORM", b"DXT3": "BC2_UNORM", b"DXT5": "BC3_UNORM",
                b"ATI1": "BC4_UNORM", b"BC4U": "BC4_UNORM", b"ATI2": "BC5_UNORM", b"BC5U": "BC5_UNORM"}


def dds_info(path: Path):
    with open(path, "rb") as f:
        head = f.read(148)
    if head[:4] != b"DDS ":
        return None
    height, width = struct.unpack_from("<II", head, 12)
    mips = struct.unpack_from("<I", head, 28)[0]
    fourcc = head[84:88]
    caps2 = struct.unpack_from("<I", head, 112)[0]
    fmt, array = None, 1
    if fourcc == b"DX10":
        dxgi, dim, misc, array = struct.unpack_from("<IIII", head, 128)
        fmt = DXGI_NAMES.get(dxgi, f"DXGI_{dxgi}")
        cube = bool(misc & 0x4)
    else:
        fmt = FOURCC_NAMES.get(fourcc, fourcc.decode("latin1").strip("\0") or "UNCOMPRESSED")
        cube = bool(caps2 & 0x200)
    return {"format": fmt, "width": width, "height": height, "mips": max(mips, 1),
            "cube": cube, "array": array, "srgb": fmt.endswith("_SRGB")}


def load_image(path: Path):
    """Decode to a float32 HxWxC array in 0..1 (HDR formats may exceed 1)."""
    img = Image.open(path)
    img.load()
    if img.mode in ("RGBA", "RGB", "L", "LA"):
        arr = np.asarray(img, dtype=np.float32) / 255.0
    else:  # BC6H and float formats come through as "RGB;16F"-like modes or F
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return img, arr


def classify(arr: np.ndarray, info: dict | None):
    """Return (role, details) for a decoded texture."""
    fmt = (info or {}).get("format", "")
    h, w = arr.shape[:2]
    if h <= 8 and w <= 8:
        return "constant", {}
    if info and info.get("cube"):
        return "environment", {}
    if fmt.startswith("BC6H") or "FLOAT" in fmt:
        return "hdr", {}
    if fmt.startswith("BC5") or fmt == "R8G8_UNORM":
        return "normal", {"channels": "RG"}
    if fmt.startswith("BC4") or arr.shape[2] == 1 or fmt in ("R8_UNORM", "A8_UNORM"):
        return "gray", {}

    step = max(1, max(h, w) // 128)
    s = arr[::step, ::step]
    rgb = s[..., :3]
    mean = rgb.reshape(-1, 3).mean(0)
    std = rgb.reshape(-1, 3).std(0)
    alpha = s[..., 3] if s.shape[2] == 4 else None

    # Tangent-space normal in RGB: R,G centred on 0.5, B high.
    if abs(mean[0] - .5) < .08 and abs(mean[1] - .5) < .08 and mean[2] > .7:
        return "normal", {"channels": "RGB"}
    # Two-channel normal (BC5) saved uncompressed: R,G centred on 0.5, blue flat.
    if abs(mean[0] - .5) < .08 and abs(mean[1] - .5) < .08 and std[2] < .02 and std[0] > .01:
        return "normal", {"channels": "RG"}
    # DXT5nm: X in alpha, Y in green, red/blue constant.
    if alpha is not None and std[0] < .02 and std[2] < .02 and abs(alpha.mean() - .5) < .1 \
            and abs(mean[1] - .5) < .1:
        return "normal", {"channels": "AG"}

    # Normal packed in A+G with other masks in R/B (Frostpunk, many DXT5nm-era
    # engines): G and A centred on 0.5 and both varying.
    if alpha is not None and abs(mean[1] - .5) < .12 and abs(alpha.mean() - .5) < .1             and alpha.std() > .03 and std[1] > .03:
        return "normal", {"channels": "AG"}
    # The same packing on a nearly flat surface: G and A pinned at 0.5 (the flat
    # normal) while the masks in R/B vary. Frostpunk's frosted-planks set had its
    # normal map (t0018) taken for the albedo, which painted walls green/orange.
    if alpha is not None and abs(mean[1] - .5) < .05 and abs(alpha.mean() - .5) < .05 \
            and std[1] < .03 and alpha.std() < .03 and max(std[0], std[2]) > .03:
        return "normal", {"channels": "AG"}

    # A channel pinned at 0 or 1 everywhere while others vary is a packed mask
    # (e.g. R=metal, G unused, B=AO), never a real colour texture.
    pinned = [(std[i] < .005 and (mean[i] < .01 or mean[i] > .99)) for i in range(3)]
    if any(pinned) and not all(pinned) and max(std) > .03:
        return "packed", {}

    chroma = np.abs(rgb - rgb.mean(-1, keepdims=True)).mean()
    if chroma < .015:
        return "gray", {}
    # Channels that are individually varied but mutually uncorrelated look like
    # packed masks (e.g. R=metal, G=rough, B=AO) rather than a colour image.
    flat = rgb.reshape(-1, 3)
    if min(std) > .03:
        c = np.corrcoef(flat.T)
        if np.nanmean(np.abs(c[np.triu_indices(3, 1)])) < .35:
            return "packed", {"corr": float(np.nanmean(np.abs(c[np.triu_indices(3, 1)])))}
    # Alpha is only opacity if it looks like a cut-out: mostly opaque, with clean
    # holes and few in-between values. Engines often keep masks there instead
    # (Frostpunk: alpha is 0 on most building albedos), which would erase walls.
    cutout = False
    if alpha is not None and alpha.min() < .5:
        opaque, clear = (alpha > .9).mean(), (alpha < .1).mean()
        cutout = bool(opaque > .5 and clear > .01 and 1 - opaque - clear < .15)
    return "albedo", {"has_alpha": cutout}


def classify_with_stats(arr: np.ndarray, info: dict | None):
    """classify() plus the statistics the material rules read later."""
    role, details = classify(arr, info)
    if arr.shape[-1] == 4:  # sprites: how much of the sheet is visible at all
        a = arr[::4, ::4, 3]
        details["alpha_mean"], details["alpha_std"] = round(float(a.mean()), 4), round(float(a.std()), 4)
    rgb = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, -1)
    details["mean_rgb"] = [round(float(x), 4) for x in rgb.reshape(-1, 3)[::97].mean(0)]
    if role == "gray":
        # Brightness along the edges: tiling materials look the same there as inside,
        # while stamped masks (terrain snow/height blends) fade to black.
        g = rgb.mean(-1)
        k = max(1, min(g.shape) // 64)
        edge = np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])
        details["border_mean"] = round(float(edge.mean()), 4)
    return role, details


def slot_consensus(draws, entries, min_textures=3, agree=0.75):
    """Fix albedo <-> normal mix-ups from the shaders. A pixel shader reads each
    texture slot the same way, so when most distinct textures bound to a
    (shader, slot) share a role, a texture classified the other way everywhere
    it is bound gets that role. Frostpunk: t0159 (a packed normal) was taken for
    an albedo and t0091 (planks) for a normal. Grayscale/mask roles are left to
    scene_common.assign_slots, which knows the terrain masks.
    draws: iterable of (pixel shader, {slot: texture id}); entries are edited in
    place. Returns {texture id: (old role, new role)}."""
    from collections import Counter, defaultdict

    def key(e):
        ch = (e.get("details") or {}).get("channels")
        return e["role"] + (f":{ch}" if e["role"] == "normal" and ch else "")

    bound = defaultdict(set)
    for shader, tex in draws:
        for slot, tid in tex.items():
            if (entries.get(tid) or {}).get("role") in ("albedo", "normal", "gray", "packed"):
                bound[(shader, str(slot))].add(tid)
    consensus = {}
    for k, tids in bound.items():
        role, n = Counter(key(entries[t]) for t in tids).most_common(1)[0]
        if len(tids) >= min_textures and n / len(tids) >= agree:
            consensus[k] = role
    wanted = defaultdict(set)
    for k, tids in bound.items():
        if k in consensus:
            for t in tids:
                wanted[t].add(consensus[k])
    changed = {}
    for t, roles in wanted.items():
        if len(roles) != 1:
            continue
        new, old = next(iter(roles)), key(entries[t])
        if new == old or {new.split(":")[0], old.split(":")[0]} != {"albedo", "normal"}:
            continue
        role, _, channels = new.partition(":")
        details = {k: v for k, v in (entries[t].get("details") or {}).items() if k not in ("channels", "has_alpha")}
        details.update({"channels": channels} if channels else {"has_alpha": False})
        details["role_from"] = "slot consensus"
        entries[t].update(role=role, details=details)
        changed[t] = (old, new)
    return changed


def detail_layers(draws, entries, min_partners=2):
    """A draw has one base colour. A colour texture bound next to several different
    other colour textures can't be the base colour of all of them: it is a detail or
    overlay layer that the shader blends in some other way. Frostpunk: t0033, a 4096
    atlas of snow, ice, moss and rock tiles, is bound with the rocks' own albedos
    (t0021, t0019, t0077), and was picked over them as the larger texture. On the dead
    trees, whose only other colour is the grey bark t0046, it was read through the
    trunk's UVs and painted it with the atlas's moss tile. Each real albedo there has
    one colour partner (the atlas). Such a layer gets role "detail", which no slot
    rule uses. Grey maps don't count as partners: an albedo usually comes with its
    roughness. draws: iterable of (pixel shader, {slot: texture id}); entries are
    edited in place. Returns {texture id: (old role, "detail")}."""
    from collections import defaultdict
    partners = defaultdict(set)
    for _, tex in draws:
        colour = {t for t in tex.values() if (entries.get(t) or {}).get("role") == "albedo"}
        for t in colour:
            partners[t] |= colour - {t}
    changed = {}
    for t, others in partners.items():
        if len(others) >= min_partners:
            changed[t] = (entries[t]["role"], "detail")
            entries[t].update(role="detail", details=dict(entries[t].get("details") or {},
                                                          role_from=f"bound with {len(others)} other albedos"))
    return changed


def surface_draws(meshes):
    """(pixel shader, {slot: texture id}) of the manifest's surface draws."""
    return [(m["shaders"][1], m.get("textures", {})) for m in meshes
            if m.get("category") == "surface" and len(m.get("shaders") or []) >= 2]


def reclassify(capture: Path):
    """Re-run the role rules on a processed capture's saved textures (no rip
    needed) and write manifest.json. Returns {texture id: (old, new)}."""
    import json
    capture = Path(capture)
    path = capture / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["textures"]
    before = {t: (e.get("role"), (e.get("details") or {}).get("channels")) for t, e in entries.items()}
    for tid, e in entries.items():
        # "shared" comes from usage, not pixels; EXR/constant/cube textures keep their role.
        if e.get("role") in (None, "shared", "hdr", "constant", "environment") or not e.get("file", "").endswith(".png"):
            continue
        with open(capture / "textures" / e["file"], "rb") as fh:
            img = Image.open(fh)
            img.load()
        arr = np.asarray(img, dtype=np.float32) / 255.0
        role, details = classify_with_stats(arr[..., None] if arr.ndim == 2 else arr, e.get("info"))
        e.update(role=role, details=details)
    slot_consensus(surface_draws(manifest["meshes"]), entries)
    detail_layers(surface_draws(manifest["meshes"]), entries)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    after = {t: (e.get("role"), (e.get("details") or {}).get("channels")) for t, e in entries.items()}
    return {t: (before[t], after[t]) for t in entries if before[t] != after[t]}


def _cached(src: Path, dst_dir: Path):
    """A previously converted PNG that is newer than the source and decodes fully."""
    out = dst_dir / (src.stem + ".png")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        return None, None
    try:
        with open(out, "rb") as fh:  # explicit handle: Windows can't delete a file PIL still holds
            img = Image.open(fh)
            img.load()
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return img, arr[..., None] if arr.ndim == 2 else arr
    except Exception:  # truncated by an interrupted run
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        return None, None


def convert(src: Path, dst_dir: Path):
    """Convert one texture; returns a manifest entry or None if unreadable."""
    info = dds_info(src) if src.suffix.lower() == ".dds" else None
    img, arr = _cached(src, dst_dir)
    cached = img is not None
    try:
        if not cached:
            img, arr = load_image(src)
    except Exception as e:  # unsupported/odd formats: keep going, report later
        return {"source": str(src), "error": str(e), "info": info}
    role, details = classify_with_stats(arr, info)
    out = dst_dir / (src.stem + (".exr" if role == "hdr" else ".png"))
    if cached and out.suffix == ".png":
        return {"source": str(src), "file": out.name, "role": role, "details": details, "info": info,
                "size": [arr.shape[1], arr.shape[0]]}
    if role == "hdr":
        import os
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2  # OpenCV writes EXR; Pillow can't
        cv2.imwrite(str(out), arr[..., :3][..., ::-1].astype(np.float32))
    else:
        tmp = out.with_suffix(".tmp.png")  # write then rename, so a killed run never leaves half a PNG
        img.save(tmp, compress_level=1)    # ~5x faster than the default, files ~30% larger
        tmp.replace(out)
    return {"source": str(src), "file": out.name, "role": role, "details": details, "info": info,
            "size": [arr.shape[1], arr.shape[0]]}
