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
    role, details = classify(arr, info)
    if arr.shape[-1] == 4:  # sprites: how much of the sheet is visible at all
        a = arr[::4, ::4, 3]
        details["alpha_mean"], details["alpha_std"] = round(float(a.mean()), 4), round(float(a.std()), 4)
    rgb = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, -1)
    details["mean_rgb"] = [round(float(x), 4) for x in rgb.reshape(-1, 3)[::97].mean(0)]
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
