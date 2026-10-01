"""The look's colour pipeline as a 3D LUT for Unreal, without Blender:

    python -m ue.colour_lut <look.json> <out.png> [--compare-blender]

Blender's chain, reproduced: the compositor grade (Color Balance lift/gamma/gain ->
Hue/Saturation -> Bright/Contrast, in numpy), then the view (exposure -> look -> view
transform -> display -> gamma) through OpenColorIO with Blender's own colour config.
Fog and bloom are spatial, so Unreal does those itself. Checked against Blender 5.2's
bake (ue/blender_lut.py, `--compare-blender`): within 2/65535 everywhere.

Blender's colour config: the installed Blender's (config.json blender_exe), so the
colours match that version, or else Blender 5.2's, downloaded once from Blender's
repository into .cache/colormanagement (out of git: the files carry no licence that
lets this project ship them). Only the files a transform needs are fetched (AgX: the
config and two 3D LUTs, about 4.5 MB).

LUT layout (16-bit PNG, display-referred sRGB, as ue/blender_lut.py): width size*size,
height size. Pixel (x, y) holds the input colour (r, g, b) = decode(x % size, y,
x // size), decode(i) = 2 ** (lo + (hi - lo) * i / (size - 1)) scene-linear, row 0 at top.

The grade, measured against Blender 5.2 (each node alone, through the Standard view and
through AgX):
- Color Balance (lift/gamma/gain) works in linear: gain * (x + (lift - 1) * (1 - x)),
  clamped at 0, then ** (1 / gamma). (Blender 4's sRGB-space formula was off by up to
  8000/65535 on dark tints.)
- Hue/Saturation: HSV with the saturation scaled and not clamped, the result clamped at
  0. Clamping the saturation at 1 was off by up to 25000/65535 on saturated colours.
- Bright/Contrast: c * a + b with Blender's a and b from brightness and contrast.
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "colormanagement"
BLENDER_SRC = ("https://projects.blender.org/blender/blender/raw/branch/blender-v5.2-release/"
               "release/datafiles/colormanagement/")
SEARCH = ("luts", "filmic", "icc")       # the config's search_path


# --------------------------------------------------------------------------- grade

def _lgg(c, lift, gamma, gain):
    v = np.maximum(np.asarray(gain, float) * (c + (np.asarray(lift, float) - 1.0) * (1.0 - c)), 0.0)
    return np.power(v, 1.0 / np.maximum(np.asarray(gamma, float), 1e-6))


def _rgb_to_hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(-1)
    d = v - c.min(-1)
    s = np.where(v != 0, d / np.where(v != 0, v, 1.0), 0.0)
    dd = np.where(d != 0, d, 1.0)
    h = np.where(v == r, (g - b) / dd, np.where(v == g, 2.0 + (b - r) / dd, 4.0 + (r - g) / dd))
    return np.where(d != 0, (h / 6.0) % 1.0, 0.0), s, v


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(int) % 6
    return np.stack([np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]),
                     np.choose(i, [p, p, t, v, v, q])], -1)


def _saturation(c, sat):
    h, s, v = _rgb_to_hsv(c)
    return np.maximum(_hsv_to_rgb(h, s * sat, v), 0.0)


def _bright_contrast(c, bright, contrast):
    b, k = bright / 100.0, contrast / 100.0
    delta = k / 2.0
    if k > 0:
        a = 1.0 / max(1.0 - delta * 2.0, 1.1920929e-07)
        off = a * (b - delta)
    else:
        a = max(1.0 + delta * 2.0, 0.0)
        off = a * b - delta
    return c * a + off


def grade(c, g):
    """Blender's compositor grade (gtb_scene.apply_compositor) on scene-linear RGB (..., 3)."""
    c = _lgg(c, g["lift"], g["gamma"], g["gain"])
    c = _saturation(c, g["saturation"])
    return _bright_contrast(c, g["brightness"], g["contrast"])


# --------------------------------------------------------------------------- view

def blender_config(blender_exe):
    """The installed Blender's colour config, or None."""
    if not blender_exe or not Path(blender_exe).exists():
        return None
    found = sorted(Path(blender_exe).parent.glob("*/datafiles/colormanagement/config.ocio"))
    return found[-1] if found else None


def _download(rel, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(BLENDER_SRC + rel, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dst)


def config_file(blender_exe=None, log=print):
    """Blender's config.ocio: the installed Blender's, else the cached download."""
    cfg = blender_config(blender_exe)
    if cfg is not None:
        return cfg, False
    cfg = CACHE / "config.ocio"
    if not cfg.exists():
        log(f"[ue] downloading Blender 5.2's colour config into {CACHE}")
        _download("config.ocio", cfg)
    return cfg, True


def processor(look, blender_exe=None, log=print):
    """OpenColorIO CPU processor: scene-linear -> sRGB display, with Blender's exposure,
    look, view transform and gamma (its legacy viewing pipeline)."""
    import PyOpenColorIO as OCIO
    path, cached = config_file(blender_exe, log)
    for _ in range(12):      # fetch the files the transform needs, one at a time
        try:
            cfg = OCIO.Config.CreateFromFile(str(path))
            vp = OCIO.LegacyViewingPipeline()
            vp.setDisplayViewTransform(OCIO.DisplayViewTransform(src="scene_linear", display="sRGB",
                                                                 view=look["view_transform"]))
            if look.get("look") and look["look"] != "None":
                vp.setLooksOverrideEnabled(True)
                vp.setLooksOverride(look["look"])
            s = 2.0 ** float(look.get("exposure", 0.0))
            vp.setLinearCC(OCIO.MatrixTransform.Scale([s, s, s, 1.0]))
            gm = float(look.get("gamma", 1.0) or 1.0)
            if gm != 1.0:
                vp.setDisplayCC(OCIO.ExponentTransform([1.0 / gm] * 3 + [1.0]))
            return vp.getProcessor(cfg).getDefaultCPUProcessor()
        except (OCIO.Exception, OCIO.ExceptionMissingFile) as e:     # MissingFile isn't an OCIO.Exception
            m = re.search(r"'([^'/\\]+\.(?:cube|spi1d|spi3d|spimtx|icc|cc|cdl|clf|ctf))'", str(e))
            if not cached or m is None:
                raise
            name = m.group(1)
            for sub in SEARCH:
                try:
                    log(f"[ue] downloading {sub}/{name}")
                    _download(f"{sub}/{name}", CACHE / sub / name)
                    OCIO.ClearAllCaches()        # it remembers the file as missing otherwise
                    break
                except OSError:
                    continue
            else:
                raise
    raise RuntimeError("OpenColorIO kept asking for files")


# --------------------------------------------------------------------------- bake

def grid(size, lo, hi):
    levels = 2.0 ** (lo + (hi - lo) * np.arange(size) / (size - 1))
    x = np.arange(size * size)
    r = levels[x % size][None, :].repeat(size, 0)
    b = levels[x // size][None, :].repeat(size, 0)
    g = levels[np.arange(size)][:, None].repeat(size * size, 1)
    return np.stack([r, g, b], -1)


def bake(look, out, size=64, lo=-12.0, hi=8.0, blender_exe=None, log=print):
    """Write the look's LUT (layout above). `look` is a loaded look.json (defaults merged)."""
    rgb = grade(grid(size, lo, hi), look["grade"])
    flat = np.ascontiguousarray(rgb.reshape(-1, 3).astype(np.float32))
    processor(look, blender_exe, log).applyRGB(flat)
    img = np.clip(flat.reshape(rgb.shape), 0.0, 1.0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.round(img[:, :, ::-1] * 65535.0).astype(np.uint16))
    return out


def compare(a, b):
    """Largest and mean difference of two LUT PNGs, in 16-bit steps."""
    A, B = (cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float64) for p in (a, b))
    d = np.abs(A - B)
    return float(d.max()), float(d.mean())


def main():
    sys.path.insert(0, str(ROOT))
    from gtb import config
    from gtb.scene_common import load_look
    from ue.look import LUT_HI, LUT_LO, LUT_SIZE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("look")
    ap.add_argument("out")
    ap.add_argument("--compare-blender", action="store_true",
                    help="also bake with Blender (ue/blender_lut.py) and print the difference")
    a = ap.parse_args()
    cfg = config.load()
    look = load_look(Path(a.look))
    out = bake(look, a.out, LUT_SIZE, LUT_LO, LUT_HI, cfg.get("blender_exe"))
    print(f"[ue] LUT {LUT_SIZE}^3 -> {out}")
    if a.compare_blender:
        import subprocess
        ref = Path(a.out).with_name(Path(a.out).stem + "_blender.png")
        subprocess.run([cfg["blender_exe"], "-b", "--factory-startup", "--python", str(ROOT / "ue" / "blender_lut.py"),
                        "--", a.look, str(ref), str(LUT_SIZE), str(LUT_LO), str(LUT_HI)],
                       capture_output=True, check=True)
        mx, mean = compare(out, ref)
        print(f"[ue] vs Blender's bake: max {mx:.0f}, mean {mean:.2f} (16-bit steps; 257 = one 8-bit step)")


if __name__ == "__main__":
    main()
