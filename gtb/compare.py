"""Render-vs-screenshot comparison used during the tuning pass.

Produces comparison.png (screenshot | render | edge overlay | luminance diff)
and a metrics dict: global/regional luminance ratios, colour cast, contrast and
how well the render's edges line up with the game's (camera alignment).
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lum(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _read(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    if size is not None and img.shape[1::-1] != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _edges(img):
    g = (cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.Canny(g, 60, 150) > 0


def compare(screenshot: Path, render: Path, out_png: Path, grid=3, ignore_mask=None):
    ren = _read(render)
    h, w = ren.shape[:2]
    ref = _read(screenshot, (w, h))
    ref_lin, ren_lin = _srgb_to_linear(ref), _srgb_to_linear(ren)
    ref_l, ren_l = _lum(ref_lin), _lum(ren_lin)

    valid = np.ones((h, w), bool)
    if ignore_mask is not None:  # e.g. HUD regions, given as [x0,y0,x1,y1] fractions
        for x0, y0, x1, y1 in ignore_mask:
            valid[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False

    def stats(lin, l):
        m = lin[valid].mean(0)
        return {"mean_rgb": m.round(4).tolist(),
                "lum_p5_p50_p95": np.percentile(l[valid], [5, 50, 95]).round(4).tolist(),
                "chroma_balance": (m / max(m.mean(), 1e-6)).round(3).tolist()}

    regions = []
    for gy in range(grid):
        row = []
        for gx in range(grid):
            sl = (slice(gy * h // grid, (gy + 1) * h // grid), slice(gx * w // grid, (gx + 1) * w // grid))
            v = valid[sl]
            if v.sum() < 10:
                row.append(None)
                continue
            a, b = ref_l[sl][v].mean(), ren_l[sl][v].mean()
            row.append(round(float(b / max(a, 1e-5)), 2))
        regions.append(row)

    e_ref, e_ren = _edges(ref), _edges(ren)
    dist = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 3)
    ren_edge_px = e_ren & valid
    align = float((dist[ren_edge_px] <= 3).mean()) if ren_edge_px.any() else 0.0

    metrics = {
        "screenshot": stats(ref_lin, ref_l),
        "render": stats(ren_lin, ren_l),
        # Log-average (geometric mean) luminance: smooth under exposure changes,
        # unlike the median, which jumps between modes in dark-vs-snow images.
        "exposure_offset_stops": round(float(np.log2(ren_l[valid] + 0.02).mean()
                                             - np.log2(ref_l[valid] + 0.02).mean()), 2),
        "region_lum_ratio_render_over_game": regions,
        "edge_alignment_within_3px": round(align, 3),
    }

    overlay = ref.copy() * 0.6
    overlay[e_ren] = (1, 0.2, 0.2)
    overlay[e_ref & ~e_ren] = np.maximum(overlay[e_ref & ~e_ren], (0.2, 0.9, 1.0))
    diff = np.clip((np.log2(ren_l + 1e-3) - np.log2(ref_l + 1e-3)) / 3, -1, 1)
    heat = np.stack([np.clip(diff, 0, 1), np.full_like(diff, 0.15), np.clip(-diff, 0, 1)], -1)
    top = np.concatenate([ref, ren], 1)
    bottom = np.concatenate([overlay, heat], 1)
    sheet = np.concatenate([top, bottom], 0)
    labels = ["GAME", "BLENDER", "EDGES red=blender cyan=game", "LUM DIFF red=too bright blue=too dark"]
    img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    for i, t in enumerate(labels):
        cv2.putText(img, t, ((i % 2) * w + 10, (i // 2) * h + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, ((i % 2) * w + 10, (i // 2) * h + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)
    return metrics


if __name__ == "__main__":
    import sys
    m = compare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(json.dumps(m, indent=2))
