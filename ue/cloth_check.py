"""Check that the fluttering banners (M_GTB_Cloth) stay out of the scene geometry.

Replays gtb_hlsl.CLOTH_WPO in numpy (keep in sync) over some seconds for every
cloth actor of the capture's unreal/plan.json, and ray-casts each vertex's
motion against the solid surfaces around it; also flags snapping and
banners that barely move. Needs no Unreal; a few seconds.

    python -m ue.cloth_check <capture>      (uses the capture's look.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ue.export import CM, TriSoup, cast2d, cloth_taper, cloth_weight, to_ue  # noqa: E402


def cloth_offsets(P, cloth, t, wind, ripple=0.03, sway=0.08, wavelength=0.75, speed=0.5):
    """gtb_hlsl.CLOTH_WPO for vertices P (Unreal cm, world) at time t."""
    z = P[:, 2]
    L = max(np.ptp(z), 1.0)
    halfw = 0.5 * np.linalg.norm(np.ptp(P[:, :2], axis=0))
    d = np.clip((z.max() - z) / L, 0, 1)
    held = cloth[2] > 0.5
    w = cloth_weight(z, held)
    c = (P.min(0) + P.max(0)) / 2
    phase = np.modf(np.sin(c[:2] @ [12.9898, 78.233]) * 43758.5453)[0] % 1.0 * 2 * np.pi
    ws = np.linalg.norm(wind)
    wd = wind / ws if ws > 1e-3 else np.array([1.0, 0.0])
    smax = min(ws / 250.0, 1.5)
    gust = 0.7 + 0.3 * np.sin(t * 0.45 + phase) * np.sin(t * 0.23 + phase * 1.3)
    n = np.array(cloth[:2])
    ac = np.array([-n[1], n[0]])
    sides = cloth[3] > 0.5
    lo, hi = P.min(0), P.max(0)
    xa = (P[:, :2] - (lo[:2] + hi[:2]) / 2) @ ac
    wn = w * (cloth_taper(xa, 0.5 * np.abs(ac) @ (hi[:2] - lo[:2])) if sides else 1.0)
    bud = np.array(cloth[4:12])
    sp = d * 2 * np.pi / max(wavelength, 0.3)
    wt = t * speed * 2 * np.pi
    tr = sp - wt + phase
    r = np.sin(tr) + 0.35 * np.sin(0.6 * sp - 1.7 * wt + phase * 2.1)
    tw = 0.12 * ((P[:, :2] - c[:2]) @ ac) * np.sin(0.7 * tr + 1.3)
    A = ripple * L * 1.35 + 0.12 * halfw
    rn = (ripple * L * r + tw) / max(A, 1e-3)
    a = min(A * smax, 0.5 * (bud[0] + bud[4]))
    swing = sway * L * smax * (0.0 if held else 1.0)
    C = np.clip(swing * (wd @ n), a - bud[4], bud[0] - a)
    SA = np.clip(swing * (wd @ ac), -bud[6], bud[2]) * (0.0 if sides else 1.0)
    f = 1.0
    for j in (-1, 0, 1):
        v = np.array([C + j * a, SA])
        rr = np.linalg.norm(v)
        if rr > 1e-3:
            k = (np.arctan2(v[1], v[0]) / (2 * np.pi) % 1.0) * 8.0
            k0 = int(np.floor(k)) % 8
            f = min(f, (bud[k0] + (bud[(k0 + 1) % 8] - bud[k0]) * (k - np.floor(k))) / rr)
    off = (np.outer((C + a * rn) * wn, n) + np.outer(SA * w, ac)) * f * gust
    lift = (off ** 2).sum(1) / (2 * np.maximum(d * L, 1.0)) * (0.0 if held else 1.0)
    return np.column_stack([off, lift])


def check(capture, seconds=10.0, tolerance=3.0, fps=30):
    """Replay every banner for `seconds` with the capture's look and report:
      clip   banners whose vertices go more than `tolerance` cm into geometry
             (export.cloth_room allows 3 cm next to geometry they already touch)
      jerky  banners whose fastest vertex moves over 1.3x the fastest vertex of
             the same motion without room limits: limits that snap
      still  banners that move less than 20% of that unlimited motion
    Expected on the Frostpunk capture: 0 clip, 0 jerky, 4 still (boxed in)."""
    from gtb.scene_common import load_look
    from ue.look import cloth_params
    capture = Path(capture)
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((capture / "unreal" / "plan.json").read_text(encoding="utf-8"))
    files = {m["name"]: capture / m["file"] for m in manifest["meshes"]}
    actors = {a["name"]: a for a in plan["actors"]}
    cp = cloth_params(load_look(capture / "look.json"))
    wind = np.array(cp["Wind"][:2])
    params = dict(ripple=cp["Ripple"], sway=cp["Sway"], wavelength=cp["WaveLength"], speed=cp["FlutterSpeed"])
    solids = []
    for name, a in actors.items():
        if a.get("cloth") or a.get("hidden") or a.get("folder") != "Geometry":
            continue
        data = np.load(files[name])
        solids.append((to_ue(data["positions"].astype(np.float64)), data["indices"].astype(np.int64).reshape(-1, 3)))
    soup = TriSoup(solids)
    report = {"clip": [], "jerky": [], "still": []}
    ts = np.arange(0.0, seconds, 1.0 / fps)
    for name, a in actors.items():
        if not a.get("cloth"):
            continue
        P = to_ue(np.load(files[name])["positions"].astype(np.float64))
        offs = np.stack([cloth_offsets(P, a["cloth"], t, wind, **params) for t in ts])
        free = np.stack([cloth_offsets(P, a["cloth"][:4] + [1e4] * 8, t, wind, **params) for t in ts])
        speed = lambda o: float((np.linalg.norm(np.diff(o, axis=0), axis=2) * fps).max())  # noqa: E731
        if speed(offs) > 1.3 * speed(free):
            report["jerky"].append((name, speed(offs), speed(free)))
        reach, reach_free = (float(np.linalg.norm(o[..., :2], axis=2).max()) for o in (offs, free))
        if reach < 0.2 * reach_free:
            report["still"].append((name, reach, reach_free))
        m = np.linalg.norm(offs[..., :2], axis=2)               # (steps, V); the lift is a few cm
        near = soup.near(P.min(0) - m.max() - 1, P.max(0) + m.max() + 1)
        depth = []
        for v in range(len(P)):                                 # horizontal sweep at the vertex's height
            mv = m[:, v] > tolerance
            if mv.any():
                segs = near.slice(P[v, 2])
                hit = cast2d(np.tile(P[v, :2], (mv.sum(), 1)), offs[mv, v, :2] / m[mv, v, None], segs)
                depth.append((m[mv, v] - hit).max())
        worst = float(max(depth)) if depth else 0.0
        if worst > tolerance:
            report["clip"].append((name, worst))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    cap = Path(ap.parse_args().capture)
    if not cap.exists():
        cap = ROOT / "captures" / cap
    r = check(cap)
    for name, depth in sorted(r["clip"], key=lambda x: -x[1]):
        print(f"{name}: {depth:.0f} cm into geometry")
    for name, v, vf in r["jerky"]:
        print(f"{name}: jerky, {v:.0f} cm/s where the unlimited motion peaks at {vf:.0f} cm/s")
    for name, m, mf in r["still"]:
        print(f"{name}: barely moves ({m:.0f} of {mf:.0f} cm; boxed in)")
    print(f"{len(r['clip'])} banner(s) clip, {len(r['jerky'])} jerky, {len(r['still'])} barely move")
