"""Routes through the city for the walking people (ue/walkers.py, docs/WALKING_PEOPLE.md):
a map of where a person can walk, built from the rip's surfaces, and closed routes on
it that keep to the game's roads.

The map: a grid of CELL m columns over the walkers' part of the scene. Points sampled
SPACING m apart on every surface fall into VOXEL m boxes. In each column, a run of
filled boxes whose top faces up (normal z > 0.5) with HEAD m free above it is a floor,
at the mean height of the up-facing points in its top box. The run's thickness doesn't
matter: Frostpunk's road tiles have skirts reaching a metre into the terrain, and a
wall, post or crate top is cut off anyway, because floors only link to neighbouring
floors within STEP m. Floors whose 8 neighbours are all linked keep CELL m from
anything solid; tighter ones cost more but stay usable (doorways and the gaps between
buildings and road-side stands; without them Frostpunk's city fell into islands).

Roads: the textured surface most people stand on (Frostpunk: MI_M000, the snow paths
of the ring-and-spoke streets; the bare terrain has no UVs and doesn't count). Walking
on them costs 1, elsewhere OFFROAD.

A route: a street the person walks up and back. From the floor nearest it, one end is
up to OUT m ahead along the cheapest ways, on a road, picked so the way starts off
along the person's heading; the other end as far behind, leaving the start the opposite
way. So at time 0 the person is in mid-street, walking on as the game had it, and turns
round at the far ends in half circles. Walkers keep LANE m to the right of their way, so
the way up and the way back are two lanes and people coming the other way pass. The line
is straightened into runs where a person can walk straight, its corners rounded and the
lane offset smoothed along it (route_points); then pinned to the person's spot and put on
the ground with downward ray casts.
"""
import heapq
import math

import numpy as np

CELL = 0.5
VOXEL = 0.25
HEAD = 1.9
STEP = 0.35
SPACING = 0.2
OUT = (25.0, 60.0)        # m out to the turning point
OFFROAD = 2.5
LANE = 0.75
FADE = 5.0
MARGIN = 65.0             # m of map around the walkers
DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
STEPLEN = np.linalg.norm(DIRS, axis=1) * CELL


def sample_points(tri, spacing=SPACING):
    """Points on triangles (T,3,3) at most `spacing` apart, and each one's triangle."""
    e = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], 1)
    k = np.maximum(1, np.ceil(np.linalg.norm(e, axis=2).max(1) / spacing)).astype(int)
    pts, ids = [], []
    for kk in np.unique(k):
        sel = np.where(k == kk)[0]
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        ok = i + j <= kk
        u, v = i[ok] / kk, j[ok] / kk
        t = tri[sel]
        P = t[:, None, 0] + u[None, :, None] * (t[:, None, 1] - t[:, None, 0]) \
            + v[None, :, None] * (t[:, None, 2] - t[:, None, 0])
        pts.append(P.reshape(-1, 3))
        ids.append(np.repeat(sel, len(u)))
    return np.concatenate(pts), np.concatenate(ids)


def surface_below(meshes, p, reach=0.3, depth=3.0):
    """Index of the mesh whose surface is highest under point p (at most `reach` above it)."""
    best, best_z = None, -np.inf
    for i, (P, T, *_) in enumerate(meshes):
        lo, hi = P.min(0), P.max(0)
        if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] + reach and hi[2] >= p[2] - depth):
            continue
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        v0, v1, v2 = c[:, :2] - a[:, :2], b[:, :2] - a[:, :2], p[:2] - a[:, :2]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-12
        u = np.where(ok, (d11 * d20 - d01 * d21) / np.where(ok, den, 1.0), -1.0)
        v = np.where(ok, (d00 * d21 - d01 * d20) / np.where(ok, den, 1.0), -1.0)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        if not inside.any():
            continue
        z = a[inside, 2] + u[inside] * (c[inside, 2] - a[inside, 2]) + v[inside] * (b[inside, 2] - a[inside, 2])
        z = z[z <= p[2] + reach]
        if len(z) and z.max() > best_z:
            best, best_z = i, float(z.max())
    return best


def road_meshes(meshes, feet, log=print):
    """Which meshes are roads: those with the textured material most people stand on.
    meshes: [(positions, tris, material, has_uv)], feet: points under people."""
    count = {}
    for f in feet:
        i = surface_below(meshes, f)
        if i is not None and meshes[i][3]:
            count[meshes[i][2]] = count.get(meshes[i][2], 0) + 1
    if not count:
        return set(), None
    mat = max(count, key=count.get)
    if count[mat] < 3:
        return set(), None
    roads = {i for i, m in enumerate(meshes) if m[2] == mat}
    log(f"[people] roads: {mat} ({len(roads)} meshes; {count[mat]} of {len(feet)} people stand on it)")
    return roads, mat


class WalkMap:
    """Where a person can walk (see the module doc). meshes: [(positions, tris, is_road)],
    Blender m; lo, hi: the box (x, y, z) it covers."""

    def __init__(self, meshes, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.nx, self.ny = np.ceil((self.hi[:2] - self.lo[:2]) / CELL).astype(int)
        self.nz = int(np.ceil((self.hi[2] - self.lo[2]) / VOXEL))
        tris, road = [], []
        for P, T, r in meshes:
            t = P[T]
            keep = (t.max(1) >= self.lo).all(1) & (t.min(1) <= self.hi).all(1)
            if keep.any():
                tris.append(t[keep])
                road.append(np.full(keep.sum(), bool(r)))
        tri = np.concatenate(tris)
        road = np.concatenate(road)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        up = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        pts, ids = sample_points(tri)
        ix = np.clip(((pts[:, 0] - self.lo[0]) / CELL).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((pts[:, 1] - self.lo[1]) / CELL).astype(np.int64), 0, self.ny - 1)
        iz = np.clip(((pts[:, 2] - self.lo[2]) / VOXEL).astype(np.int64), 0, self.nz - 1)
        # Road footprint: every column under a road triangle (the terrain pokes through the
        # road surface along its triangle diagonals, so the top surface alone misses some).
        self.road_col = np.zeros(self.nx * self.ny, bool)
        self.road_col[(ix * self.ny + iy)[road[ids]]] = True
        key = (ix * self.ny + iy) * self.nz + iz
        order = np.argsort(key, kind="stable")
        key, pz, pn = key[order], pts[order, 2], up[ids[order]]
        del pts, ids
        vox, start = np.unique(key, return_index=True)
        zmax = np.maximum.reduceat(pz, start)
        nzmax = np.maximum.reduceat(pn, start)
        w = (pn > 0.5).astype(np.float64)
        cnt = np.add.reduceat(w, start)
        zup = np.where(cnt > 0, np.add.reduceat(pz * w, start) / np.maximum(cnt, 1), zmax)
        col, iv = vox // self.nz, vox % self.nz
        first = np.where(np.r_[True, (col[1:] != col[:-1]) | (iv[1:] != iv[:-1] + 1)])[0]
        last = np.r_[first[1:], len(vox)] - 1
        s_col = col[first]
        top = zup[last]
        above = np.r_[self.lo[2] + iv[first[1:]] * VOXEL, np.inf]
        clear = np.where(np.r_[s_col[1:], -1] == s_col, above - top, np.inf)
        floor = (nzmax[last] > 0.5) & (clear >= HEAD)
        order = np.lexsort((top[floor], s_col[floor]))
        self.col = s_col[floor][order]
        self.z = top[floor][order]
        self.road = self.road_col[self.col]
        self._link()

    def _link(self):
        """Neighbours (the floor of the next column closest in height, within STEP), the
        clearance classes and the main network (the largest linked set of floors)."""
        n = len(self.col)
        fx, fy = self.col // self.ny, self.col % self.ny
        ucol, first, count = np.unique(self.col, return_index=True, return_counts=True)
        self.nbr = np.full((n, 8), -1, np.int64)
        for d, (dx, dy) in enumerate(DIRS):
            tx, ty = fx + dx, fy + dy
            inside = (tx >= 0) & (tx < self.nx) & (ty >= 0) & (ty < self.ny)
            tcol = tx * self.ny + ty
            pos = np.clip(np.searchsorted(ucol, tcol), 0, len(ucol) - 1)
            has = inside & (ucol[pos] == tcol)
            best, bestd = np.full(n, -1, np.int64), np.full(n, np.inf)
            for k in range(int(count.max())):
                valid = has & (k < count[pos])
                idx = np.where(valid, first[pos] + k, 0)
                dz = np.abs(self.z[idx] - self.z)
                better = valid & (dz < bestd)
                best, bestd = np.where(better, idx, best), np.where(better, dz, bestd)
            self.nbr[:, d] = np.where(bestd <= STEP, best, -1)
        linked = self.nbr >= 0
        self.interior = linked.all(1)
        self.inner = self.interior & np.where(linked, self.interior[np.maximum(self.nbr, 0)], False).all(1)
        comp = np.full(n, -1, np.int64)
        c = 0
        for s in range(n):
            if comp[s] >= 0:
                continue
            stack, comp[s] = [s], c
            while stack:
                u = stack.pop()
                for v in self.nbr[u]:
                    if v >= 0 and comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
        self.main = comp == np.argmax(np.bincount(comp)) if n else np.zeros(0, bool)
        self.cost = (np.where(self.road, 1.0, OFFROAD)
                     * np.where(self.inner, 1.0, np.where(self.interior, 2.0, 6.0)))

    def xy(self, i):
        i = np.asarray(i)
        return np.stack([self.lo[0] + (self.col[i] // self.ny + 0.5) * CELL,
                         self.lo[1] + (self.col[i] % self.ny + 0.5) * CELL], -1)

    def nearest(self, p, reach=1.5):
        """The main-network floor (not a tight one) nearest point p that it can step to in a
        straight line over floors, or None. A person boxed in (Frostpunk: mesh_637 between a
        crate and a wall) would walk out through the crate, so it keeps its pose."""
        cand = np.where(self.main & self.interior)[0]
        if not len(cand):
            return None
        d2 = ((self.xy(cand) - p[:2]) ** 2).sum(1) + 4.0 * (self.z[cand] - p[2]) ** 2
        for k in np.argsort(d2)[:40]:
            if d2[k] > reach * reach:
                break
            q = self.xy(cand[k])
            gap = float(np.linalg.norm(q - p[:2]))
            # from 0.3 m out: the person's own spot is fine (the game put it there)
            t = np.linspace(min(0.3 / max(gap, 1e-6), 1.0), 1.0, max(2, int(gap / 0.1) + 1))[:, None]
            line = p[:2] * (1 - t) + q * t
            if self.walkable_at(line, np.full(len(line), p[2]), tight=True).all():
                return int(cand[k])
        return None

    def walkable_at(self, xy, z, tol=0.6, tight=False):
        """Is there a main-network floor in the column under xy, within tol of z? Tight
        floors (next to something solid) count only with tight=True."""
        ix = np.floor((xy[:, 0] - self.lo[0]) / CELL).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.lo[1]) / CELL).astype(np.int64)
        c = ix * self.ny + iy
        pos = np.clip(np.searchsorted(self.col, c), 0, len(self.col) - 1)
        out = np.zeros(len(xy), bool)
        for k in range(3):
            i = np.clip(pos + k, 0, len(self.col) - 1)
            out |= (self.col[i] == c) & self.main[i] & (np.abs(self.z[i] - z) < tol) & (tight | (self.cost[i] < 6.0))
        return out

    def dijkstra(self, start, cost, limit=np.inf):
        """Cheapest cost, walked length and parent of every main-network floor from start."""
        n = len(self.col)
        dist, length, parent = np.full(n, np.inf), np.full(n, np.inf), np.full(n, -1, np.int64)
        dist[start], length[start] = 0.0, 0.0
        heap = [(0.0, start)]
        nbr, main = self.nbr, self.main
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u] or d > limit:
                continue
            for k in range(8):
                v = nbr[u, k]
                if v < 0 or not main[v]:
                    continue
                nd = d + STEPLEN[k] * 0.5 * (cost[u] + cost[v])
                if nd < dist[v]:
                    dist[v], parent[v], length[v] = nd, u, length[u] + STEPLEN[k]
                    heapq.heappush(heap, (nd, v))
        return dist, length, parent


def _path(parent, t):
    p = [t]
    while parent[p[-1]] >= 0:
        p.append(int(parent[p[-1]]))
    return p[::-1]


def _forward(pts, heading, ahead=3.0):
    """How well a polyline starts off along the heading (cosine, 3 m ahead)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    k = int(np.searchsorted(np.cumsum(seg), ahead))
    d = pts[min(k + 1, len(pts) - 1)] - pts[0]
    return float(d @ heading / (np.linalg.norm(d) + 1e-9))


def _pick(wm, cand, parent, length, heading, xy):
    """The candidate end whose way from the start best follows `heading`, the longer the better."""
    best = None
    for t in cand[np.argsort(-length[cand])][::max(1, len(cand) // 400)]:
        way = _path(parent, t)
        score = _forward(xy[way], heading) + length[t] / OUT[1]
        if best is None or score > best[0]:
            best = (score, way)
    return best


def plan_route(wm, p0, heading):
    """A street to walk up and down: (floors from one end to the other, index of the
    person's floor on it), or None. The person walks it along `heading` first: one end is
    OUT m ahead along the cheapest ways (ending on a road), the other as far behind."""
    s = wm.nearest(p0)
    if s is None:
        return None
    dist, length, parent = wm.dijkstra(s, wm.cost, limit=OUT[1] * OFFROAD * 6)
    xy = wm.xy(np.arange(len(wm.col)))
    ok = np.isfinite(length) & (length >= OUT[0]) & (length <= OUT[1])
    cand = np.where(ok & wm.road)[0]
    if len(cand) < 10:
        cand = np.where(ok)[0]
    if not len(cand):
        return None
    ahead = _pick(wm, cand, parent, length, heading, xy)
    if ahead[0] < 0.0 + OUT[0] / OUT[1]:            # nothing ahead: turn round, walk the best way
        ahead = _pick(wm, cand, parent, length, -heading, xy)
    fwd = ahead[1]
    # Behind: the way that leaves the start most opposite to the way ahead, on other floors.
    first = xy[fwd[min(len(fwd) - 1, 6)]] - xy[s]
    first /= np.linalg.norm(first) + 1e-9
    used = np.zeros(len(wm.col), bool)
    used[fwd[3:]] = True
    behind = np.where(np.isfinite(length) & (length >= 3.0) & (length <= OUT[1]) & ~used)[0]
    if (wm.road[behind]).sum() >= 10:                # keep to the roads behind too
        behind = behind[wm.road[behind]]
    back = [s]
    if len(behind):
        best = _pick(wm, behind, parent, length, -first, xy)
        if best[0] > 0.3:
            back = best[1]
    street = back[::-1] + fwd[1:]
    return street, len(back) - 1


def _smooth_closed(P, k):
    """Moving average over 2k+1 points of a closed polyline."""
    out = np.zeros_like(P)
    for d in range(-k, k + 1):
        out += np.roll(P, d, 0)
    return out / (2 * k + 1)


def _resample_closed(P, n):
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, s[-1], n, endpoint=False)
    closed = np.vstack([P, P[:1]])
    return np.stack([np.interp(q, s, closed[:, i]) for i in range(P.shape[1])], 1), float(s[-1])


def _smooth_open(P, k):
    """Moving average over 2k+1 points of an open polyline (the ends stay put)."""
    out = P.copy()
    n = len(P)
    for i in range(n):
        r = min(k, i, n - 1 - i)
        out[i] = P[i - r:i + r + 1].mean(0)
    return out


def _col_of(wm, xy):
    ix = np.floor((xy[:, 0] - wm.lo[0]) / CELL).astype(np.int64)
    iy = np.floor((xy[:, 1] - wm.lo[1]) / CELL).astype(np.int64)
    ok = (ix >= 0) & (ix < wm.nx) & (iy >= 0) & (iy < wm.ny)
    return np.where(ok, ix * wm.ny + iy, 0), ok


def _clear(wm, a, b, road):
    """Can a person walk the straight line a -> b (xy, z) over interior floors (and only
    over road where `road`)?"""
    n = max(2, int(np.linalg.norm(b[:2] - a[:2]) / 0.15) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    xy = a[:2] * (1 - t) + b[:2] * t
    z = a[2] * (1 - t[:, 0]) + b[2] * t[:, 0]
    ok = wm.walkable_at(xy, z, tol=0.4)
    if road:
        c, inside = _col_of(wm, xy)
        ok &= inside & wm.road_col[c]
    return bool(ok.all())


def _straighten(wm, P, road, keep, reach=10.0):
    """Waypoints of a street (P: floor points, road: their road flags): from each one,
    the furthest point up to `reach` m on that can be walked in a straight line, over the
    road where the street is road. `keep` stays a waypoint (the person's start)."""
    out, i, n = [0], 0, len(P)
    while i < n - 1:
        stop = min([k for k in keep if k > i] + [n - 1])
        best, j = i + 1, i + 1
        while j <= stop and np.linalg.norm(P[j, :2] - P[i, :2]) <= reach:
            on_road = road[i] and road[j] and road[i:j + 1].mean() >= 0.8
            if _clear(wm, P[i], P[j], on_road):
                best = j
            j += 1
        out.append(best)
        i = best
    return out


def _densify(P, step=0.25):
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(q, s, P[:, i]) for i in range(P.shape[1])], 1), s, q


def _guarded_smooth(wm, P, k, closed=False):
    """Moving average over 2k+1 points, kept only where the smoothed point (and the step to
    the next one) is still on interior floors."""
    S = _smooth_closed(P, k) if closed else _smooth_open(P, k)
    nxt = np.roll(S[:, :2], -1, 0)
    ok = wm.walkable_at(S[:, :2], S[:, 2]) & wm.walkable_at(0.5 * (S[:, :2] + nxt), S[:, 2])
    return np.where(ok[:, None], S, P)


def _window(v, k, fn):
    """fn (np.minimum or np.add) over a circular window of 2k+1."""
    out = v.copy()
    for d in range(1, k + 1):
        out = fn(out, np.roll(v, d))
        out = fn(out, np.roll(v, -d))
    return out


def route_points(wm, route, p0=None):
    """The street walked up and back as a smooth closed line (xy, floor z) every 0.25 m,
    keeping LANE m to the right of the way, turning round in half circles at the ends,
    starting where the person stands. Returns (points, length, start along it).

    The street's grid floors are straightened into runs of up to 10 m wherever a person
    can walk straight (and stays on the road where the street is road), and the corners
    rounded. The lane offset is worked out per point (how much of it stays on interior
    floors), then smoothed along the way, so a lane narrows gradually where the street
    does: switched point by point, it made the walkers zigzag."""
    street, k0 = route
    F = np.column_stack([wm.xy(street), wm.z[street]])
    way = _straighten(wm, F, wm.road[street], {k0})
    W = F[way]
    start = float(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1))])[way.index(k0)])
    P, _, q = _densify(W)
    for k in (6, 3):                                            # round the corners: +-1.5 m, then +-0.75 m
        P = _guarded_smooth(wm, P, k)
    # Room to turn round: shorten an end of the street (up to 3 m, never past the start)
    # until a half circle of LANE m fits there; squeezed, the walker spun on the spot.
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])
    i0 = int(np.argmin(np.abs(cum - start)))
    ring = np.linspace(0.0, np.pi, 12)
    for end in (-1, 0):
        for _ in range(6):
            a, b = (P[-1], P[-3]) if end == -1 else (P[0], P[2])
            ahead = a[:2] - b[:2]
            ahead /= max(np.linalg.norm(ahead), 1e-9)
            rr = np.array([ahead[1], -ahead[0]])
            pts = a[:2] + LANE * (np.cos(ring)[:, None] * rr + np.sin(ring)[:, None] * ahead)
            if wm.walkable_at(pts, np.full(len(pts), a[2])).all() or len(P) < 12:
                break
            if end == -1 and len(P) - 3 > i0 + 4:
                P = P[:-2]
            elif end == 0 and i0 > 6:
                P, i0 = P[2:], i0 - 2
            else:
                break
    # Directions along the street (smoothed over +-1 m) and their right-hand side.
    t = np.gradient(P[:, :2], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    t = _smooth_open(t, 4)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    r = np.column_stack([t[:, 1], -t[:, 0]])                    # Blender: right of the way
    n = len(P)
    # The loop: up the street on its right-hand lane, a half circle round the far end, back
    # on the other lane, a half circle round the near end. base + side * offset each.
    cap = np.linspace(0.0, np.pi, 12)[1:-1]

    def half_circle(e, side, ahead):
        return (np.repeat(e[None], len(cap), 0),
                np.cos(cap)[:, None] * side + np.sin(cap)[:, None] * ahead)

    b1, s1 = half_circle(P[-1], r[-1], t[-1])
    b2, s2 = half_circle(P[0], -r[0], -t[0])
    base = np.vstack([P, b1, P[::-1], b2])
    side = np.vstack([r, s1, -r[::-1], s2])
    m = len(base)
    seg = np.linalg.norm(np.diff(np.vstack([base[:, :2], base[:1, :2]]), axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = float(seg.sum()) + np.pi * LANE * 2                    # the half circles add about pi * lane each
    gap = np.abs(arc - arc[i0])
    near = np.minimum(gap, L - gap)
    fade = np.ones(m)                   # full lanes; pinning the start (below) puts the walker on its spot
    # How much of the lane each point may take: the most of (0, 1/3, 2/3, 1) that stays on
    # interior floors, then the least within +-1.5 m, averaged over +-1.5 m.
    allow = np.zeros(m)
    for frac in (1.0 / 3, 2.0 / 3, 1.0):
        trial = base[:, :2] + side * (LANE * fade * frac)[:, None]
        allow = np.where(wm.walkable_at(trial, base[:, 2]), frac, allow)
    in_cap = np.zeros(m, bool)
    in_cap[n:n + len(cap)] = True
    in_cap[2 * n + len(cap):] = True
    for _ in range(4):
        smooth = _window(_window(allow, 6, np.minimum), 6, np.add) / 13.0
        # A turn keeps at least a third of the lane (0.25 m): at radius 0 the walker spun
        # round on the spot, either way.
        smooth = np.where(in_cap, np.maximum(smooth, 1.0 / 3), smooth)
        loop = base.copy()
        loop[:, :2] += side * (LANE * fade * smooth)[:, None]
        nxt = np.roll(loop[:, :2], -1, 0)
        bad = ~(wm.walkable_at(loop[:, :2], loop[:, 2]) & wm.walkable_at(0.5 * (loop[:, :2] + nxt), loop[:, 2]))
        if not bad.any():
            break
        allow = np.where(_window(bad.astype(float), 2, np.add) > 0, np.maximum(allow - 1.0 / 3, 0.0), allow)
    if p0 is not None:
        # Pin the start to where the person stands: the grid and the smoothing moved it
        # up to a metre or so. The shift fades out over FADE m either way (cosine), or
        # over less where that would sweep the line across something solid (the person's
        # own 0.3 m is fine: the game put it there).
        shift = np.asarray(p0[:2]) - loop[i0, :2]
        for reach in (FADE, FADE / 2, FADE / 4):
            w = 0.5 + 0.5 * np.cos(np.pi * np.clip(near / reach, 0.0, 1.0))
            trial = loop[:, :2] + w[:, None] * shift
            check = (w > 0.0) & (np.linalg.norm(trial - p0[:2], axis=1) > 0.3)
            mid = 0.5 * (trial + np.roll(trial, -1, 0))
            if (wm.walkable_at(trial[check], loop[check, 2], tight=True).all()
                    and wm.walkable_at(mid[check], loop[check, 2], tight=True).all()):
                break
        loop[:, :2] = trial
    seg = np.linalg.norm(np.diff(np.vstack([loop[:, :2], loop[:1, :2]]), axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return loop, float(arc[-1]), float(arc[i0])
