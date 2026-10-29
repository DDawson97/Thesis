"""Crossing-mass-aware force-directed graph drawing.

First obtain a Fruchterman-Reingold layout, then improve the *maximum vertex
crossing mass* by searching for vertex moves that uncross edge pairs.

M_D(v) = sum_{e incident to v} crossings_D(e)
M(D)   = max_v M_D(v)

Assumptions: simple undirected graph, straight-line edges, distinct vertex
positions, and proper interior edge crossings. Tangencies/collinear overlaps
are not counted; general-position inputs are recommended.

Install: pip install networkx matplotlib
Run:     python mass_force_directed.py
"""

import math
import random

import matplotlib.pyplot as plt
import networkx as nx


def force_directed_layout(graph, iterations=300, area=1.0, seed=42):
    """Baseline Fruchterman-Reingold implementation; no layout library used."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    if area <= 0 or iterations <= 0:
        raise ValueError('area and iterations must be positive')

    k = math.sqrt(area / n)
    side = math.sqrt(area)
    temperature_0 = 0.1 * side
    pos = {v: [rng.uniform(-side / 2, side / 2),
               rng.uniform(-side / 2, side / 2)] for v in nodes}
    edges = list(graph.edges())

    for t in range(iterations):
        delta = {v: [0.0, 0.0] for v in nodes}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                dx, dy = pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]
                dist = math.hypot(dx, dy)
                if dist < 1e-12:
                    theta = rng.uniform(0, 2 * math.pi)
                    dx, dy, dist = math.cos(theta), math.sin(theta), 1.0
                force = k * k / dist
                fx, fy = force * dx / dist, force * dy / dist
                delta[u][0] += fx
                delta[u][1] += fy
                delta[v][0] -= fx
                delta[v][1] -= fy
        for u, v in edges:
            dx, dy = pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                continue
            force = dist * dist / k
            fx, fy = force * dx / dist, force * dy / dist
            delta[u][0] -= fx
            delta[u][1] -= fy
            delta[v][0] += fx
            delta[v][1] += fy
        cap = temperature_0 * (1 - t / iterations)
        for v in nodes:
            dx, dy = delta[v]
            size = math.hypot(dx, dy)
            if size > 1e-12:
                step = min(size, cap)
                pos[v][0] += step * dx / size
                pos[v][1] += step * dy / size
    return _normalize(pos)


def _normalize(pos):
    if not pos:
        return {}
    cx = sum(x for x, y in pos.values()) / len(pos)
    cy = sum(y for x, y in pos.values()) / len(pos)
    scale = max((max(abs(x - cx), abs(y - cy)) for x, y in pos.values()),
                default=1.0)
    scale = max(scale, 1e-12)
    return {v: ((p[0] - cx) / scale, (p[1] - cy) / scale)
            for v, p in pos.items()}


def _orientation(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))

def _proper_cross(a, b, c, d):
    """Whether open segments ab and cd cross transversely (not at endpoints)."""
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (ab_c * ab_d < -1e-24 and cd_a * cd_b < -1e-24)


def crossing_mass(graph, pos):
    """Return (maximum mass, per-vertex mass, total crossings, crossing edge pairs)."""
    edges = list(graph.edges())
    mass = {v: 0 for v in graph.nodes()}
    crossing_pairs = []
    for i, (u, v) in enumerate(edges):
        for a, b in edges[i + 1:]:
            if len({u, v, a, b}) < 4:
                continue  # Adjacent edges share an endpoint, not a proper crossing.
            if _proper_cross(pos[u], pos[v], pos[a], pos[b]):
                crossing_pairs.append(((u, v), (a, b)))
                for node in (u, v, a, b):
                    mass[node] += 1
    return max(mass.values(), default=0), mass, len(crossing_pairs), crossing_pairs


def _score(mass, crossings):
    """Lexicographic: first min max mass, then total crossings, then imbalance."""
    return (max(mass.values(), default=0), crossings,
            sum(v * v for v in mass.values()))


def _uncross_target(pos, v, partner, a, b, margin, reflect=False):
    """Move v to the same side of line (a,b) as its edge's other endpoint.

    This removes the specific crossing (v,partner) x (a,b) in general position.
    All *other* crossings must still be checked before accepting a move.
    """
    ax, ay = pos[a]
    bx, by = pos[b]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    nx, ny = -dy / length, dx / length
    sx = (pos[v][0] - ax) * nx + (pos[v][1] - ay) * ny
    sp = (pos[partner][0] - ax) * nx + (pos[partner][1] - ay) * ny
    if abs(sp) < 1e-12:
        return None
    sign = 1 if sp > 0 else -1
    target_signed = sign * (max(abs(sx), margin) if reflect else margin)
    shift = target_signed - sx
    return (pos[v][0] + shift * nx, pos[v][1] + shift * ny)


def _try_move(v, target, pos, edges, incident, mass, crossing_count):
    """Evaluate one vertex move in O(deg(v)*|E|), using exact mass deltas.

    Only crossing pairs containing an edge incident to v can change.
    """
    trial_mass = mass.copy()
    trial_count = crossing_count
    old_v = pos[v]

    for u, w in incident[v]:
        # Only compare with independent edges; crossings of adjacent edges
        # are excluded by the geometric graph drawing convention.
        for a, b in edges:
            if len({u, w, a, b}) < 4:
                continue
            old_cross = _proper_cross(pos[u], pos[w], pos[a], pos[b])
            pu = target if u == v else pos[u]
            pw = target if w == v else pos[w]
            new_cross = _proper_cross(pu, pw, pos[a], pos[b])
            change = int(new_cross) - int(old_cross)
            if change:
                trial_count += change
                for endpoint in (u, w, a, b):
                    trial_mass[endpoint] += change

    return _score(trial_mass, trial_count), trial_mass, trial_count

def mass_force_directed_layout(graph, iterations=300, improvement_rounds=250,
                               crossings_per_round=10, seed=42):
    """Hybrid FR + discrete crossing-mass local search.

    Minimizes the lexicographic score (max vertex mass, crossing count,
    sum of squared vertex masses). Accepts only strictly improving single-
    vertex moves; thus improvement isn't guaranteed to reach a global optimum.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError('Expected a simple undirected NetworkX graph')
    if improvement_rounds < 0 or crossings_per_round < 1:
        raise ValueError('Invalid local-search settings')

    rng = random.Random(seed)
    pos = {v: list(pt) for v, pt in
           force_directed_layout(graph, iterations=iterations, seed=seed).items()}
    if not pos:
        return {}

    edges = list(graph.edges())
    incident = {v: [] for v in graph.nodes()}
    for u, v in edges:
        if u != v:
            incident[u].append((u, v))
            incident[v].append((u, v))

    _, mass, count, pairs = crossing_mass(graph, pos)
    score = _score(mass, count)
    no_progress = 0

    for t in range(improvement_rounds):
        if not pairs:
            break
        best_move = None
        best_score = score
        best_mass = None
        best_count = count

        # Favor crossings incident to highly burdened vertices, but sample
        # some others as well to avoid repeatedly checking identical moves.
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda pair: max(mass[x] for e in pair for x in e),
                      reverse=True)
        candidates = shuffled[:crossings_per_round]

        for (u, v), (a, b) in candidates:
            for moved, partner, other_a, other_b in (
                (u, v, a, b), (v, u, a, b),
                (a, b, u, v), (b, a, u, v),
            ):
                for reflect in (False, True):
                    target = _uncross_target(pos, moved, partner,
                                             other_a, other_b,
                                             margin=0.035, reflect=reflect)
                    if target is None:
                        continue
                    trial_score, trial_mass, trial_count = _try_move(
                        moved, target, pos, edges, incident, mass, count)
                    if trial_score < best_score:
                        best_move = (moved, target)
                        best_score = trial_score
                        best_mass = trial_mass
                        best_count = trial_count

        # Additional random exploration near the most burdened vertices.
        # This helps when an uncrossing move alone is too restrictive.
        worst_nodes = sorted(pos, key=lambda v: mass[v], reverse=True)[:4]
        radius = 0.24 * (1 - t / max(1, improvement_rounds)) + 0.02
        for v in worst_nodes:
            for _ in range(5):
                angle = rng.uniform(0, 2 * math.pi)
                r = radius * rng.uniform(0.2, 1)
                target = (pos[v][0] + r * math.cos(angle),
                          pos[v][1] + r * math.sin(angle))
                trial_score, trial_mass, trial_count = _try_move(
                    v, target, pos, edges, incident, mass, count)
                if trial_score < best_score:
                    best_move = (v, target)
                    best_score = trial_score
                    best_mass = trial_mass
                    best_count = trial_count

        if best_move is None:
            no_progress += 1
            if no_progress >= 25:
                break
            continue
        no_progress = 0
        moved, target = best_move
        pos[moved] = list(target)
        mass, count, score = best_mass, best_count, best_score
        _, checked_mass, checked_count, pairs = crossing_mass(graph, pos)
        assert checked_mass == mass and checked_count == count

    return _normalize(pos)

def main():
    # Petersen graph: nonplanar, so some crossings are unavoidable.
    graph = nx.petersen_graph()
    before = force_directed_layout(graph, seed=8)
    after = mass_force_directed_layout(graph, seed=8)

    for name, positions in [('FR', before), ('Mass-aware', after)]:
        maxmass, masses, crossings, _ = crossing_mass(graph, positions)
        print(f'{name}: maximum vertex mass={maxmass}, '
              f'crossings={crossings}, masses={masses}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, title, positions in zip(
        axes, ['Original force-directed', 'Crossing-mass-aware'],
        [before, after]
    ):
        maxmass, masses, crossings, _ = crossing_mass(graph, positions)
        nx.draw_networkx(graph, pos=positions, ax=ax, node_size=350,
                         node_color=[masses[v] for v in graph], cmap='OrRd',
                         vmin=0, vmax=max(1, maxmass), edge_color='#64748b',
                         font_size=9)
        ax.set_title(f'{title}\nM(D)={maxmass}, crossings={crossings}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
