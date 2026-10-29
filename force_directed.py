"""Fruchterman-Reingold force-directed graph drawing, implemented from scratch.

Install dependencies:
    pip install networkx matplotlib

Run the demo:
    python force_directed.py

To use in code:
    pos = force_directed_layout(G, iterations=500, seed=7)
"""

import math
import random

import matplotlib.pyplot as plt
import networkx as nx


def force_directed_layout(graph, iterations=300, area=1.0, seed=42, initial_pos=None):
    """Compute 2D positions for a NetworkX graph using Fruchterman-Reingold.

    Repulsive force between every pair of vertices:  f_r(d) = k^2 / d
    Attractive force along every edge:                f_a(d) = d^2 / k
    where k = sqrt(area / number_of_vertices).

    Parameters
    ----------
    graph : networkx.Graph
        Graph to lay out; nodes can have any hashable labels.
    iterations : int
        Number of rounds of force calculation and position updates.
    area : float
        Desired (relative) layout area; controls the natural edge length.
    seed : int
        Seed for reproducible initial locations and overlap handling.
    initial_pos : dict, optional
        Mapping node -> (x, y) for initial locations. Missing nodes get
        randomized positions.

    Returns
    -------
    dict
        Mapping node -> (x, y), centered and uniformly scaled to [-1, 1].

    Complexity
    ----------
    O(iterations * (V^2 + E)) time, O(V + E) auxiliary space.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if area <= 0:
        raise ValueError("area must be positive")

    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    side = math.sqrt(area)
    k = math.sqrt(area / n)   # Ideal spacing between connected vertices.
    initial_temperature = 0.1 * side

    pos = {}
    for u in nodes:
        if initial_pos is not None and u in initial_pos:
            x, y = initial_pos[u]
            pos[u] = [float(x), float(y)]
        else:
            pos[u] = [rng.uniform(-side / 2, side / 2),
                      rng.uniform(-side / 2, side / 2)]

    # Convert directed edges into undirected springs; a self-loop exerts no force.
    spring_graph = graph.to_undirected() if graph.is_directed() else graph
    edges = [(u, v) for u, v in spring_graph.edges() if u != v]

    for iteration in range(iterations):
        displacement = {u: [0.0, 0.0] for u in nodes}

        # (1) Every pair of vertices repels each other.
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                distance = math.hypot(dx, dy)

                if distance < 1e-12:
                    # Distinct vertices need a direction in which to separate.
                    angle = rng.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(angle), math.sin(angle)
                    distance = 1.0

                force = (k * k) / distance
                fx = (dx / distance) * force
                fy = (dy / distance) * force

                displacement[u][0] += fx
                displacement[u][1] += fy
                displacement[v][0] -= fx
                displacement[v][1] -= fy

        # (2) Endpoints of each edge attract each other.
        for u, v in edges:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            distance = math.hypot(dx, dy)
            if distance < 1e-12:
                continue

            force = (distance * distance) / k
            fx = (dx / distance) * force
            fy = (dy / distance) * force

            displacement[u][0] -= fx
            displacement[u][1] -= fy
            displacement[v][0] += fx
            displacement[v][1] += fy

        # (3) Move in the net force direction, but cap the step size.
        temperature = initial_temperature * (1 - iteration / iterations)
        for u in nodes:
            dx, dy = displacement[u]
            magnitude = math.hypot(dx, dy)
            if magnitude > 1e-12:
                step = min(magnitude, temperature)
                pos[u][0] += (dx / magnitude) * step
                pos[u][1] += (dy / magnitude) * step

    # Center and uniformly rescale the picture without distorting its aspect ratio.
    cx = sum(pos[u][0] for u in nodes) / n
    cy = sum(pos[u][1] for u in nodes) / n
    scale = max(max(abs(pos[u][0] - cx), abs(pos[u][1] - cy)) for u in nodes)
    if scale < 1e-12:
        scale = 1.0

    return {u: ((pos[u][0] - cx) / scale, (pos[u][1] - cy) / scale)
            for u in nodes}


def main():
    # Example: Zachary's karate club graph (34 vertices, 78 edges).
    G = nx.karate_club_graph()
    positions = force_directed_layout(G, iterations=400, seed=7)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx(
        G,
        pos=positions,
        node_size=330,
        font_size=8,
        node_color="#b9ddf7",
        edge_color="#64748b",
        width=1.1,
    )
    plt.title("Fruchterman-Reingold Force-Directed Layout")
    plt.axis("off")
    plt.tight_layout()
