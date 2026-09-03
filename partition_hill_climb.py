from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable
import heapq
import math
import random

"""
From the paper: "A Seed-Growth Heuristic for Graph Bisection" by Marks, Ruml, Shieber, and Ngo
"""

def build_adjacency(vertices: Iterable[Hashable], edges: Iterable[tuple[Hashable, Hashable]]):
    """
    Converts arbitrary vertex labels into indexed adjacency sets.

    Returns:
        labels: list where labels[i] is the original vertex label
        adj: adjacency sets using integer indices
    """
    labels = list(vertices)
    idx = {v: i for i, v in enumerate(labels)}

    adj = [set() for _ in labels]

    for u, v in edges:
        if u == v:
            continue

        if u not in idx or v not in idx:
            raise ValueError(f"Edge ({u}, {v}) contains a vertex not in V")

        i, j = idx[u], idx[v]
        adj[i].add(j)
        adj[j].add(i)

    return labels, adj


def cut_size(adj: list[set[int]], part: list[int]) -> int:
    """
    Counts edges crossing the partition.

    part[v] must be 0 or 1.
    """
    total = 0

    for u in range(len(adj)):
        for v in adj[u]:
            if u < v and part[u] != part[v]:
                total += 1

    return total


def random_seed_pair(
    n: int,
    seed_size: int,
    rng: random.Random,
) -> tuple[frozenset[int], frozenset[int]]:
    """
    Randomly chooses disjoint seed sets sx and sy.
    """
    chosen = rng.sample(range(n), 2 * seed_size)
    sx = frozenset(chosen[:seed_size])
    sy = frozenset(chosen[seed_size:])
    return sx, sy


def seed_growth_from_seeds(
    adj: list[set[int]],
    sx: frozenset[int],
    sy: frozenset[int],
    rng: random.Random,
    x_target: int | None = None,
) -> tuple[list[int], int]:
    """
    Runs the seed-growth heuristic from fixed seed sets.

    If x_target is given:
        final |X| = x_target
        final |Y| = n - x_target

    Otherwise:
        requires even n and performs a bisection.

    Returns:
        part: list where part[v] is 0 for X and 1 for Y
        score: cut size
    """
    n = len(adj)

    if x_target is None:
        if n % 2 != 0:
            raise ValueError("Graph bisection requires an even number of vertices.")
        x_target = n // 2

    y_target = n - x_target

    if x_target < 0 or y_target < 0:
        raise ValueError("Invalid target sizes.")

    if len(sx) > x_target:
        raise ValueError("sx has more vertices than x_target.")

    if len(sy) > y_target:
        raise ValueError("sy has more vertices than y_target.")

    part = [-1] * n
    assigned = [False] * n

    x_count = 0
    y_count = 0

    for v in sx:
        part[v] = 0
        assigned[v] = True
        x_count += 1

    for v in sy:
        part[v] = 1
        assigned[v] = True
        y_count += 1

    nx = [0] * n
    ny = [0] * n

    for v in sx:
        for w in adj[v]:
            if not assigned[w]:
                nx[w] += 1

    for v in sy:
        for w in adj[v]:
            if not assigned[w]:
                ny[w] += 1

    version = [0] * n

    heap_x = []
    heap_y = []

    def push_vertex(v: int):
        heapq.heappush(heap_x, (ny[v] - nx[v], rng.random(), version[v], v))
        heapq.heappush(heap_y, (nx[v] - ny[v], rng.random(), version[v], v))

    for v in range(n):
        if not assigned[v]:
            push_vertex(v)

    def pop_best_for_x() -> int:
        while heap_x:
            _, _, ver, v = heapq.heappop(heap_x)
            if not assigned[v] and ver == version[v]:
                return v
        raise RuntimeError("No valid vertex available for X.")

    def pop_best_for_y() -> int:
        while heap_y:
            _, _, ver, v = heapq.heappop(heap_y)
            if not assigned[v] and ver == version[v]:
                return v
        raise RuntimeError("No valid vertex available for Y.")

    def add_vertex(v: int, side: int):
        assigned[v] = True
        part[v] = side

        for w in adj[v]:
            if assigned[w]:
                continue

            if side == 0:
                nx[w] += 1
            else:
                ny[w] += 1

            version[w] += 1
            push_vertex(w)

    while x_count < x_target or y_count < y_target:
        if x_count < x_target:
            a = pop_best_for_x()
            add_vertex(a, 0)
            x_count += 1

        if y_count < y_target:
            b = pop_best_for_y()
            add_vertex(b, 1)
            y_count += 1

    score = cut_size(adj, part)
    return part, score
@dataclass
class Candidate:
    sx: frozenset[int]
    sy: frozenset[int]
    part: list[int]
    score: int


def mutate_seed_pair(
    sx: frozenset[int],
    sy: frozenset[int],
    n: int,
    rng: random.Random,
) -> tuple[frozenset[int], frozenset[int]]:
    """
    Replaces one random seed vertex with one non-seed vertex.
    """
    sx_new = set(sx)
    sy_new = set(sy)

    if not sx_new and not sy_new:
        return sx, sy

    mutate_x = rng.random() < 0.5

    if mutate_x and sx_new:
        old = rng.choice(tuple(sx_new))
        sx_new.remove(old)

        while True:
            new = rng.randrange(n)
            if new not in sx_new and new not in sy_new:
                sx_new.add(new)
                break

    elif sy_new:
        old = rng.choice(tuple(sy_new))
        sy_new.remove(old)

        while True:
            new = rng.randrange(n)
            if new not in sx_new and new not in sy_new:
                sy_new.add(new)
                break

    return frozenset(sx_new), frozenset(sy_new)


def phc_seed_growth(
    adj: list[set[int]],
    *,
    population_size: int = 100,
    iterations: int = 2500,
    seed_fraction: float = 0.01,
    rng_seed: int | None = None,
    x_target: int | None = None,
) -> tuple[list[int], int]:
    """
    Parallel hill-climbing wrapper around seed-growth.

    If x_target is given, returns a split with:
        |X| = x_target
        |Y| = n - x_target

    Otherwise, performs a balanced bisection.
    """
    n = len(adj)

    if x_target is None:
        if n % 2 != 0:
            raise ValueError("Graph bisection requires an even number of vertices.")
        x_target = n // 2

    y_target = n - x_target

    if x_target <= 0 or y_target <= 0:
        raise ValueError("Both sides of the split must be nonempty.")

    rng = random.Random(rng_seed)

    max_seed_size = min(x_target, y_target)

    seed_size = max(1, math.floor(seed_fraction * n))
    seed_size = min(seed_size, max_seed_size)

    population: list[Candidate] = []

    for _ in range(population_size):
        sx, sy = random_seed_pair(n, seed_size, rng)
        part, score = seed_growth_from_seeds(
            adj,
            sx,
            sy,
            rng,
            x_target=x_target,
        )
        population.append(Candidate(sx, sy, part, score))

    best = min(population, key=lambda c: c.score)
    best_part = best.part[:]
    best_score = best.score

    for _ in range(iterations):
        population.sort(key=lambda c: c.score)

        if len(population) == 1:
            weights = [1.0]
        else:
            weights = [
                4.0 - 3.0 * i / (len(population) - 1)
                for i in range(len(population))
            ]

        parent = rng.choices(population, weights=weights, k=1)[0]

        sx_new, sy_new = mutate_seed_pair(parent.sx, parent.sy, n, rng)

        part_new, score_new = seed_growth_from_seeds(
            adj,
            sx_new,
            sy_new,
            rng,
            x_target=x_target,
        )

        child = Candidate(sx_new, sy_new, part_new, score_new)
        population.append(child)

        if score_new < best_score:
            best_score = score_new
            best_part = part_new[:]

        worst_index = max(range(len(population)), key=lambda i: population[i].score)
        population.pop(worst_index)

    return best_part, best_score

# Recursive helper functions
def induced_subgraph(
    adj: list[set[int]],
    vertices: list[int],
) -> tuple[list[int], list[set[int]]]:
    """
    Builds the induced subgraph on a subset of original vertex indices.

    Returns:
        local_to_global: local index -> original index
        sub_adj: adjacency list using local indices
    """
    local_to_global = list(vertices)
    global_to_local = {v: i for i, v in enumerate(local_to_global)}

    sub_adj = [set() for _ in local_to_global]

    for local_u, global_u in enumerate(local_to_global):
        for global_v in adj[global_u]:
            if global_v in global_to_local:
                local_v = global_to_local[global_v]
                sub_adj[local_u].add(local_v)

    return local_to_global, sub_adj


def recursive_bisect_indices(
    adj: list[set[int]],
    vertices: list[int],
    k: int,
    *,
    population_size: int = 100,
    iterations: int = 2500,
    seed_fraction: float = 0.01,
    rng_seed: int | None = None,
) -> list[frozenset[int]]:
    """
    Recursively splits a vertex set into k parts using PHC/SG.

    Returns:
        list of parts, where each part is a frozenset of original vertex indices.
    """
    if k <= 1 or len(vertices) <= 1:
        return [frozenset(vertices)]

    if k > len(vertices):
        raise ValueError("Cannot split into more parts than vertices.")

    rng = random.Random(rng_seed)

    left_k = k // 2
    right_k = k - left_k

    n = len(vertices)

    # Choose the split size proportional to how many final parts each side needs.
    left_size = round(n * left_k / k)

    # Make sure both recursive sides can receive enough vertices.
    left_size = max(left_k, left_size)
    left_size = min(n - right_k, left_size)

    local_to_global, sub_adj = induced_subgraph(adj, vertices)

    split_seed = rng.randrange(10**18)

    part, _ = phc_seed_growth(
        sub_adj,
        population_size=population_size,
        iterations=iterations,
        seed_fraction=seed_fraction,
        rng_seed=split_seed,
        x_target=left_size,
    )

    left_vertices = []
    right_vertices = []

    for local_i, side in enumerate(part):
        global_i = local_to_global[local_i]

        if side == 0:
            left_vertices.append(global_i)
        else:
            right_vertices.append(global_i)

    left_parts = recursive_bisect_indices(
        adj,
        left_vertices,
        left_k,
        population_size=population_size,
        iterations=iterations,
        seed_fraction=seed_fraction,
        rng_seed=rng.randrange(10**18),
    )

    right_parts = recursive_bisect_indices(
        adj,
        right_vertices,
        right_k,
        population_size=population_size,
        iterations=iterations,
        seed_fraction=seed_fraction,
        rng_seed=rng.randrange(10**18),
    )

    return left_parts + right_parts


def multiway_cut_size(
    adj: list[set[int]],
    parts: list[frozenset[int]],
) -> int:
    """
    Counts the number of edges crossing between different parts.
    """
    part_id = {}

    for i, part in enumerate(parts):
        for v in part:
            part_id[v] = i

    total = 0

    for u in range(len(adj)):
        for v in adj[u]:
            if u < v and part_id[u] != part_id[v]:
                total += 1

    return total

def recursive_partition_graph(
    vertices: Iterable[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    k: int,
    population_size: int = 100,
    iterations: int = 2500,
    seed_fraction: float = 0.01,
    rng_seed: int | None = None,
):
    """
    Recursively partitions the graph into k parts.

    Returns:
        parts: list of sets of original vertex labels
        score: number of edges crossing between parts
    """
    labels, adj = build_adjacency(vertices, edges)

    n = len(labels)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > n:
        raise ValueError("k cannot exceed the number of vertices.")

    index_parts = recursive_bisect_indices(
        adj,
        list(range(n)),
        k,
        population_size=population_size,
        iterations=iterations,
        seed_fraction=seed_fraction,
        rng_seed=rng_seed,
    )

    parts = [
        {labels[i] for i in part}
        for part in index_parts
    ]

    score = multiway_cut_size(adj, index_parts)

    return parts, score


def partition_graph(
    vertices: Iterable[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    population_size: int = 100,
    iterations: int = 2500,
    seed_fraction: float = 0.01,
    rng_seed: int | None = None,
):
    """
    User-facing wrapper.

    Returns:
        X: set of original vertex labels
        Y: set of original vertex labels
        score: cut size
    """
    labels, adj = build_adjacency(vertices, edges)

    part, score = phc_seed_growth(
        adj,
        population_size=population_size,
        iterations=iterations,
        seed_fraction=seed_fraction,
        rng_seed=rng_seed,
    )

    X = {labels[i] for i, side in enumerate(part) if side == 0}
    Y = {labels[i] for i, side in enumerate(part) if side == 1}

    return X, Y, score