import csv
import random
import statistics
from time import perf_counter

from greedy_circular_generation import greedy_generation
from improving_an_order import improve_order, edge_crossing_numbers
from annealing_order import simulated_annealing_order
from partition_hill_climb import partition_graph


# ------------------------------------------------------------
# Known graph generators
# ------------------------------------------------------------

def cycle_graph(n):
    V = list(range(n))
    E = [(i, (i + 1) % n) for i in range(n)]
    return f"C_{n}", V, E


def path_graph(n):
    V = list(range(n))
    E = [(i, i + 1) for i in range(n - 1)]
    return f"P_{n}", V, E


def complete_graph(n):
    V = list(range(n))
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return f"K_{n}", V, E


def complete_bipartite_graph(m, n):
    left = [f"a{i}" for i in range(m)]
    right = [f"b{j}" for j in range(n)]
    V = left + right
    E = [(u, v) for u in left for v in right]
    return f"K_{{{m},{n}}}", V, E


def wheel_graph(n):
    if n < 4:
        raise ValueError("Wheel graph needs n >= 4")

    V = list(range(n))
    E = [(0, i) for i in range(1, n)]

    for i in range(1, n):
        E.append((i, 1 if i == n - 1 else i + 1))

    return f"W_{n}", V, E


def grid_graph(rows, cols):
    V = [(r, c) for r in range(rows) for c in range(cols)]
    E = []

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                E.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                E.append(((r, c), (r, c + 1)))

    return f"Grid_{rows}x{cols}", V, E


def petersen_graph():
    V = list(range(10))

    outer = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0)
    ]
    inner = [
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)
    ]
    spokes = [
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    return "Petersen", V, outer + inner + spokes


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------

def normalize_order_result(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def edge_crossing_vector(order, edges):
    """
    Lexicographic tie-breaker.

    The first component is still k, the maximum edge-crossing count.
    Later components distinguish orders that have the same k.
    """
    counts = edge_crossing_numbers(order, edges)
    return tuple(sorted(counts.values(), reverse=True))


def score_order(order, edges):
    counts = edge_crossing_numbers(order, edges)
    return max(counts.values(), default=0)


def pct_improvement(before, after):
    if before == 0:
        return 0.0
    return 100.0 * (before - after) / before


def random_order(vertices, rng):
    order = list(vertices)
    rng.shuffle(order)
    return order


def relabel_graph(vertices, edges, rng):
    shuffled = list(vertices)
    rng.shuffle(shuffled)

    mapping = {
        old: f"v{i}"
        for i, old in enumerate(shuffled)
    }

    V2 = [mapping[v] for v in vertices]
    E2 = [(mapping[u], mapping[v]) for u, v in edges]
    return V2, E2


def run_sa_order(V, E, omega_start, *, max_steps, alpha, seed):
    result = simulated_annealing_order(
        V,
        E,
        omega_start,
        max_steps=max_steps,
        alpha=alpha,
        seed=seed,
    )
    return normalize_order_result(result)


# ------------------------------------------------------------
# Partition construction
# ------------------------------------------------------------

def partition_then_order(
    V,
    E,
    omega_start,
    *,
    rng_seed=None,
    population_size=100,
    iterations=2500,
):
    """
    Construct a circular order from a graph bisection.

    omega_start controls only the order retained inside X and Y.
    For a genuinely independent Partition pipeline, pass a random order,
    not the greedy order.
    """
    X, Y, _ = partition_graph(
        V,
        E,
        population_size=population_size,
        iterations=iterations,
        rng_seed=rng_seed,
    )

    X = set(X)
    Y = set(Y)

    omega_xy = (
        [v for v in omega_start if v in X]
        + [v for v in omega_start if v in Y]
    )
    omega_yx = (
        [v for v in omega_start if v in Y]
        + [v for v in omega_start if v in X]
    )

    if edge_crossing_vector(omega_yx, E) < edge_crossing_vector(omega_xy, E):
        return omega_yx

    return omega_xy


# ------------------------------------------------------------
# Offensiveness heuristic
# ------------------------------------------------------------

def build_incident_edges(vertices, edges):
    incident = {vertex: [] for vertex in vertices}

    for edge in edges:
        u, v = edge
        incident[u].append(edge)
        incident[v].append(edge)

    return incident


def offensiveness(vertices, edges, omega):
    raw_counts = edge_crossing_numbers(omega, edges)
    crossing_counts = normalize_crossing_counts(raw_counts)

    incident = build_incident_edges(vertices, edges)
    offensive_scores = {}

    for vertex in vertices:
        vertex_edges = incident[vertex]

        if not vertex_edges:
            offensive_scores[vertex] = 0.0
            continue

        total_crossings = sum(
            crossing_counts.get(frozenset(edge), 0)
            for edge in vertex_edges
        )

        offensive_scores[vertex] = (
            total_crossings / len(vertex_edges)
        )

    return offensive_scores

def normalize_crossing_counts(crossing_counts):
    """
    Normalize undirected edge keys so (u, v) and (v, u)
    are treated as the same edge.
    """
    return {
        frozenset(edge): count
        for edge, count in crossing_counts.items()
    }

def candidate_positions_by_offensiveness(order, scores):
    """
    Rank circular gaps from lowest surrounding offensiveness to highest.

    Position i represents the gap between order[i - 1] and order[i].
    """
    if not order:
        return [0]

    positions = []

    for position in range(len(order)):
        left = order[position - 1]
        right = order[position]
        gap_score = (scores[left] + scores[right]) / 2.0
        positions.append((gap_score, position))

    positions.sort()
    return [position for _, position in positions]


def best_reinsertion(vertex, vertices, edges, order):
    """
    Try reinserting one vertex into every circular gap.

    Candidate gaps are examined in low-offensiveness order, but the chosen
    move is determined by the actual edge-crossing vector.
    """
    reduced_order = [v for v in order if v != vertex]

    best_order = list(order)
    best_vector = edge_crossing_vector(order, edges)

    if not reduced_order:
        return best_order, score_order(best_order, edges)

    scores = offensiveness(vertices, edges, order)
    positions = candidate_positions_by_offensiveness(reduced_order, scores)

    for position in positions:
        candidate = (
            reduced_order[:position]
            + [vertex]
            + reduced_order[position:]
        )
        candidate_vector = edge_crossing_vector(candidate, edges)

        if candidate_vector < best_vector:
            best_order = candidate
            best_vector = candidate_vector

    return best_order, score_order(best_order, edges)


def offensive_improve(
    vertices,
    edges,
    initial_order,
    *,
    max_rounds=100,
):
    """
    Repeatedly rank vertices by offensiveness and accept the first
    improving single-vertex reinsertion.
    """
    current_order = list(initial_order)
    current_vector = edge_crossing_vector(current_order, edges)

    for _ in range(max_rounds):
        scores = offensiveness(vertices, edges, current_order)

        ranked_vertices = sorted(
            vertices,
            key=lambda vertex: scores[vertex],
            reverse=True,
        )

        improved = False

        for vertex in ranked_vertices:
            candidate_order, _ = best_reinsertion(
                vertex,
                vertices,
                edges,
                current_order,
            )
            candidate_vector = edge_crossing_vector(candidate_order, edges)

            if candidate_vector < current_vector:
                current_order = candidate_order
                current_vector = candidate_vector
                improved = True
                break

        if not improved:
            break

    return current_order


# ------------------------------------------------------------
# Timed pipeline helpers
# ------------------------------------------------------------

def timed_order_operation(operation):
    t0 = perf_counter()
    order = normalize_order_result(operation())
    t1 = perf_counter()
    return order, 1000.0 * (t1 - t0)


def store_pipeline(row, name, start_k, order, local_time_ms, prefix_time_ms):
    final_k = score_order(order, row["_edges"])

    row[f"{name}_k"] = final_k
    row[f"{name}_local_time_ms"] = local_time_ms
    row[f"{name}_total_time_ms"] = prefix_time_ms + local_time_ms
    row[f"{name}_improvement_pct"] = pct_improvement(start_k, final_k)

    return final_k


def store_sa_pipeline(
    row,
    name,
    baseline_k,
    order,
    *,
    V,
    E,
    max_steps,
    alpha,
    seed,
    prefix_time_ms,
):
    sa_order, sa_time_ms = timed_order_operation(
        lambda: run_sa_order(
            V,
            E,
            order,
            max_steps=max_steps,
            alpha=alpha,
            seed=seed,
        )
    )

    sa_k = score_order(sa_order, E)

    row[f"{name}_sa_k"] = sa_k
    row[f"{name}_sa_local_time_ms"] = sa_time_ms
    row[f"{name}_sa_total_time_ms"] = prefix_time_ms + sa_time_ms
    row[f"{name}_sa_improvement_pct"] = pct_improvement(baseline_k, sa_k)
    row[f"{name}_sa_gain_pct"] = pct_improvement(
        score_order(order, E),
        sa_k,
    )


# ------------------------------------------------------------
# One benchmark trial
# ------------------------------------------------------------

def run_single_trial(
    graph_name,
    V,
    E,
    trial,
    *,
    seed,
    max_rounds=100,
    offensive_max_rounds=100,
    sa_max_steps=5000,
    sa_alpha=0.995,
    partition_population=100,
    partition_iterations=2500,
):
    """
    Tests eight pipelines:

    1) Greedy -> Improve
    2) Greedy -> Improve -> SA
    3) Partition -> Improve
    4) Partition -> Improve -> SA
    5) Greedy -> Offensive
    6) Greedy -> Offensive -> SA
    7) Partition -> Offensive
    8) Partition -> Offensive -> SA
    """
    row = {
        "graph": graph_name,
        "trial": trial,
        "n": len(V),
        "m": len(E),
        "seed": seed,
        "_edges": E,
    }

    rng = random.Random(seed)

    # Shared Greedy start
    omega_greedy, greedy_time_ms = timed_order_operation(
        lambda: greedy_generation(V, E)
    )
    greedy_k = score_order(omega_greedy, E)

    row["greedy_start_k"] = greedy_k
    row["greedy_start_time_ms"] = greedy_time_ms

    # Shared Partition start, independent of Greedy
    partition_seed_order = random_order(V, rng)

    omega_partition, partition_time_ms = timed_order_operation(
        lambda: partition_then_order(
            V,
            E,
            partition_seed_order,
            rng_seed=seed,
            population_size=partition_population,
            iterations=partition_iterations,
        )
    )
    partition_k = score_order(omega_partition, E)

    row["partition_start_k"] = partition_k
    row["partition_start_time_ms"] = partition_time_ms

    # 1) Greedy -> Improve
    greedy_improve, local_ms = timed_order_operation(
        lambda: improve_order(
            V,
            E,
            omega_greedy,
            max_rounds=max_rounds,
        )
    )
    greedy_improve_k = store_pipeline(
        row,
        "greedy_improve",
        greedy_k,
        greedy_improve,
        local_ms,
        greedy_time_ms,
    )
    store_sa_pipeline(
        row,
        "greedy_improve",
        greedy_k,
        greedy_improve,
        V=V,
        E=E,
        max_steps=sa_max_steps,
        alpha=sa_alpha,
        seed=seed,
        prefix_time_ms=greedy_time_ms + local_ms,
    )

    # 2) Partition -> Improve
    partition_improve, local_ms = timed_order_operation(
        lambda: improve_order(
            V,
            E,
            omega_partition,
            max_rounds=max_rounds,
        )
    )
    partition_improve_k = store_pipeline(
        row,
        "partition_improve",
        partition_k,
        partition_improve,
        local_ms,
        partition_time_ms,
    )
    store_sa_pipeline(
        row,
        "partition_improve",
        partition_k,
        partition_improve,
        V=V,
        E=E,
        max_steps=sa_max_steps,
        alpha=sa_alpha,
        seed=seed,
        prefix_time_ms=partition_time_ms + local_ms,
    )

    # 3) Greedy -> Offensive
    greedy_offensive, local_ms = timed_order_operation(
        lambda: offensive_improve(
            V,
            E,
            omega_greedy,
            max_rounds=offensive_max_rounds,
        )
    )
    greedy_offensive_k = store_pipeline(
        row,
        "greedy_offensive",
        greedy_k,
        greedy_offensive,
        local_ms,
        greedy_time_ms,
    )
    store_sa_pipeline(
        row,
        "greedy_offensive",
        greedy_k,
        greedy_offensive,
        V=V,
        E=E,
        max_steps=sa_max_steps,
        alpha=sa_alpha,
        seed=seed,
        prefix_time_ms=greedy_time_ms + local_ms,
    )

    # 4) Partition -> Offensive
    partition_offensive, local_ms = timed_order_operation(
        lambda: offensive_improve(
            V,
            E,
            omega_partition,
            max_rounds=offensive_max_rounds,
        )
    )
    partition_offensive_k = store_pipeline(
        row,
        "partition_offensive",
        partition_k,
        partition_offensive,
        local_ms,
        partition_time_ms,
    )
    store_sa_pipeline(
        row,
        "partition_offensive",
        partition_k,
        partition_offensive,
        V=V,
        E=E,
        max_steps=sa_max_steps,
        alpha=sa_alpha,
        seed=seed,
        prefix_time_ms=partition_time_ms + local_ms,
    )

    del row["_edges"]
    return row


# ------------------------------------------------------------
# Summary and output
# ------------------------------------------------------------

PIPELINES = [
    "greedy_improve",
    "partition_improve",
    "greedy_offensive",
    "partition_offensive",
]


def summarize_results(rows):
    by_graph = {}

    for row in rows:
        by_graph.setdefault(row["graph"], []).append(row)

    summary = []

    for graph, graph_rows in by_graph.items():
        def mean(key):
            return statistics.mean(row[key] for row in graph_rows)

        def median(key):
            return statistics.median(row[key] for row in graph_rows)

        first = graph_rows[0]

        item = {
            "graph": graph,
            "n": first["n"],
            "m": first["m"],
            "trials": len(graph_rows),
            "median_greedy_start_k": median("greedy_start_k"),
            "median_partition_start_k": median("partition_start_k"),
            "mean_greedy_start_time_ms": mean("greedy_start_time_ms"),
            "mean_partition_start_time_ms": mean("partition_start_time_ms"),
        }

        for pipeline in PIPELINES:
            item[f"median_{pipeline}_k"] = median(f"{pipeline}_k")
            item[f"median_{pipeline}_sa_k"] = median(f"{pipeline}_sa_k")
            item[f"mean_{pipeline}_total_time_ms"] = mean(
                f"{pipeline}_total_time_ms"
            )
            item[f"mean_{pipeline}_sa_total_time_ms"] = mean(
                f"{pipeline}_sa_total_time_ms"
            )
            item[f"mean_{pipeline}_improvement_pct"] = mean(
                f"{pipeline}_improvement_pct"
            )
            item[f"mean_{pipeline}_sa_improvement_pct"] = mean(
                f"{pipeline}_sa_improvement_pct"
            )
            item[f"mean_{pipeline}_sa_gain_pct"] = mean(
                f"{pipeline}_sa_gain_pct"
            )

        summary.append(item)

    return summary


def print_summary(summary):
    print()
    print("Benchmark summary")
    print("-" * 142)

    header = (
        f"{'Graph':<16}"
        f"{'G->I':>8}"
        f"{'G->I+SA':>10}"
        f"{'P->I':>8}"
        f"{'P->I+SA':>10}"
        f"{'G->O':>8}"
        f"{'G->O+SA':>10}"
        f"{'P->O':>8}"
        f"{'P->O+SA':>10}"
        f"{'G->I ms':>11}"
        f"{'P->I ms':>11}"
        f"{'G->O ms':>11}"
        f"{'P->O ms':>11}"
    )

    print(header)
    print("-" * 142)

    for row in summary:
        print(
            f"{row['graph']:<16}"
            f"{row['median_greedy_improve_k']:>8.2f}"
            f"{row['median_greedy_improve_sa_k']:>10.2f}"
            f"{row['median_partition_improve_k']:>8.2f}"
            f"{row['median_partition_improve_sa_k']:>10.2f}"
            f"{row['median_greedy_offensive_k']:>8.2f}"
            f"{row['median_greedy_offensive_sa_k']:>10.2f}"
            f"{row['median_partition_offensive_k']:>8.2f}"
            f"{row['median_partition_offensive_sa_k']:>10.2f}"
            f"{row['mean_greedy_improve_total_time_ms']:>11.2f}"
            f"{row['mean_partition_improve_total_time_ms']:>11.2f}"
            f"{row['mean_greedy_offensive_total_time_ms']:>11.2f}"
            f"{row['mean_partition_offensive_total_time_ms']:>11.2f}"
        )


def write_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main benchmark suite
# ------------------------------------------------------------

def main():
    trials_per_graph = 5
    max_rounds = 1000
    offensive_max_rounds = 100
    base_seed = 12345

    sa_max_steps = 20_000
    sa_alpha = 0.995

    partition_population = 100
    partition_iterations = 2500

    graphs = [
        path_graph(20),
        cycle_graph(20),
        wheel_graph(20),
        grid_graph(4, 5),
        petersen_graph(),
        complete_graph(6),
        complete_graph(8),
        complete_bipartite_graph(3, 3),
        complete_bipartite_graph(4, 4),
        complete_bipartite_graph(5, 5),
    ]

    all_rows = []

    for graph_index, (graph_name, V, E) in enumerate(graphs):
        print(f"Running {graph_name} with n={len(V)}, m={len(E)}")

        for trial in range(trials_per_graph):
            seed = base_seed + 1000 * graph_index + trial
            rng = random.Random(seed)

            V_trial, E_trial = relabel_graph(V, E, rng)

            row = run_single_trial(
                graph_name=graph_name,
                V=V_trial,
                E=E_trial,
                trial=trial,
                seed=seed,
                max_rounds=max_rounds,
                offensive_max_rounds=offensive_max_rounds,
                sa_max_steps=sa_max_steps,
                sa_alpha=sa_alpha,
                partition_population=partition_population,
                partition_iterations=partition_iterations,
            )

            all_rows.append(row)

    summary = summarize_results(all_rows)

    write_csv("benchmark_raw.csv", all_rows)
    write_csv("benchmark_summary.csv", summary)

    print_summary(summary)

    print()
    print("Wrote:")
    print("  benchmark_raw.csv")
    print("  benchmark_summary.csv")


if __name__ == "__main__":
    main()
