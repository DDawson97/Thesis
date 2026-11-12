#!/usr/bin/env python3
"""Reproducible Rome-graph benchmark: circular offensiveness vs force-directed layouts.

Dependencies: networkx, matplotlib
Rome GraphML archive: https://zenodo.org/records/6510069

Default arms:
  FR (unrestricted); FR->circle; FR->circle->offensive_reinsertion.
Optional arms:
  --my-repo PATH: user's exact Greedy->Offensive pipeline (from benchmark2.py).
  --include-partition: user's exact Partition->Offensive pipeline (requires --my-repo).
  --mass-aware: mass-aware FR, and its projected circular variant.

The embedded offensive reinsertion uses the same vertex ranking, gap selection,
lexicographically sorted edge-crossing-vector objective, and first improvement
scheme as the user's benchmark2.py. It is included for an ablation with *the
same* FR-circular initialization and does not replace the full user pipeline.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import networkx as nx

BINS = [(10, 19), (20, 39), (40, 59), (60, 79), (80, 100)]
ROME_URL = 'https://zenodo.org/records/6510069/files/rome-graphml.tgz?download=1'


def bin_name(n):
    return next((f'{lo}-{hi}' for lo, hi in BINS if lo <= n <= hi), None)


def normalize_graph(raw):
    """Use same simple, undirected graph for all methods (keep all vertices)."""
    g = nx.Graph(raw)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def graphml_from_bytes(data):
    return normalize_graph(nx.read_graphml(io.BytesIO(data)))


def graph_records(path):
    """Stream the whole archive; reservoir sampler handles the fixed population.

    The extraction never writes archive members to disk, so paths inside the
    archive cannot escape the output directory.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob('*.graphml'))
        if not files:
            raise ValueError(f'No .graphml files found under {path}')
        for p in files:
            yield str(p.relative_to(path)), graphml_from_bytes(p.read_bytes())
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, 'r:*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith('.graphml'):
                    continue
                opened = archive.extractfile(member)
                if opened is None:
                    continue
                yield member.name, graphml_from_bytes(opened.read())
    elif path.suffix.lower() == '.graphml':
        yield path.name, graphml_from_bytes(path.read_bytes())
    else:
        raise ValueError('Expected a directory of .graphml files, one .graphml file, or a .tgz archive')


def stratified_subset(path, per_bin=2, seed=2026):
    """Uniform (without replacement) sample within each |V| bin.

    Reservoir sampling over graph records ensures reproducibility; it does not
    preferentially sample the earliest/lexicographically named Rome graphs.
    """
    rng = random.Random(seed)
    seen = defaultdict(int)
    samples = defaultdict(list)
    for name, graph in graph_records(path):
        label = bin_name(graph.number_of_nodes())
        if label is None:
            continue
        seen[label] += 1
        count = seen[label]
        bucket = samples[label]
        if len(bucket) < per_bin:
            bucket.append((name, graph))
        else:
            index = rng.randrange(count)
            if index < per_bin:
                bucket[index] = (name, graph)
    selected = []
    for lo, hi in BINS:
        label = f'{lo}-{hi}'
        for name, graph in sorted(samples[label], key=lambda x: x[0]):
            selected.append((name, label, graph))
    if not selected:
        raise ValueError('No graphs in the requested Rome size strata.')
    return selected, dict(seen)


def relabel_for_trial(graph, seed):
    """Randomized labels for unbiased tie-breaking, consistent across methods."""
    rng = random.Random(seed)
    old = sorted(graph.nodes(), key=str)
    labels = [f'v{i:04d}' for i in range(len(old))]
    rng.shuffle(labels)
    return nx.relabel_nodes(graph, dict(zip(old, labels)), copy=True)


def order_as_circle(order):
    n = len(order)
    return {v: (math.cos(2 * math.pi * i / max(1, n)),
                math.sin(2 * math.pi * i / max(1, n)))
            for i, v in enumerate(order)}


def angle_order(graph, positions):
    nodes = list(graph.nodes())
    if not nodes:
        return []
    cx = sum(positions[v][0] for v in nodes) / len(nodes)
    cy = sum(positions[v][1] for v in nodes) / len(nodes)
    return sorted(nodes, key=lambda v: (
        math.atan2(positions[v][1] - cy, positions[v][0] - cx), str(v)))


def circular_edge_counts(order, edges):
    """Exact combinatorial crossing counts for a circular chord diagram."""
    index = {v: i for i, v in enumerate(order)}
    counts = {frozenset((u, v)): 0 for u, v in edges}
    for i, (u, v) in enumerate(edges):
        a, b = sorted((index[u], index[v]))
        for x, y in edges[i+1:]:
            if len({u, v, x, y}) < 4:
                continue
            c, d = index[x], index[y]
            if (a < c < b) != (a < d < b):
                counts[frozenset((u, v))] += 1
                counts[frozenset((x, y))] += 1
    return counts


def crossing_vector(order, edges):
    return tuple(sorted(circular_edge_counts(order, edges).values(), reverse=True))


def chord_crosses(indices, first, second):
    u, v = first
    x, y = second
    if len({u, v, x, y}) < 4:
        return False
    a, b = sorted((indices[u], indices[v]))
    return (a < indices[x] < b) != (a < indices[y] < b)


def reinsertion_vector(current_order, candidate_order, edges, counts, v,
                       old_index):
    """Score a reinserted vertex in O(n + deg(v)*m + m log m) rather than O(m^2).

    Only crossings involving an incident edge of v can change because the
    circular order of every other vertex is preserved. This optimization
    preserves the user's exact sorted-edge-count objective.
    """
    new_index = {x: i for i, x in enumerate(candidate_order)}
    new_counts = dict(counts)
    incident_edges = [e for e in edges if v in e]
    for e in incident_edges:
        ek = frozenset(e)
        for f in edges:
            if len(set(e) | set(f)) < 4:
                continue
            difference = int(chord_crosses(new_index, e, f)) - int(chord_crosses(old_index, e, f))
            if difference:
                new_counts[ek] += difference
                new_counts[frozenset(f)] += difference
    return tuple(sorted(new_counts.values(), reverse=True))


def offensive_improve_from_fr(vertices, edges, initial_order, max_rounds=25):
    """Exact decision rules of user's 'offensive_improve' strategy.

    The objective is max crossings/edge with lexicographic tie-breaks, not a
    direct minimizer of max vertex mass. Candidate scores use a fast exact
    incremental computation, which does not change the accepted moves.
    """
    incident = {v: [] for v in vertices}
    for u, v in edges:
        incident[u].append(frozenset((u, v)))
        incident[v].append(frozenset((u, v)))
    current_order = list(initial_order)
    for _ in range(max_rounds):
        c = circular_edge_counts(current_order, edges)
        current_vector = tuple(sorted(c.values(), reverse=True))
        scores = {v: sum(c[e] for e in incident[v]) / len(incident[v])
                  if incident[v] else 0.0 for v in vertices}
        ranked = sorted(vertices, key=lambda v: scores[v], reverse=True)
        improved = False
        old_index = {x: i for i, x in enumerate(current_order)}
        for v in ranked:
            reduced = [u for u in current_order if u != v]
            if not reduced:
                continue
            gaps = sorted(range(len(reduced)), key=lambda i:
                          (scores[reduced[i-1]] + scores[reduced[i]]) / 2)
            best_order = current_order
            best_vector = current_vector
            for p in gaps:
                candidate = reduced[:p] + [v] + reduced[p:]
                candidate_vector = reinsertion_vector(current_order, candidate,
                                                       edges, c, v, old_index)
                if candidate_vector < best_vector:
                    best_order, best_vector = candidate, candidate_vector
            if best_vector < current_vector:
                current_order = best_order
                improved = True
                break
        if not improved:
            break
    return current_order


def orient(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def proper_crossing(a, b, c, d, tol=1e-12):
    """Strict interior intersection; shared endpoints are excluded by caller."""
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    # Zero orientations are degenerate/ambiguous, not proper crossings.
    if min(abs(o1), abs(o2), abs(o3), abs(o4)) < tol:
        return False
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def evaluate(graph, positions):
    """One independent evaluator for all algorithms, both drawing models."""
    nodes = list(graph.nodes())
    if set(positions) != set(nodes):
        raise ValueError('Layout omitted or added vertices')
    edges = list(graph.edges())
    mass = dict.fromkeys(nodes, 0)
    per_edge = dict.fromkeys(range(len(edges)), 0)
    crossings = 0
    for i, (u, v) in enumerate(edges):
        for j in range(i+1, len(edges)):
            x, y = edges[j]
            if len({u, v, x, y}) < 4:
                continue
            if proper_crossing(positions[u], positions[v], positions[x], positions[y]):
                crossings += 1
                per_edge[i] += 1
                per_edge[j] += 1
                for w in (u, v, x, y):
                    mass[w] += 1
    assert sum(mass.values()) == 4 * crossings
    assert sum(per_edge.values()) == 2 * crossings
    return dict(max_mass=max(mass.values(), default=0),
                crossings=crossings,
                max_edge_crossings=max(per_edge.values(), default=0),
                mean_mass=sum(mass.values()) / max(1, len(nodes)))


def load_original_pipeline(repo_path):
    """Requires the actual repo's benchmark2.py and four imported modules."""
    sys.path.insert(0, str(Path(repo_path).resolve()))
    try:
        from benchmark2 import greedy_generation, partition_then_order, offensive_improve
    except Exception as e:
        raise RuntimeError(
            'Could not import your original benchmark2.py dependencies. '
            'Place this script alongside benchmark2.py, '
            'greedy_circular_generation.py, improving_an_order.py, '
            'annealing_order.py and partition_hill_climb.py, or pass '
            '--my-repo /path/to/your/Crawford-folder. '
            f'Original error: {e}'
        ) from e
    return greedy_generation, partition_then_order, offensive_improve


def append_result(rows, graph_name, group, trial, seed, graph, method, model,
                  positions, elapsed_ms):
    summary = evaluate(graph, positions)
    row = dict(graph=graph_name, size_bin=group, trial=trial, seed=seed,
               n=graph.number_of_nodes(), m=graph.number_of_edges(),
               method=method, drawing_model=model,
               time_ms=round(elapsed_ms, 3), **summary)
    rows.append(row)
    print(f'{graph_name[:25]:25} trial={trial} {method:25} '
          f'M={row["max_mass"]:3} C={row["crossings"]:4} '
          f'k={row["max_edge_crossings"]:3} t={elapsed_ms:7.1f}ms', flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    result = []
    methods = sorted(set(row['method'] for row in rows))
    bins = ['ALL'] + [f'{lo}-{hi}' for lo, hi in BINS]
    for method in methods:
        for label in bins:
            subset = [r for r in rows if r['method'] == method
                      and (label == 'ALL' or r['size_bin'] == label)]
            if not subset:
                continue
            result.append(dict(method=method, size_bin=label, runs=len(subset),
               graphs=len(set(r['graph'] for r in subset)),
               median_max_mass=statistics.median(r['max_mass'] for r in subset),
               mean_max_mass=round(statistics.mean(r['max_mass'] for r in subset), 3),
               median_crossings=statistics.median(r['crossings'] for r in subset),
               median_max_edge_crossings=statistics.median(r['max_edge_crossings'] for r in subset),
               median_time_ms=round(statistics.median(r['time_ms'] for r in subset), 3)))
    return result


def paired_results(rows):
    """All deltas are method minus FR-circle, matched graph and trial."""
    index = {(r['graph'], r['trial'], r['method']): r for r in rows}
    result = []
    for r in rows:
        if r['method'] == 'FR-circle':
            continue
        base = index.get((r['graph'], r['trial'], 'FR-circle'))
        if base is None:
            continue
        result.append(dict(graph=r['graph'], trial=r['trial'],
                           size_bin=r['size_bin'], method=r['method'],
                           same_model=r['drawing_model'] == 'circular',
                           delta_max_mass=r['max_mass'] - base['max_mass'],
                           delta_crossings=r['crossings'] - base['crossings'],
                           delta_max_edge_crossings=(r['max_edge_crossings']
                                                     - base['max_edge_crossings'])))
    return result


def plot_summary(summary, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = [f'{lo}-{hi}' for lo, hi in BINS]
    methods = sorted({r['method'] for r in summary})
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method in methods:
        present = [(b, next((r['median_max_mass'] for r in summary
                             if r['method'] == method and r['size_bin'] == b), None))
                   for b in bins]
        indices = [i for i, (_, value) in enumerate(present) if value is not None]
        values = [value for _, value in present if value is not None]
        if indices:
            ax.plot(indices, values, marker='o', label=method)
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel('Number of vertices |V| (size stratum)')
    ax.set_ylabel('Median maximum vertex crossing mass')
    ax.set_title('Rome graphs — crossing mass by graph size')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'mass_by_size.png', dpi=170)
    plt.close(fig)


def save_example(graph, layouts, output_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = [x for x in ('FR', 'FR-circle', 'FR-circle+Offensive',
                            'Greedy+Offensive', 'Partition+Offensive',
                            'Mass-FR') if x in layouts]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(4.3*len(methods), 4.7))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        pos = layouts[method]
        stats = evaluate(graph, pos)
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False,
                         node_size=30, width=.7, edge_color='#64748b',
                         node_color='#2d9cdb')
        ax.set_title(f'{method}\nM={stats["max_mass"]}, C={stats["crossings"]}',
                     fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def demo_records():
    # For harness verification ONLY. Not Rome experimental data.
    graphs = [nx.cycle_graph(12), nx.petersen_graph(),
              nx.watts_strogatz_graph(24, 4, 0.2, seed=1)]
    return [(f'DEMO_{i}.graphml', bin_name(len(g)), g)
            for i, g in enumerate(graphs, start=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rome', type=Path, help='Rome GraphML .tgz, .graphml, or folder')
    parser.add_argument('--demo', action='store_true', help='Test on synthetic graphs; NOT Rome')
    parser.add_argument('--per-bin', type=int, default=2,
                        help='Graphs sampled per 10-19, 20-39, 40-59, 60-79, 80-100 bin')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--iterations', type=int, default=250, help='FR iterations')
    parser.add_argument('--offensive-rounds', type=int, default=25)
    parser.add_argument('--my-repo', type=Path,
                        help='Directory with your original benchmark2.py and its dependencies')
    parser.add_argument('--include-partition', action='store_true',
                        help='Also test original Partition->Offensive (slow)')
    parser.add_argument('--partition-iterations', type=int, default=2500)
    parser.add_argument('--partition-population', type=int, default=100)
    parser.add_argument('--mass-aware', action='store_true', help='Include mass-force-directed variant (slow)')
    parser.add_argument('--mass-rounds', type=int, default=60)
    parser.add_argument('--output', type=Path, default=Path('rome_mass_results'))
    args = parser.parse_args()
    if not args.demo and not args.rome:
        parser.error('Provide --rome /path/to/rome-graphml.tgz or --demo')
    if args.include_partition and not args.my_repo:
        parser.error('--include-partition requires --my-repo')
    if args.per_bin <= 0 or args.trials <= 0:
        parser.error('--per-bin and --trials must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    repo_functions = load_original_pipeline(args.my_repo) if args.my_repo else None
    # These modules must be beside this file (included in supplied kit).
    from force_directed import force_directed_layout
    if args.mass_aware:
        from mass_force_directed import mass_force_directed_layout
    if args.demo:
        selected = demo_records()
        seen = {}
        print('DEMO MODE: these are not Rome graph results.')
    else:
        print(f'Selecting {args.per_bin} Rome graphs per size stratum (seed={args.seed})...', flush=True)
        selected, seen = stratified_subset(args.rome, per_bin=args.per_bin, seed=args.seed)
        print(f'Population by size: {seen} | selected: {len(selected)}', flush=True)
    manifest = [dict(graph=name, size_bin=label, n=g.number_of_nodes(),
                     m=g.number_of_edges()) for name, label, g in selected]
    write_csv(args.output / 'graph_selection.csv', manifest)
    rows = []
    for graph_idx, (name, label, original) in enumerate(selected):
        for trial in range(args.trials):
            seed = args.seed + graph_idx * 100003 + trial
            graph = relabel_for_trial(original, seed)
            vertices = list(graph.nodes())
            edges = list(graph.edges())
            plot_data = {}
            t0 = perf_counter()
            fr_pos = force_directed_layout(graph, iterations=args.iterations, seed=seed)
            fr_ms = (perf_counter() - t0) * 1000
            append_result(rows, name, label, trial, seed, graph, 'FR',
                          'unrestricted', fr_pos, fr_ms)
            plot_data['FR'] = fr_pos
            t0 = perf_counter()
            fr_order = angle_order(graph, fr_pos)
            fr_circle = order_as_circle(fr_order)
            fr_circle_ms = (perf_counter() - t0)*1000 + fr_ms
            append_result(rows, name, label, trial, seed, graph, 'FR-circle',
                          'circular', fr_circle, fr_circle_ms)
            plot_data['FR-circle'] = fr_circle
            # Self-check: exact circular interleaving equals geometric crossings.
            assert sum(circular_edge_counts(fr_order, edges).values()) // 2 == evaluate(graph, fr_circle)['crossings'], (
                'Geometric/chord crossing count mismatch')
            t0 = perf_counter()
            adapted_order = offensive_improve_from_fr(
                vertices, edges, fr_order, max_rounds=args.offensive_rounds)
            adapted_circle = order_as_circle(adapted_order)
            adapted_ms = fr_circle_ms + (perf_counter() - t0) * 1000
            append_result(rows, name, label, trial, seed, graph,
                          'FR-circle+Offensive', 'circular', adapted_circle, adapted_ms)
            plot_data['FR-circle+Offensive'] = adapted_circle
            if repo_functions:
                greedy_generation, partition_then_order, offensive_improve = repo_functions
                random.seed(seed)
                t0 = perf_counter()
                greedy_order = greedy_generation(vertices, edges)
                if isinstance(greedy_order, tuple):
                    greedy_order = greedy_order[0]
                my_order = offensive_improve(
                    vertices, edges, greedy_order, max_rounds=args.offensive_rounds)
                if isinstance(my_order, tuple):
                    my_order = my_order[0]
                if set(my_order) != set(vertices) or len(my_order) != len(vertices):
                    raise ValueError('Original Greedy->Offensive produced invalid vertex order')
                my_ms = (perf_counter() - t0) * 1000
                pos = order_as_circle(my_order)
                append_result(rows, name, label, trial, seed, graph,
                              'Greedy+Offensive', 'circular', pos, my_ms)
                plot_data['Greedy+Offensive'] = pos
                if args.include_partition:
                    rng = random.Random(seed)
                    start = list(vertices)
                    rng.shuffle(start)
                    random.seed(seed)
                    t0 = perf_counter()
                    partition_order = partition_then_order(vertices, edges, start,
                        rng_seed=seed, population_size=args.partition_population,
                        iterations=args.partition_iterations)
                    if isinstance(partition_order, tuple):
                        partition_order = partition_order[0]
                    partition_order = offensive_improve(
                        vertices, edges, partition_order, max_rounds=args.offensive_rounds)
                    if isinstance(partition_order, tuple):
                        partition_order = partition_order[0]
                    if set(partition_order) != set(vertices) or len(partition_order) != len(vertices):
                        raise ValueError('Original Partition->Offensive produced invalid vertex order')
                    par_ms = (perf_counter() - t0) * 1000
                    pos = order_as_circle(partition_order)
                    append_result(rows, name, label, trial, seed, graph,
                                  'Partition+Offensive', 'circular', pos, par_ms)
                    plot_data['Partition+Offensive'] = pos
            if args.mass_aware:
                t0 = perf_counter()
                mf_pos = mass_force_directed_layout(
                    graph, iterations=args.iterations,
                    improvement_rounds=args.mass_rounds, seed=seed)
                mf_ms = (perf_counter() - t0) * 1000
                append_result(rows, name, label, trial, seed, graph,
                              'Mass-FR', 'unrestricted', mf_pos, mf_ms)
                plot_data['Mass-FR'] = mf_pos
                t0 = perf_counter()
                mass_order = angle_order(graph, mf_pos)
                mass_circle = order_as_circle(mass_order)
                mf_circle_ms = mf_ms + (perf_counter() - t0) * 1000
                append_result(rows, name, label, trial, seed, graph,
                              'Mass-FR-circle', 'circular', mass_circle, mf_circle_ms)
            if graph_idx == 0 and trial == 0:
                save_example(graph, plot_data, args.output / 'example_drawings.png')
        # Save after each graph, so intermediate results survive interruption.
        write_csv(args.output / 'runs.csv', rows)
    summary = summarise(rows)
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'paired_vs_FR_circle.csv', paired_results(rows))
    plot_summary(summary, args.output)
    print('\nSummary (overall):')
    for r in summary:
        if r['size_bin'] == 'ALL':
            print(f'  {r["method"]:25} median M={r["median_max_mass"]:>5} '
                  f'median C={r["median_crossings"]:>5} '
                  f'median t={r["median_time_ms"]:>9} ms')
    print(f'\nSaved outputs to {args.output.resolve()}')


if __name__ == '__main__':
    main()
