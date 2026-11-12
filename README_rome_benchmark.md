# Rome graphs: crossing-mass benchmark

This kit compares circular offensiveness-guided vertex orderings against Fruchterman–Reingold (FR) layouts on a **reproducible, stratified subset of the Rome graphs**.

## Setup

Requires Python 3.10+.

```bash
python -m pip install networkx matplotlib
```

Download the Rome GraphML archive (about 4.7 MB) from:

https://zenodo.org/records/6510069

Use the `rome-graphml.tgz` file **without extracting** it. All files in this kit should be in one directory.

## First run: force-directed versus offensiveness refinement

```bash
python rome_mass_benchmark.py \
  --rome rome-graphml.tgz \
  --per-bin 2 \
  --trials 3 \
  --iterations 250 \
  --offensive-rounds 25 \
  --output results_pilot
```

The script samples **2 graphs in each of five size bins** (`10-19`, `20-39`, `40-59`, `60-79`, `80-100` vertices), selected with deterministic reservoir sampling. It repeats each graph 3 times with different, controlled vertex-label shuffles and force-layout seeds. Default arms:

- **FR**: ordinary unrestricted Fruchterman–Reingold.
- **FR-circle**: angular vertex ordering from FR, projected to evenly spaced circular positions.
- **FR-circle+Offensive**: *your offensiveness reinsertion decision rules* applied to **the very same FR-circle starting order**. This is an ablation of the local-improvement stage, not a complete Greedy/Partition pipeline.

The standalone offensiveness method is transcribed from your earlier `benchmark2.py`: it prioritizes high mean incident-edge crossing counts, ranks gaps by neighboring offensiveness, and accepts a move when it improves the **descending edge-crossing-count vector** lexicographically. It evaluates candidate reinsertion scores incrementally, an exact acceleration verified against brute-force scoring. It is **not** intrinsically minimizing max vertex crossing mass.

## Running your full original algorithm

Your `benchmark2.py` has these imports:

```
greedy_circular_generation.py
improving_an_order.py
annealing_order.py
partition_hill_climb.py
```

They were not available in the shared material, so the full original Greedy/Partition pipelines **cannot be executed in this kit without those modules**.

On your own machine, point to the folder containing those files:

```bash
python rome_mass_benchmark.py \
  --rome rome-graphml.tgz \
  --my-repo /path/to/graph-drawing-2026-surf/Crawford \
  --per-bin 2 --trials 3 \
  --offensive-rounds 25 \
  --output results_my_algo
```

This adds **Greedy+Offensive**, using the original imported `greedy_generation` and `offensive_improve`, alongside the force-directed baselines.

To also compare the independent **Partition+Offensive** pipeline:

```bash
python rome_mass_benchmark.py \
  --rome rome-graphml.tgz \
  --my-repo /path/to/Crawford \
  --include-partition \
  --partition-iterations 2500 \
  --partition-population 100 \
  --output results_my_algo_with_partition
```

Note: the original full pipeline can be much slower on large Rome graphs, particularly partitioning and exhaustive reinsertion. Both construction and improvement time count toward the method's `time_ms`.

## Add the mass-aware force-directed variant

```bash
python rome_mass_benchmark.py \
  --rome rome-graphml.tgz \
  --mass-aware --mass-rounds 60 \
  --output results_mass_fr
```

Adds **Mass-FR** (free coordinates) and **Mass-FR-circle** (mass-aware free drawing reordered onto a circle). The circular projection may undo some crossing improvements found in free space.

## Evaluation metrics

Each drawing receives the **same geometric crossing evaluator**, testing pairs of nonadjacent straight edges for proper interior intersections.

- `max_mass` = \(\max_v\sum_{e\ni v}c_D(e)\), the **primary** measure.
- `crossings` = total number of unordered crossing edge pairs.
- `max_edge_crossings` = \(\max_e c_D(e)\), the user's original k-planarity metric.
- `time_ms` = complete elapsed algorithm runtime (including FR or greedy initialization); evaluation time excluded.
- `mean_mass` = average vertex crossing mass.

Every proper crossing contributes one unit to **four vertex masses** and **two edge crossing counts**; the evaluator asserts both identities.

These metrics are compatible across drawing families, but *unrestricted FR vs circular algorithms is not a controlled comparison of optimization heuristics under the same feasible set*. For controlled circular drawing comparisons, use `FR-circle` as the reference, not free `FR`.

## Outputs

- `graph_selection.csv`: exact Rome file names, |V|, |E| and size bin.
- `runs.csv`: individual, paired graph+trial results with method, seed, time, mass, crossings, and k.
- `summary.csv`: median and mean maximum mass, median crossing count and elapsed time, by method and size bin.
- `paired_vs_FR_circle.csv`: per-graph-and-trial *method minus FR-circle* differences. **Negative is better** for crossing metrics; `same_model` flags the unrestricted methods.
- `mass_by_size.png`: median max mass versus size stratum.
- `example_drawings.png`: same graph drawn with compared algorithms.

A stronger paper-quality experiment should increase `--per-bin` and `--trials` (for example `--per-bin 10 --trials 10`), report per-graph paired results with uncertainty intervals, state the hardware/runtime conditions, and record all algorithm hyperparameters. Results from `--demo` or synthetic test archives are **not** Rome benchmark findings.

## Quick smoke test (synthetic graphs only)

```bash
python rome_mass_benchmark.py --demo --trials 1 \
  --iterations 50 --offensive-rounds 3 \
  --output smoke_test
```

## Data citation

Matthias Hummel, *Planar Graph Classifier – GD Rome input*, Zenodo, 2022. DOI: [10.5281/zenodo.6510069](https://doi.org/10.5281/zenodo.6510069). The graphs are sourced from the Rome Graphs / GDToolkit benchmark collection.
