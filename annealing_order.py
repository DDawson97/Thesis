import math
import random
import time

# Application of "Simulated Annealing Techniques" by Albert Y. Zomaya and Rick Kazman

try:
    from greedy_circular_generation import edge_crossing_numbers
except ImportError:
    from improving_an_order import edge_crossing_numbers


def crossing_stats(order, edges):
    """
    Returns:
        local_k: maximum crossings on any single edge
        total_crossings: total number of edge crossings
    """
    K = edge_crossing_numbers(order, edges)

    if not K:
        return 0, 0

    local_k = max(K.values())

    # Each crossing is counted once for each of its two edges
    total_crossings = sum(K.values()) // 2

    return local_k, total_crossings


def energy(order, edges):
    """
    Energy function for simulated annealing.

    We primarily minimize local crossing number.
    Total crossings are used as a tie-breaker.

    Energy = BIG * local_k + total_crossings

    BIG is chosen so that reducing local_k by 1 is always better
    than any possible improvement in total crossings.
    """
    m = len(edges)
    max_possible_crossings = m * (m - 1) // 2
    BIG = max_possible_crossings + 1

    local_k, total_crossings = crossing_stats(order, edges)

    return BIG * local_k + total_crossings, local_k, total_crossings


def rotate_to_anchor(order, anchor):
    """
    Circular orders are equivalent up to rotation.

    This keeps the same anchor vertex at index 0 so the algorithm
    does not waste time exploring equivalent rotations.
    """
    i = order.index(anchor)
    return order[i:] + order[:i]


def random_neighbor(order, rng):
    """
    Generate a neighboring circular order.

    Uses one of three perturbations:
        1. swap two vertices
        2. remove one vertex and reinsert it elsewhere
        3. reverse a block
    """
    n = len(order)

    if n <= 2:
        return order[:]

    candidate = order[:]
    move = rng.choice(["swap", "insert", "reverse"])

    if move == "swap":
        i, j = rng.sample(range(n), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]

    elif move == "insert":
        i, j = rng.sample(range(n), 2)
        v = candidate.pop(i)
        candidate.insert(j, v)

    elif move == "reverse":
        i, j = sorted(rng.sample(range(n), 2))
        candidate[i:j + 1] = reversed(candidate[i:j + 1])

    return candidate


def estimate_initial_temperature(
    order,
    edges,
    rng,
    samples=100,
    target_acceptance=0.80,
):
    """
    Estimate a starting temperature T0.

    We sample uphill moves and choose T0 so that an average uphill
    move is accepted with probability roughly target_acceptance.
    """
    current_energy, _, _ = energy(order, edges)

    uphill_deltas = []

    for _ in range(samples):
        candidate = random_neighbor(order, rng)
        candidate_energy, _, _ = energy(candidate, edges)

        delta = candidate_energy - current_energy

        if delta > 0:
            uphill_deltas.append(delta)

    if not uphill_deltas:
        return 1.0

    avg_delta = sum(uphill_deltas) / len(uphill_deltas)

    return -avg_delta / math.log(target_acceptance)


def simulated_annealing_order(
    vertices,
    edges,
    initial_order=None,
    seed=None,
    max_steps=20_000,
    steps_per_temp=None,
    t0=None,
    t_min=1e-3,
    alpha=0.995,
    target_initial_acceptance=0.80,
    time_limit_seconds=None,
    return_history=False,
):
    """
    Simulated annealing for improving a circular vertex order.

    Parameters:
        vertices:
            List of graph vertices.

        edges:
            List of edges.

        initial_order:
            Starting circular order. Good choices:
                - greedy_generation(V, E)
                - random order

        seed:
            Random seed for reproducible experiments.

        max_steps:
            Total number of neighbor moves attempted.

        steps_per_temp:
            Number of attempted moves before cooling temperature.

        t0:
            Initial temperature. If None, it is estimated automatically.

        t_min:
            Stop once T falls below this.

        alpha:
            Geometric cooling rate.
            T <- alpha * T

            Common values:
                0.99     faster cooling
                0.995    medium
                0.999    slower, better search, more time

        time_limit_seconds:
            Optional wall-clock cutoff.

        return_history:
            If True, also return progress data.

    Returns:
        best_order

    Or, if return_history=True:
        best_order, info
    """
    rng = random.Random(seed)

    vertices = list(vertices)
    edges = list(edges)

    if initial_order is None:
        current = vertices[:]
        rng.shuffle(current)
    else:
        current = list(initial_order)

    if set(current) != set(vertices):
        raise ValueError("initial_order must contain exactly the vertices in V.")

    if len(current) != len(vertices):
        raise ValueError("initial_order has duplicate or missing vertices.")

    if len(vertices) <= 2:
        if return_history:
            return current, {"history": []}
        return current

    anchor = current[0]
    current = rotate_to_anchor(current, anchor)

    current_energy, current_local_k, current_total = energy(current, edges)

    best = current[:]
    best_energy = current_energy
    best_local_k = current_local_k
    best_total = current_total

    if steps_per_temp is None:
        steps_per_temp = 10 * len(vertices)

    if t0 is None:
        T = estimate_initial_temperature(
            current,
            edges,
            rng,
            samples=min(200, max_steps),
            target_acceptance=target_initial_acceptance,
        )
    else:
        T = t0

    history = []
    start_time = time.perf_counter()

    step = 0

    while step < max_steps and T > t_min:

        accepted = 0
        improved = 0

        for _ in range(steps_per_temp):

            if step >= max_steps:
                break

            if time_limit_seconds is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_limit_seconds:
                    break

            candidate = random_neighbor(current, rng)
            candidate = rotate_to_anchor(candidate, anchor)

            candidate_energy, candidate_local_k, candidate_total = energy(
                candidate,
                edges,
            )

            delta = candidate_energy - current_energy

            accept = False

            if delta <= 0:
                accept = True
            else:
                probability = math.exp(-delta / T)
                if rng.random() < probability:
                    accept = True

            if accept:
                current = candidate
                current_energy = candidate_energy
                current_local_k = candidate_local_k
                current_total = candidate_total
                accepted += 1

                if current_energy < best_energy:
                    best = current[:]
                    best_energy = current_energy
                    best_local_k = current_local_k
                    best_total = current_total
                    improved += 1

            step += 1

        history.append({
            "step": step,
            "temperature": T,
            "current_energy": current_energy,
            "current_local_k": current_local_k,
            "current_total_crossings": current_total,
            "best_energy": best_energy,
            "best_local_k": best_local_k,
            "best_total_crossings": best_total,
            "accepted_moves": accepted,
            "improvements": improved,
            "acceptance_rate": accepted / steps_per_temp,
        })

        if time_limit_seconds is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit_seconds:
                break

        T *= alpha

    if return_history:
        info = {
            "best_energy": best_energy,
            "best_local_k": best_local_k,
            "best_total_crossings": best_total,
            "steps": step,
            "final_temperature": T,
            "history": history,
        }
        return best, info

    return best