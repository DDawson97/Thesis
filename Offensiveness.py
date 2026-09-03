"""
Algorithm I devised:

function Offensiveness(Graph G = (V, E)):

K = COUNT_EDGE_CROSSINGS
Offensive_List = list[|V|]
for v in V:
    Offensiveness = 0
    for e in N(v):
        Offensiveness += K[e]
    Offensiveness = Offensiveness // |N(v)|
    Offensive_List[v] = Offensiveness
return Offensive_List
"""
from improving_an_order import edge_crossing_numbers


def build_incident_edges(vertices, edges):
    incident = {vertex: [] for vertex in vertices}

    for edge in edges:
        u, v = edge
        incident[u].append(edge)
        incident[v].append(edge)

    return incident


def offensiveness(vertices, edges, omega):
    crossing_counts = edge_crossing_numbers(omega, edges)
    incident = build_incident_edges(vertices, edges)
    offensive_scores = {}

    for vertex in vertices:
        vertex_edges = incident[vertex]

        if not vertex_edges:
            offensive_scores[vertex] = 0
        else:
            offensive_scores[vertex] = (
                sum(crossing_counts[edge] for edge in vertex_edges)
                / len(vertex_edges)
            )

    return offensive_scores

"""
Remove v from the circular order.
Try inserting v into every possible position.
Score each resulting order.
Keep the best improving position.
"""

def best_reinsertion(vertex, vertices, order, edges):
    reduced_order = [v for v in order if v != vertex]

    best_order = list(order)
    best_score = order_score(order, edges)

    scores = offensiveness(vertices, edges, order)

    candidate_positions = candidate_positions_by_offensiveness(
        reduced_order,
        scores,
    )

    for position in candidate_positions:
        candidate = (
            reduced_order[:position]
            + [vertex]
            + reduced_order[position:]
        )

        candidate_score = order_score(candidate, edges)

        if candidate_score < best_score:
            best_order = candidate
            best_score = candidate_score

    return best_order, best_score


def order_score(order, edges):
    """
    k-planarity objective:
    maximum number of crossings on any edge.
    """
    crossing_counts = edge_crossing_numbers(order, edges)
    return max(crossing_counts.values(), default=0)

def candidate_positions_by_offensiveness(order, scores):
    """
    Return insertion positions ordered from lowest surrounding
    offensiveness to highest.
    """
    positions = []

    n = len(order)

    for i in range(n):
        left = order[i]
        right = order[(i + 1) % n]

        gap_score = (scores[left] + scores[right]) / 2
        insertion_position = i + 1

        positions.append((gap_score, insertion_position))

    positions.sort()
    return [position for _, position in positions]


def improve_most_offensive_vertex(vertices, edges, order):
    scores = offensiveness(vertices, edges, order)

    most_offensive_vertex = max(
        scores,
        key=scores.get,
    )

    new_order, new_score = best_reinsertion(
        most_offensive_vertex,
        vertices,
        order,
        edges,
    )

    return new_order, new_score, most_offensive_vertex


def offensiveness_hill_climb(
    vertices,
    edges,
    initial_order,
    max_rounds=100,
):
    current_order = list(initial_order)
    current_score = order_score(current_order, edges)

    for _ in range(max_rounds):
        scores = offensiveness(vertices, edges, current_order)

        # Try vertices from most offensive to least offensive.
        ranked_vertices = sorted(
            vertices,
            key=lambda v: scores[v],
            reverse=True,
        )

        improved = False

        for vertex in ranked_vertices:
            candidate_order, candidate_score = best_reinsertion(
                vertex,
                vertices,
                current_order,
                edges,
            )

            if candidate_score < current_score:
                current_order = candidate_order
                current_score = candidate_score
                improved = True
                break

        if not improved:
            break

    return current_order, current_score