from collections import defaultdict
from itertools import combinations
from math import inf


def sort_descending(vertices, edges):
    # Sort vertices by descending degree
    degree = defaultdict(int)

    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    return sorted(vertices, key=lambda x: (-degree[x], str(x)))



def edges_inside_order(order, edges):
    # Only keep edges w/ endpoints already inserted into the order
    present = set(order)
    return [(u, v) for u, v in edges if u in present and v in present and u != v]



def crosses(e1, e2, position):
    # Decide whether or not two edges cross
    a, b = e1
    c, d = e2

    # edges which share an endpoint do not cross
    if len({a, b, c, d}) < 4:
        return False
    
    a, b = position[a], position[b]
    c, d = position[c], position[d]

    # place a before b in circular order
    if a > b:
        a, b = b, a
    
    # exactly one endpoint is between a and b
    c_inside = a < c < b
    d_inside = a < d < b

    return c_inside != d_inside



def local_circular_crossing_number(order, edges):
    # This computes the local circular crossing number of the given ordering (local circular crossing = max number of crossings of any single edge)
    position = {v: i for i, v in enumerate(order)}
    current_edges = edges_inside_order(order, edges)

    crossing_counts = defaultdict(int)

    for e1, e2 in combinations(current_edges, 2):
        if crosses(e1, e2, position):
            crossing_counts[e1] += 1
            crossing_counts[e2] += 1

    if not current_edges:
        return 0
    
    return max(crossing_counts[e] for e in current_edges)



def swap(order, i, j):
    # Return a copy of order w/ positions i and j swapped
    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order



def reinsert(order, old_index, new_index):
    # Move old_index to new_index
    new_order = order[:]
    v = new_order.pop(old_index)
    new_order.insert(new_index, v)
    return new_order



def greedy_generation(vertices, edges):
    # This is Algorithm 1 from Marcel Peters' thesis: "Exact and Heuristic Algorithms for Computing the Local Circular Crossing Number."
    # Input: vertices: iterable of vertices
    #        edges: iterable of pairs (u, v)
    vertices = list(vertices)
    edges = [tuple(e) for e in edges]

    Y = sort_descending(vertices, edges)

    if len(Y) <= 3:
        return Y[:]
    
    # Initialize Omega(ordering) with [Y0, Y1, Y2]
    Omega = [Y[0], Y[1], Y[2]]

    #TODO: check my indexing here
    for v in Y[3:]:
        i_min = None
        k_min = inf

        Omega.append(v)

        for i in range(len(Omega) - 1, 0, -1):
            k = local_circular_crossing_number(Omega, edges)

            if k_min > k:
                i_min = i 
                k_min = k

            Omega = swap(Omega, i, i - 1) 

        Omega = reinsert(Omega, 0, i_min)

    return Omega
