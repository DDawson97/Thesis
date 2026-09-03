from itertools import combinations



def edge_key(edge):
    # store undirected edges
    u, v = edge

    if u == v:
        raise ValueError("Use simple graphs, loops are not supported.")
    return frozenset((u, v))



def clean_edges(edges):
    # remove duplicates
    seen = set()
    result = []

    for e in edges:
        k = edge_key(e)
        if k not in seen:
            seen.add(k)
            result.append(k)

    return result



def endpoints(edge):
    # retrieve endpoints from a frozenset edge
    u, v = tuple(edge)
    return u, v



def crosses(e1, e2, position):
    # two edges cross in a circular layout if their endpoints alternate in the order
    a, b = endpoints(e1)
    c, d = endpoints(e2)

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



def edge_crossing_numbers(order, edges):
    # Return dictionary K where K[e] is  num crossings on edge e
    position = {v: i for i, v in enumerate(order)}
    E = clean_edges(edges)

    K = {e: 0 for e in E}

    for e in E:
        u, v = endpoints(e)
        if u not in position or v not in position:
            raise ValueError(f"Edge {(u, v)} has an endpoint missing from Omega.")
        
    for e1, e2 in combinations(E, 2):
        if crosses(e1, e2, position):
            K[e1] += 1
            K[e2] += 1
    
    return K



def edge_with_most_crossings(K):
    if not K:
        raise ValueError("Empty edge set")
    
    return max(K, key=lambda e: K[e])


# This one returns the order and the popped vertex
def remove_vertex(order, v):
    new_order = list(order)
    idx = new_order.index(v)
    new_order.pop(idx)
    return new_order, idx


# This one only returns the order, popped vertex not preserved
def remove_at(order, i):
    new_order = list(order)
    new_order.pop(i)
    return new_order



def insert(order, v, i):
    # Inserts vertex v into the order at index i
    new_order = list(order)
    new_order.insert(i, v)
    return new_order


def swap(order, i, j):
    new_order = list(order)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order










def improve_order(vertices, edges, Omega, max_rounds=1000):
    # ALgorithm 3 from Marcel Peters' thesis
    # Given order Omega, along with the vertices and edges, will attempt to return an improved order Omega
    vertices = list(vertices)
    Omega = list(Omega)
    E = clean_edges(edges)

    if set(vertices) != set(Omega):
        raise ValueError("Omega must contain exactly the vertices in V.")

    if not E:
        return Omega

    K = edge_crossing_numbers(Omega, E)

    for _ in range(max_rounds):

        # Pick edge with most crossings
        uv = edge_with_most_crossings(K)
        u, v = endpoints(uv)

        # Current crossing count of that edge
        k = K[uv]

        # If the worst edge has 0 crossings, we are already done
        if k == 0:
            return Omega

        # Remove u and v from the current order
        base_order = [x for x in Omega if x != u and x != v]

        best_order = Omega[:]
        best_K = K

        # Try every possible position for u
        for i_u in range(len(base_order) + 1):

            order_with_u = base_order[:]
            order_with_u.insert(i_u, u)

            # Try every possible position for v
            for i_v in range(len(order_with_u) + 1):

                candidate = order_with_u[:]
                candidate.insert(i_v, v)

                # Safety check
                if set(candidate) != set(vertices) or len(candidate) != len(vertices):
                    continue

                K_prime = edge_crossing_numbers(candidate, E)

                # Only consider candidates that improve the chosen edge
                if K_prime[uv] < best_K[uv]:

                    p = True

                    for e in E:
                        p1 = best_K[e] < k and K_prime[e] < k
                        p2 = best_K[e] == k and K_prime[e] <= k

                        if not (p1 or p2):
                            p = False
                            break

                    if p:
                        best_K = K_prime
                        best_order = candidate

        # If no improvement happened, stop
        if best_K[uv] >= k:
            return Omega

        # Otherwise accept the best order and keep improving
        Omega = best_order
        K = best_K

    return Omega