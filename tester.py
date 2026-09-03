import unittest
from greedy_circular_generation import greedy_generation, local_circular_crossing_number
from improving_an_order import improve_order, edge_crossing_numbers, clean_edges, edge_key



class TestCircularOrderingAlgorithms(unittest.TestCase):


    def assert_is_permutation(self, original_vertices, order):
        self.assertEqual(set(original_vertices), set(order))
        self.assertEqual(len(original_vertices), len(order))

    def test_edge_key_undirected(self):
        self.assertEqual(edge_key(("a", "b")), edge_key(("b", "a")))

    def test_clean_edges_removes_duplicates(self):
        edges = [
            ("a", "b"),
            ("b", "a"),
            ("a", "c"),
            ("c", "a")
        ]

        cleaned = clean_edges(edges)

        self.assertEqual(len(cleaned), 2)
        self.assertIn(edge_key(("a", "b")), cleaned)
        self.assertIn(edge_key(("a", "c")), cleaned)

    def test_crossing_numbers_on_no_crossing_graph(self):
        V = ["a", "b", "c", "d"]
        E = [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "a")
        ]
        Omega = ["a", "b", "c", "d"]

        result = local_circular_crossing_number(Omega, E)
        self.assertEqual(result, 0)

    def test_crossing_numbers_on_one_crossing(self):
        V = ["a", "b", "c", "d"]

        # In circular crossing order a b c d has edges ac and bd crossing
        E = [("a", "c"), ("b", "d")]

        Omega = ["a", "b", "c", "d"]

        K = edge_crossing_numbers(Omega, E)

        self.assertEqual(K[edge_key(("a", "c"))], 1)
        self.assertEqual(K[edge_key(("b", "d"))], 1)

    def test_local_circular_crossing_number(self):
        V = ["a", "b", "c", "d"]

        E = [
            ("a", "c"),
            ("b", "d"),
        ]

        Omega = ["a", "b", "c", "d"]

        result = local_circular_crossing_number(Omega, E)

        self.assertEqual(result, 1)

    def test_greedy_generation_returns_valid_order(self):
        V = ["a", "b", "c", "d", "e"]

        E = [
            ("a", "c"),
            ("a", "d"),
            ("b", "d"),
            ("b", "e"),
            ("c", "e"),
        ]

        Omega = greedy_generation(V, E)

        self.assert_is_permutation(V, Omega)

    def test_greedy_generation_small_graph(self):
        V = ["a", "b", "c"]

        E = [
            ("a", "b"),
            ("b", "c"),
        ]

        Omega = greedy_generation(V, E)

        self.assert_is_permutation(V, Omega)
        self.assertEqual(len(Omega), 3)

    def test_improve_order_returns_valid_order(self):
        V = ["a", "b", "c", "d", "e"]

        E = [
            ("a", "c"),
            ("a", "d"),
            ("b", "d"),
            ("b", "e"),
            ("c", "e"),
        ]

        Omega = ["a", "b", "c", "d", "e"]

        improved = improve_order(V, E, Omega)

        self.assert_is_permutation(V, improved)

    def test_improve_order_does_not_make_max_crossing_worse(self):
        V = ["a", "b", "c", "d", "e", "f"]

        E = [
            ("a", "d"),
            ("b", "e"),
            ("c", "f"),
            ("a", "e"),
            ("b", "f"),
            ("c", "d"),
        ]

        Omega = ["a", "b", "c", "d", "e", "f"]

        before = max(edge_crossing_numbers(Omega, E).values())

        improved = improve_order(V, E, Omega)

        after = max(edge_crossing_numbers(improved, E).values())

        self.assertLessEqual(after, before)

    def test_greedy_then_improve_pipeline(self):
        V = ["a", "b", "c", "d", "e", "f", "g"]

        E = [
            ("a", "d"),
            ("a", "e"),
            ("b", "e"),
            ("b", "f"),
            ("c", "f"),
            ("c", "g"),
            ("d", "g"),
            ("e", "g"),
        ]

        Omega = greedy_generation(V, E)
        improved = improve_order(V, E, Omega)

        self.assert_is_permutation(V, improved)


if __name__ == "__main__":
    unittest.main()