import random
import networkx as nx


def generate_random_tree_graph(n, seed=None):
    """
    Random weighted tree.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)

    try:
        G = nx.random_tree(n, seed=seed)
    except AttributeError:
        prufer_sequence = [rng.randrange(n) for _ in range(n - 2)]
        G = nx.from_prufer_sequence(prufer_sequence)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 10)

    return G


def generate_balanced_tree_graph(n, seed=None):
    """
    Balanced weighted binary tree.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)

    branching_factor = 2
    height = 0

    while (branching_factor ** (height + 1) - 1) // (branching_factor - 1) < n:
        height += 1

    G = nx.balanced_tree(branching_factor, height)

    nodes_to_keep = list(G.nodes())[:n]
    G = G.subgraph(nodes_to_keep).copy()

    G = nx.convert_node_labels_to_integers(G)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 10)

    return G

if __name__ == "__main__":

    for n in [50, 100, 200]:
        random_tree = generate_random_tree_graph(n, seed=1)
        balanced_tree = generate_balanced_tree_graph(n)
        print("n =", n)
        print("Random Tree:", random_tree.number_of_nodes(), "nodes,", random_tree.number_of_edges(), "edges")
        print("Balanced Tree:", balanced_tree.number_of_nodes(), "nodes,", balanced_tree.number_of_edges(), "edges")
        print()