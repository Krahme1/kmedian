import math
import random
import networkx as nx


def generate_path_graph(n, seed=None):
    """
    Path graph:
    0 -- 1 -- 2 -- ... -- n-1
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)
    G = nx.path_graph(n)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 10)

    return G


def generate_grid_graph(n, seed=None):
    """
    2D weighted grid graph with approximately n nodes.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)

    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))

    G = nx.grid_2d_graph(rows, cols)

    nodes = list(G.nodes())
    extra_nodes = nodes[n:]
    G.remove_nodes_from(extra_nodes)

    G = nx.convert_node_labels_to_integers(G)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(1, 10)

    return G


if __name__ == "__main__":
    for n in [50, 100, 200]:
        path = generate_path_graph(n)
        grid = generate_grid_graph(n)

        print("n =", n)
        print("Path:", path.number_of_nodes(), "nodes,", path.number_of_edges(), "edges")
        print("Grid:", grid.number_of_nodes(), "nodes,", grid.number_of_edges(), "edges")
        print()