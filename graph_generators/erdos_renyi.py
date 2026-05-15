import random
import networkx as nx


def generate_weighted_erdos_renyi(
    n,
    p,
    weight_range=(1, 10),
    seed=None,
    ensure_connected=True,
):
    """
    Generate a connected weighted Erdős-Rényi graph.

    Parameters
    ----------
    n : int
        Number of vertices.

    p : float
        Probability that each edge exists independently.

    weight_range : tuple[int, int]
        Inclusive range for random integer edge weights.

    seed : int or None
        Random seed.

    ensure_connected : bool
        If True, regenerate until the graph is connected.

    Returns
    -------
    G : networkx.Graph
        Weighted Erdős-Rényi graph.

    Notes
    -----
    Each edge receives an attribute:
        G[u][v]["weight"]
    """

    rng = random.Random(seed)

    attempt = 0

    while True:

        current_seed = None if seed is None else seed + attempt

        # Generate Erdős-Rényi graph
        G = nx.erdos_renyi_graph(
            n=n,
            p=p,
            seed=current_seed
        )

        # Stop if graph is connected
        if not ensure_connected or nx.is_connected(G):
            break

        attempt += 1

    # ------------------------------------------------------------
    # Assign random positive integer weights to edges
    # ------------------------------------------------------------

    low, high = weight_range

    for u, v in G.edges():

        G[u][v]["weight"] = rng.randint(low, high)

    return G


# ================================================================
# Example experimental generation
# ================================================================

if __name__ == "__main__":

    instances = []

    # ------------------------------------------------------------
    # Different graph sizes
    # ------------------------------------------------------------

    n_values = [50, 100, 200]

    # ------------------------------------------------------------
    # Different edge probabilities
    #
    # Small p:
    #     sparse graph
    #
    # Large p:
    #     denser graph
    # ------------------------------------------------------------

    p_values = [0.02, 0.05, 0.10]

    instance_id = 0

    for n in n_values:

        for p in p_values:

            G = generate_weighted_erdos_renyi(
                n=n,
                p=p,
                weight_range=(1, 10),
                seed=100 + instance_id,
                ensure_connected=True,
            )

            instances.append(G)

            print("------------------------------------------------")
            print(f"Instance {instance_id}")
            print("n =", n)
            print("p =", p)
            print("Vertices:", G.number_of_nodes())
            print("Edges:", G.number_of_edges())
            print("Connected:", nx.is_connected(G))

            # ----------------------------------------------------
            # Example:
            # compute weighted shortest-path distances
            # ----------------------------------------------------

            distances = dict(
                nx.all_pairs_dijkstra_path_length(
                    G,
                    weight="weight"
                )
            )

            # ----------------------------------------------------
            # Example:
            # print first few weighted edges
            # ----------------------------------------------------

            print("First 10 weighted edges:")

            edge_count = 0

            for u, v, data in G.edges(data=True):

                print(
                    f"({u}, {v}) "
                    f"weight = {data['weight']}"
                )

                edge_count += 1

                if edge_count >= 10:
                    break

            instance_id += 1
