import random
import networkx as nx


def generate_weighted_barabasi_albert(
    n,
    m,
    weight_range=(1, 10),
    seed=None,
):
    """
    Generate a weighted Barabási-Albert graph.

    Parameters
    ----------
    n : int
        Total number of vertices.

    m : int
        Number of edges added by each new vertex.

    weight_range : tuple[int, int]
        Inclusive range for random integer edge weights.

    seed : int or None
        Random seed.

    Returns
    -------
    G : networkx.Graph
        Weighted Barabási-Albert graph.

    Notes
    -----
    Each edge receives an attribute:
        G[u][v]["weight"]

    The generated graph is connected provided:
        1 <= m < n
    """

    rng = random.Random(seed)

    # ------------------------------------------------------------
    # Generate Barabási-Albert graph
    # ------------------------------------------------------------

    G = nx.barabasi_albert_graph(
        n=n,
        m=m,
        seed=seed
    )

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
    # Different values of m
    #
    # Small m:
    #     sparser graph with stronger hub structure
    #
    # Larger m:
    #     denser graph with weaker hub dominance
    # ------------------------------------------------------------

    m_values = [2, 3, 5]

    instance_id = 0

    for n in n_values:

        for m in m_values:

            # m must satisfy 1 <= m < n
            if m >= n:
                continue

            G = generate_weighted_barabasi_albert(
                n=n,
                m=m,
                weight_range=(1, 10),
                seed=100 + instance_id,
            )

            instances.append(G)

            print("------------------------------------------------")
            print(f"Instance {instance_id}")
            print("n =", n)
            print("m =", m)
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

            # ----------------------------------------------------
            # Example:
            # print highest-degree vertices (hubs)
            # ----------------------------------------------------

            degree_sequence = sorted(
                G.degree(),
                key=lambda x: x[1],
                reverse=True
            )

            print("Top 5 hubs:")

            for node, degree in degree_sequence[:5]:

                print(
                    f"vertex {node}: "
                    f"degree = {degree}"
                )

            instance_id += 1
