import random
import networkx as nx


def generate_weighted_sbm(
    sizes,
    p_in=0.30,
    p_out=0.02,
    weight_range=(1, 10),
    seed=None,
    ensure_connected=True,
):
    """
    Generate a connected weighted stochastic block model (SBM) graph.

    Parameters
    ----------
    sizes : list[int]
        Sizes of the communities (blocks).
        Example:
            [25, 25, 25, 25]

    p_in : float
        Probability of an edge between two vertices in the same community.

    p_out : float
        Probability of an edge between two vertices in different communities.

    weight_range : tuple[int, int]
        Inclusive range for random integer edge weights.

    seed : int or None
        Random seed.

    ensure_connected : bool
        If True, regenerate the graph until it is connected.

    Returns
    -------
    G : networkx.Graph
        Weighted SBM graph.

    Notes
    -----
    Each node receives a node attribute:
        G.nodes[u]["block"]

    Each edge receives an edge attribute:
        G[u][v]["weight"]
    """

    rng = random.Random(seed)

    # Number of communities
    t = len(sizes)

    # Construct the probability matrix P
    # P[i][j] = edge probability between communities i and j
    probs = [
        [p_in if i == j else p_out for j in range(t)]
        for i in range(t)
    ]

    attempt = 0

    while True:

        current_seed = None if seed is None else seed + attempt

        # Generate SBM graph
        G = nx.stochastic_block_model(
            sizes,
            probs,
            seed=current_seed
        )

        # Stop if graph is connected
        if not ensure_connected or nx.is_connected(G):
            break

        attempt += 1

    # ------------------------------------------------------------
    # Add community/block labels to vertices
    # ------------------------------------------------------------

    node = 0

    for block_id, size in enumerate(sizes):

        for _ in range(size):

            G.nodes[node]["block"] = block_id

            node += 1

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
    # Different community-size configurations
    # ------------------------------------------------------------

    size_configurations = [

        # Balanced communities
        [25, 25, 25, 25],

        # Imbalanced communities
        [10, 20, 30, 40],

        # Larger balanced communities
        [50, 50, 50, 50],
    ]

    # ------------------------------------------------------------
    # Different inter-community probabilities
    #
    # Small p_out:
    #     strong community structure
    #
    # Large p_out:
    #     weak community structure
    # ------------------------------------------------------------

    p_out_values = [0.005, 0.01, 0.03, 0.08]

    # ------------------------------------------------------------
    # Generate graph instances
    # ------------------------------------------------------------

    instance_id = 0

    for sizes in size_configurations:

        for p_out in p_out_values:

            G = generate_weighted_sbm(
                sizes=sizes,
                p_in=0.30,
                p_out=p_out,
                weight_range=(1, 10),
                seed=100 + instance_id,
                ensure_connected=True,
            )

            instances.append(G)

            print("------------------------------------------------")
            print(f"Instance {instance_id}")
            print("Community sizes:", sizes)
            print("p_in =", 0.30)
            print("p_out =", p_out)
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
            # print block labels for first few vertices
            # ----------------------------------------------------

            print("First 10 vertex community labels:")

            for u in range(min(10, G.number_of_nodes())):

                print(
                    f"vertex {u}: "
                    f"block {G.nodes[u]['block']}"
                )

            instance_id += 1
