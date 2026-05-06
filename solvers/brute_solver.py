import itertools
import random

import torch


def get_optimal_facilities(graph, n, k):
    """
    Brute force search to get the optimal facilities in a graph G
    """
    best_distance = 10000000000
    best_facilities = None
    for facilities in itertools.combinations(list(range(n)), k):
        distance = calculate_distance(graph, facilities, n)

        if distance < best_distance:
            best_distance = distance
            best_facilities = facilities

    return best_facilities

def get_optimal_local_search_swap(graph, facilities, n):
    best_distance = calculate_distance(graph, facilities, n)
    best_pair = None

    for suggested_facility in range(n):
        if suggested_facility in facilities:
            continue

        for i in range(len(facilities)):
            old_facility = facilities.pop(0)
            facilities.append(suggested_facility)
            new_distance = calculate_distance(graph, facilities, n)
            if new_distance < best_distance:
                best_distance = new_distance
                best_pair = {"client": suggested_facility, "facility": old_facility}

            facilities.remove(suggested_facility)
            facilities.append(old_facility)

    return best_pair

def get_good_enough_local_search_swap(graph, facilities, n, max_check, epsilon=0.1):
    original_distance = calculate_distance(graph, facilities, n)
    best_distance = original_distance
    best_pair = None

    for _ in range(max_check):
        potential_facility = random.choice(facilities)
        potential_client = random.randint(0, n - 1)
        # make sure we don't select a facility
        while potential_client in facilities:
            potential_client = random.randint(0, n - 1)

        facilities.remove(potential_facility)
        facilities.append(potential_client)
        new_distance = calculate_distance(graph, facilities, n)

        if new_distance < best_distance:
            best_distance = new_distance
            best_pair = {"client": potential_client, "facility": potential_facility}

        facilities.remove(potential_client)
        facilities.append(potential_facility)

        # early exit condition, we get a change better than epsilon
        percent_change = abs((best_distance - original_distance) / original_distance)
        if percent_change >= epsilon:
            break

    # for right now we just return two facilities
    if best_distance >= original_distance:
        print("fail")
        candidates = random.sample(facilities, k=2)
        best_pair = {"client": candidates[0], "facility": candidates[1]}

    return best_pair


def calculate_distance(graph, facilities, n=None):
    """
    n is depreciated. Remove later.
    """
    # Create a subgraph where each row is an active facility and the columns are the distances to the facilities.
    distance_sub_graph = graph._distances[facilities]

    # get the min distance for each column
    min_values, _ = torch.min(distance_sub_graph, dim=0)

    # sum up the distances to get the total distance
    return float(torch.sum(min_values))


def calculate_distance_with_facility_cost(graph, facilities, facility_costs, n=None):
    """
    Compute the k-facility-location objective:
    sum of client-to-nearest-facility distances plus the opening cost
    of the selected facilities.

    """

    # Distance part: each client is assigned to its nearest selected facility.
    distance_sub_graph = graph._distances[facilities]
    min_values, _ = torch.min(distance_sub_graph, dim=0)
    assignment_cost = float(torch.sum(min_values))

    # Facility opening cost part.
    if isinstance(facility_costs, dict):
        opening_cost = float(sum(facility_costs[int(f)] for f in facilities))
    else:
        opening_cost = 0.0
        for f in facilities:
            if isinstance(facility_costs, torch.Tensor):
                opening_cost += float(facility_costs[int(f)].item())
            else:
                opening_cost += float(facility_costs[int(f)])

    return assignment_cost + opening_cost

def calculate_capacitated_distance(graph, client_activation_values, active_facility_list):
    """
    Compute capacitated k-median cost using the actual client-to-cluster assignment.

    Assumes:
    - `client_activation_values` has shape (n_clients, k_clusters)
    - `active_facility_list[q]` is the facility serving cluster q
    - distance layout is `graph._distances[facility, client]`
    """

    n, k = client_activation_values.shape
    if len(active_facility_list) != k:
        raise ValueError(
            f"active_facility_list length ({len(active_facility_list)}) must match k ({k})."
        )

    total_distance = 0.0
    for i in range(n):
        assigned_clusters = torch.nonzero(client_activation_values[i], as_tuple=False).flatten()
        if assigned_clusters.numel() != 1:
            raise ValueError(
                f"Client {i} must be assigned to exactly one cluster, found {assigned_clusters.numel()}."
            )

        q = int(assigned_clusters.item())
        facility = active_facility_list[q]
        total_distance += float(graph._distances[facility, i])

    return total_distance

def calculate_radius(graph, facilities):
    """
    Compute the k-center radius given a set of facilities.

    :param graph: object containing the distance matrix `graph._distances` (NxN tensor)
    :param facilities: list or tensor of facility indices
    :return: the k-center radius (max distance from any client to nearest facility)
    """
    # Create a subgraph: each column corresponds to a facility
    distance_subgraph = graph._distances[:, facilities]  # shape: (n_clients, n_facilities)

    # For each client, find distance to closest facility
    min_distances, _ = torch.min(distance_subgraph, dim=1)  # shape: (n_clients,)

    # k-center radius = worst-covered client
    radius = torch.max(min_distances)

    return float(radius)






def get_facilities(h, n, k):
    facilities = set()
    for j in range(k):
        for s in range(n):
            index = (n * k) + s + (j * n)
            if h._V[index] == 1:
                facilities.add(s)
    return list(facilities)
