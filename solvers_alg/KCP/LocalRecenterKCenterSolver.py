import torch

from problems.KCProblem import KCProblem
from solvers_alg.solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class LocalRecenterKCenterSolver(KCPSolver):
    def __init__(self):
        self._name = "Local Recenter K-Center"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._facilities = None
        self._distances = None

    def initialize(self, problem: KCProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        self._distances = self._graph._normalized_distances

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        if self._k < 0 or self._k > self._n:
            raise ValueError("k must satisfy 0 <= k <= n.")

        self._facilities = torch.zeros(size=(1, self._n), dtype=torch.int)

    def getName(self):
        return self._name

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def solve(self, runNum=None):
        if self._k == 0:
            self._selectedFacilities = []
            self._solutionValue = 0
            return

        self._initialize_facilities()

        while True:
            selected_facilities, current_radius = self._calculate_facilities_and_distance()
            cluster_map = self._assign_clients_to_selected(selected_facilities)

            proposed_centers = []
            for facility in selected_facilities:
                cluster_nodes = cluster_map[facility]
                proposed_centers.append(self._best_center_for_cluster(cluster_nodes))

            unique_centers = []
            for center in proposed_centers:
                if center not in unique_centers:
                    unique_centers.append(center)

            while len(unique_centers) < self._k:
                min_distances = self._get_min_distances_to_selected(unique_centers)
                selected_mask = torch.zeros(self._n, dtype=torch.bool, device=min_distances.device)
                selected_mask[unique_centers] = True
                candidate_scores = min_distances.masked_fill(selected_mask, float("-inf"))
                next_center = torch.argmax(candidate_scores).item()
                if next_center in unique_centers:
                    break
                unique_centers.append(next_center)

            self._facilities = self._build_facility_tensor_from_indices(unique_centers)
            self._ensure_exactly_k_facilities()

            candidate_facilities, candidate_radius = self._calculate_facilities_and_distance()

            if candidate_radius < current_radius:
                self._selectedFacilities = candidate_facilities
                self._solutionValue = candidate_radius
                continue

            self._facilities = self._build_facility_tensor_from_indices(selected_facilities)
            self._ensure_exactly_k_facilities()
            break

        final_facilities, final_radius = self._calculate_facilities_and_distance()
        self._selectedFacilities = final_facilities
        self._solutionValue = final_radius

    def _initialize_facilities(self):
        self._facilities.zero_()

        # First center: node with the smallest total distance to all other nodes.
        first_center = torch.argmin(self._distances.sum(dim=1)).item()
        selected = [first_center]
        self._facilities[0, first_center] = 1

        selected_mask = torch.zeros(self._n, dtype=torch.bool, device=self._distances.device)
        selected_mask[first_center] = True
        current_min_dist = self._distances[:, first_center].clone()

        for _ in range(1, self._k):
            candidate_distances = current_min_dist.masked_fill(selected_mask, float("-inf"))
            next_center = torch.argmax(candidate_distances).item()

            selected.append(next_center)
            selected_mask[next_center] = True
            self._facilities[0, next_center] = 1
            current_min_dist = torch.minimum(current_min_dist, self._distances[:, next_center])

        self._assert_valid_facility_count(selected)

    def _assign_clients_to_selected(self, selected_facilities):
        assignments = {facility: [] for facility in selected_facilities}
        distances = self._distances[:, selected_facilities]
        nearest_indices = torch.argmin(distances, dim=1)

        for client in range(self._n):
            assigned_facility = selected_facilities[nearest_indices[client].item()]
            assignments[assigned_facility].append(client)

        return assignments

    def _best_center_for_cluster(self, cluster_nodes):
        if len(cluster_nodes) == 1:
            return cluster_nodes[0]

        cluster_tensor = torch.tensor(cluster_nodes, dtype=torch.long)
        cluster_distances = self._graph._normalized_distances[cluster_tensor][:, cluster_tensor]
        max_distances = torch.max(cluster_distances, dim=1).values
        best_index = torch.argmin(max_distances).item()
        return cluster_nodes[best_index]

    def _build_facility_tensor_from_indices(self, indices):
        facilities = torch.zeros(size=(1, self._n), dtype=torch.int)
        facilities[0, indices] = 1
        return facilities

    def _get_min_distances_to_selected(self, selected_facilities):
        if len(selected_facilities) == 0:
            return torch.full((self._n,), float("inf"), dtype=self._graph._normalized_distances.dtype)

        distances = self._graph._normalized_distances[:, selected_facilities]
        min_distances, _ = torch.min(distances, dim=1)
        return min_distances

    def _ensure_exactly_k_facilities(self):
        selected_indices = torch.nonzero(self._facilities[0] == 1, as_tuple=False).flatten().tolist()

        if len(selected_indices) > self._k:
            for index in selected_indices[self._k:]:
                self._facilities[0, index] = 0
        elif len(selected_indices) < self._k:
            min_distances = self._get_min_distances_to_selected(selected_indices)
            selected_mask = torch.zeros(self._n, dtype=torch.bool, device=min_distances.device)
            if len(selected_indices) > 0:
                selected_mask[selected_indices] = True

            while len(selected_indices) < self._k:
                candidate_scores = min_distances.masked_fill(selected_mask, float("-inf"))
                next_center = torch.argmax(candidate_scores).item()
                if selected_mask[next_center]:
                    break
                self._facilities[0, next_center] = 1
                selected_indices.append(next_center)
                selected_mask[next_center] = True
                min_distances = torch.minimum(min_distances, self._graph._normalized_distances[:, next_center])

        self._assert_valid_facility_count()

    def _calculate_facilities_and_distance(self):
        selected_facilities = []
        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected_facilities.append(i)
        radius = calculate_radius(self._graph, selected_facilities)
        return selected_facilities, radius

    def _assert_valid_facility_count(self, selected_facilities=None):
        if selected_facilities is None:
            count = int(torch.sum(self._facilities).item())
        else:
            count = len(selected_facilities)

        assert count == self._k, f"Expected exactly {self._k} facilities, found {count}."
