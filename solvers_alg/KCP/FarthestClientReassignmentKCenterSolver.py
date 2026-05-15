import torch

from problems.KCProblem import KCProblem
from solvers_alg.solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class FarthestClientReassignmentKCenterSolver(KCPSolver):
    def __init__(self):
        # Initialize variables for the solver.
        self._name = "Farthest Client Reassignment K-Center"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._facilities = None

    def initialize(self, problem: KCProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()

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
        if self._graph is None or self._n is None or self._k is None or self._facilities is None:
            raise ValueError("Solver must be initialized before calling solve().")

        if self._k == 0:
            self._selectedFacilities = []
            self._solutionValue = 0
            return

        self._initialize_facilities()

        max_iterations = self._n * max(self._k, 1)
        iterations = 0

        while iterations < max_iterations:
            selected_facilities, current_radius = self._calculate_facilities_and_distance()
            self._assert_valid_facility_count(selected_facilities)

            farthest_client = self.get_farthest_client(selected_facilities)
            if farthest_client in selected_facilities:
                break

            best_removed_facility = None
            best_radius = current_radius

            # Evaluate swapping in the farthest client for each current facility.
            for facility in selected_facilities:
                if facility == farthest_client:
                    continue

                self._facilities[0, facility] = 0
                self._facilities[0, farthest_client] = 1
                self._assert_valid_facility_count()

                candidate_facilities, candidate_radius = self._calculate_facilities_and_distance()
                self._assert_valid_facility_count(candidate_facilities)

                if candidate_radius < best_radius:
                    best_radius = candidate_radius
                    best_removed_facility = facility

                self._facilities[0, farthest_client] = 0
                self._facilities[0, facility] = 1
                self._assert_valid_facility_count()

            if best_removed_facility is None:
                break

            self._facilities[0, best_removed_facility] = 0
            self._facilities[0, farthest_client] = 1
            self._assert_valid_facility_count()
            iterations += 1

        final_facilities, final_radius = self._calculate_facilities_and_distance()
        self._assert_valid_facility_count(final_facilities)
        self._selectedFacilities = final_facilities
        self._solutionValue = final_radius

    def get_farthest_client(self, selected_facilities):
        distances = self._graph._normalized_distances[:, selected_facilities]
        min_distances, _ = torch.min(distances, dim=1)
        return torch.argmax(min_distances).item()

    def _initialize_facilities(self):
        self._facilities.zero_()

        distances = self._graph._normalized_distances

        # First center: node with the smallest total distance to all other nodes.
        first_center = torch.argmin(distances.sum(dim=1)).item()
        selected = [first_center]
        self._facilities[0, first_center] = 1

        selected_mask = torch.zeros(self._n, dtype=torch.bool, device=distances.device)
        selected_mask[first_center] = True
        current_min_dist = distances[:, first_center].clone()

        for _ in range(1, self._k):
            candidate_distances = current_min_dist.masked_fill(selected_mask, float("-inf"))
            next_center = torch.argmax(candidate_distances).item()

            selected.append(next_center)
            selected_mask[next_center] = True
            self._facilities[0, next_center] = 1
            current_min_dist = torch.minimum(current_min_dist, distances[:, next_center])

        self._assert_valid_facility_count(selected)

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
