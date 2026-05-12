import torch

from problems.KCProblem import KCProblem
from solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class DropWorstFacilityKCenterSolver(KCPSolver):
    def __init__(self):
        self._name = "Drop Worst Facility K-Center"
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

            candidate_facility = self._get_farthest_nonselected_client(selected_facilities)
            if candidate_facility is None:
                break

            self._facilities[0, candidate_facility] = 1
            weakest_facility, best_radius = self._find_weakest_facility_after_add(
                selected_facilities,
                candidate_facility
            )
            self._facilities[0, candidate_facility] = 0
            self._assert_valid_facility_count(selected_facilities)

            if weakest_facility is None or best_radius >= current_radius:
                break

            self._facilities[0, weakest_facility] = 0
            self._facilities[0, candidate_facility] = 1
            self._ensure_exactly_k_facilities()

        final_facilities, final_radius = self._calculate_facilities_and_distance()
        self._assert_valid_facility_count(final_facilities)
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

    def _get_min_distances_to_selected(self, selected_facilities):
        facility_distances = self._distances[:, selected_facilities]
        min_distances, _ = torch.min(facility_distances, dim=1)
        return min_distances

    def _get_farthest_nonselected_client(self, selected_facilities):
        min_distances = self._get_min_distances_to_selected(selected_facilities)

        selected_mask = torch.zeros(self._n, dtype=torch.bool, device=min_distances.device)
        selected_mask[selected_facilities] = True

        candidate_scores = min_distances.masked_fill(selected_mask, float("-inf"))
        farthest_index = torch.argmax(candidate_scores).item()

        if selected_mask[farthest_index]:
            return None

        return farthest_index

    def _find_weakest_facility_after_add(self, selected_facilities, added_facility):
        best_removed_facility = None
        best_radius = float("inf")

        for facility in selected_facilities:
            if facility == added_facility:
                continue

            self._facilities[0, facility] = 0
            candidate_facilities, candidate_radius = self._calculate_facilities_and_distance()

            if len(candidate_facilities) == self._k and candidate_radius < best_radius:
                best_radius = candidate_radius
                best_removed_facility = facility

            self._facilities[0, facility] = 1

        return best_removed_facility, best_radius

    def _ensure_exactly_k_facilities(self):
        selected_indices = torch.nonzero(self._facilities[0] == 1, as_tuple=False).flatten().tolist()

        if len(selected_indices) > self._k:
            for index in selected_indices[self._k:]:
                self._facilities[0, index] = 0
        elif len(selected_indices) < self._k:
            for index in range(self._n):
                if self._facilities[0, index] == 0:
                    self._facilities[0, index] = 1
                    selected_indices.append(index)
                    if len(selected_indices) == self._k:
                        break

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
