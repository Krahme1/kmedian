import random

import torch

from problems.KCProblem import KCProblem
from solvers_alg.solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class RandomizedSwapKCenterSolver(KCPSolver):
    def __init__(self):
        self._name = "Randomized Swap K-Center"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._facilities = None
        self._distances = None

        self._attempted_moves = 0
        self._accepted_moves = 0

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
        if self._graph is None or self._n is None or self._k is None or self._facilities is None:
            raise ValueError("Solver must be initialized before calling solve().")

        if self._k == 0:
            self._selectedFacilities = []
            self._solutionValue = 0
            return

        self._initialize_facilities()

        current_facilities, current_radius = self._calculate_facilities_and_distance()
        self._assert_valid_facility_count(current_facilities)

        max_attempts = 1000
        no_improve_limit = 100
        no_improve_counter = 0

        self._attempted_moves = 0
        self._accepted_moves = 0

        while self._attempted_moves < max_attempts and no_improve_counter < no_improve_limit:
            remove_node, add_node = self._random_swap_move()
            if remove_node is None or add_node is None:
                break

            self._attempted_moves += 1

            self._facilities[0, remove_node] = 0
            self._facilities[0, add_node] = 1
            self._ensure_exactly_k_facilities()

            candidate_facilities, candidate_radius = self._calculate_facilities_and_distance()
            self._assert_valid_facility_count(candidate_facilities)

            if candidate_radius < current_radius:
                current_facilities = candidate_facilities
                current_radius = candidate_radius
                self._accepted_moves += 1
                no_improve_counter = 0
            else:
                self._facilities[0, add_node] = 0
                self._facilities[0, remove_node] = 1
                self._ensure_exactly_k_facilities()
                no_improve_counter += 1

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

    def _get_selected_and_nonselected(self):
        selected = []
        nonselected = []

        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected.append(i)
            else:
                nonselected.append(i)

        return selected, nonselected

    def _random_swap_move(self):
        selected, nonselected = self._get_selected_and_nonselected()

        if len(selected) == 0 or len(nonselected) == 0:
            return None, None

        remove_node = random.choice(selected)
        add_node = random.choice(nonselected)

        if add_node in selected:
            return None, None

        return remove_node, add_node

    def _ensure_exactly_k_facilities(self):
        selected, nonselected = self._get_selected_and_nonselected()

        if len(selected) > self._k:
            for index in selected[self._k:]:
                self._facilities[0, index] = 0
        elif len(selected) < self._k:
            needed = self._k - len(selected)
            for index in random.sample(nonselected, k=needed):
                self._facilities[0, index] = 1

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
