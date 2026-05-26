import random
import time

import torch

from problems.KFProblem import KFProblem
from solvers_alg.solvers.brute_solver import calculate_distance_with_facility_cost
from solvers_alg.KF.KFSolver import KFSolver


FACILITY = 1
CLIENT = 0


class HopfieldOriginalDummy(KFSolver):
    def __init__(self, use_gpu, use_dummy_facilities=True, require_exact_k_real=True):
        self._name = "Hopfield K-Facility with Dummy Facilities"
        self._solutionValue = 0
        self._selectedFacilities = []

        self.verbose = False
        self._n = None
        self._k = None
        self._graph = None
        self._costs = None

        self._num_rows = None
        self._num_cols = None
        self._size = None

        self._num_facility_candidates = None
        self._num_dummy_facilities = 0
        self._dummy_start_index = None

        self._use_dummy_facilities = use_dummy_facilities
        self._require_exact_k_real = require_exact_k_real

        self._facility_update_value = 1.0
        self._client_update_value = 1.0

        self._use_gpu = use_gpu

        if use_gpu:
            self._device = "cuda" if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = "cpu"

        self._full_distance_values = None
        self._distance_values = None

        self._facility_inner_values = None
        self._client_inner_values = None
        self._facility_activation_values = None
        self._client_activation_values = None
        self._candidatefacility_inner_values = None

        self._math_row_indices = None
        self._k_indices = None
        self._facilities = None
        self._active_facility_list = []

        self._sorted_facility_inner_values = None
        self._sorted_facility_indices = None

    def initialize(self, problem: KFProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        self._costs = problem.getCosts()

        if self._graph is None or self._n is None or self._k is None or self._costs is None:
            raise ValueError("Graph, n, k, and facility costs must be set before calling initialize().")

        self._num_dummy_facilities = self._k - 1 if self._use_dummy_facilities else 0
        self._dummy_start_index = self._n
        self._num_facility_candidates = self._n + self._num_dummy_facilities

        self._num_rows = self._num_facility_candidates
        self._num_cols = self._k
        self._size = (self._num_facility_candidates, self._k)

        if self._use_gpu:
            real_distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            real_distance_values = (1 - self._graph._normalized_distances).clone().detach()

        if self._num_dummy_facilities > 0:
            dummy_distance_values = torch.ones(
                size=(self._n, self._num_dummy_facilities),
                device=self._device,
            )
            self._full_distance_values = torch.cat(
                [real_distance_values, dummy_distance_values],
                dim=1,
            )
        else:
            self._full_distance_values = real_distance_values

        self._distance_values = self._full_distance_values

        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._client_inner_values = torch.zeros(size=(self._n, self._k), device=self._device)

        self._facility_activation_values = torch.zeros(
            size=self._size,
            dtype=torch.int,
            device=self._device,
        )
        self._client_activation_values = torch.zeros(
            size=(self._n, self._k),
            dtype=torch.int,
            device=self._device,
        )

        self._candidatefacility_inner_values = torch.zeros(
            size=(self._n, self._num_facility_candidates),
            device=self._device,
        )

        self._math_row_indices = torch.arange(start=0, end=self._n, device=self._device)
        self._k_indices = torch.arange(start=0, end=self._k, device=self._device)

        self._facilities = torch.zeros(
            size=(1, self._num_facility_candidates),
            dtype=torch.int,
            device=self._device,
        )

    def getName(self):
        return self._name

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def getSolutionValue(self):
        return self._solutionValue

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def solve(self, runNum=None, starter_facilities=None):
        start_time = time.time()
        max_time = 235

        if starter_facilities is not None:
            self._initialize_per_run_arrays(starter_facilities)
        else:
            self._initialize_per_run_arrays()

        facility_stabilized = False
        iterations = 0

        while not facility_stabilized:
            if time.time() - start_time >= max_time:
                break

            max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
            sumBefore = torch.sum(max_values).item()

            self._sorted_facility_inner_values, self._sorted_facility_indices = torch.topk(
                max_values,
                self._k,
            )
            worstFacility = self._sorted_facility_indices[self._k - 1]
            cluster_index = max_indices[worstFacility]

            self._facility_activation_values[worstFacility, cluster_index] = 0
            self._facilities[0, worstFacility] = 0

            self._client_inner_values[:, cluster_index] = 0

            client_max_values, client_max_indices = torch.max(self._client_inner_values, dim=1)

            inactive_mask = (self._facilities == 0).view(-1)

            self._candidatefacility_inner_values = (
                self._distance_values - client_max_values.unsqueeze(1)
            ).clamp_min(0)

            self._candidatefacility_inner_values[:, ~inactive_mask] = 0

            candidate_gains = torch.sum(self._candidatefacility_inner_values, dim=0)

            old_facility = int(worstFacility.item())

            if self._is_dummy_facility(old_facility):
                old_cost = 0.0
            else:
                old_cost = float(self._costs[old_facility])

            cost_adjustments = torch.zeros(
                size=(self._num_facility_candidates,),
                device=self._device,
            )

            for candidate in range(self._n):
                cost_adjustments[candidate] = old_cost - float(self._costs[candidate])

            if self._num_dummy_facilities > 0:
                cost_adjustments[self._dummy_start_index:] = old_cost

            self._facility_inner_values[:, cluster_index] = candidate_gains + cost_adjustments

            bestFacility = torch.argmax(self._facility_inner_values[:, cluster_index])

            self._active_facility_list[cluster_index] = bestFacility.item()
            self._facility_activation_values[bestFacility, cluster_index] = 1
            self._facilities[0, bestFacility] = 1

            self._calculate_client_values()
            self._update_client()
            self._calculate_facility_values()

            max_values_after, max_indices_after = torch.max(self._facility_inner_values, dim=1)
            sumAfter = torch.sum(max_values_after).item()

            current_real_count = self._count_real_active_facilities()
            solution_stabilized = sumBefore >= sumAfter
            exact_k_real = current_real_count == self._k

            if solution_stabilized and (not self._require_exact_k_real or exact_k_real):
                facility_stabilized = True

                self._facility_activation_values[bestFacility.item(), cluster_index] = 0
                self._facility_activation_values[worstFacility.item(), cluster_index] = 1

                self._facilities[0, bestFacility] = 0
                self._facilities[0, worstFacility] = 1

                self._active_facility_list[cluster_index] = worstFacility.item()
            else:
                facility_stabilized = False

            iterations += 1

        print(f"Converged in {iterations} iterations.")
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()
        print(f"Selected facilities: {self._selectedFacilities}")
        print(f"Distance with facility cost: {self._solutionValue}")

        return self._selectedFacilities

    def _initialize_per_run_arrays(self, starter_facilities=None):
        self._facility_activation_values = torch.zeros(
            size=self._size,
            dtype=torch.int,
            device=self._device,
        )
        self._facilities = torch.zeros(
            size=(1, self._num_facility_candidates),
            dtype=torch.int,
            device=self._device,
        )
        self._active_facility_list = []

        if starter_facilities is None:
            initial_set = random.sample(range(0, self._n), k=self._k)
        else:
            initial_set = starter_facilities

            for value in initial_set:
                if self._is_dummy_facility(value):
                    raise ValueError(
                        "starter_facilities should contain real graph nodes only, not dummy facilities."
                    )

        index = 0

        for value in initial_set:
            self._facility_activation_values[value, index] = 1
            self._facilities[0, value] = 1
            self._active_facility_list.append(value)
            index += 1

        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

        tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
        print("Initial distance with facility cost:", tmp_solution_value)

    def _calculate_facilities_and_distance(self):
        selected_facilities = []

        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected_facilities.append(i)

        if self._require_exact_k_real and len(selected_facilities) != self._k:
            raise ValueError(
                f"Exact-k mode requires {self._k} real facilities, "
                f"but the neural state contains {len(selected_facilities)} real facilities."
            )

        selected_distance = self._calculate_kf_cost(selected_facilities)

        return selected_facilities, selected_distance

    def _calculate_kf_cost(self, selected_facilities):
        if selected_facilities is None or len(selected_facilities) == 0:
            return float("inf")

        return calculate_distance_with_facility_cost(self._graph, selected_facilities, self._costs, self._n)

    def _is_dummy_facility(self, facility_index):
        return facility_index >= self._dummy_start_index

    def _count_real_active_facilities(self):
        real_count = 0

        for facility in self._active_facility_list:
            if not self._is_dummy_facility(facility):
                real_count += 1

        return real_count

    def _create_distance_array(self):
        return torch.tensor(self._graph.distance_cache, device=self._device)

    def _calculate_client_values(self):
        self._client_inner_values[:, :] = self._distance_values[:, self._active_facility_list]

    def _calculate_facility_values(self):
        self._facility_inner_values = torch.zeros(
            size=(self._num_facility_candidates, self._k),
            device=self._device,
        )

        cluster_values = torch.sum(
            self._client_inner_values * self._client_activation_values,
            dim=0,
        )

        for cluster_index, facility in enumerate(self._active_facility_list):
            if self._is_dummy_facility(facility):
                opening_cost = 0.0
            else:
                opening_cost = float(self._costs[facility])

            self._facility_inner_values[facility, cluster_index] = (
                cluster_values[cluster_index] - opening_cost
            )

    def _update_client(self):
        self._client_activation_values = torch.zeros(
            size=(self._n, self._k),
            dtype=torch.int,
            device=self._device,
        )

        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices, max_indices] = 1
