import random

import torch

from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class HopfieldThirdParallelSolver(KMPSolver):
    def __init__(self, use_gpu, seed=None):
        self._name = "Hopfield Third Parallel Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._matrix_n = None
        self._k = None
        self._graph = None

        self._rng = random.Random(seed)
        # self._seed = seed

        self._num_rows = None
        self._num_cols = None
        self._size = None
        self._facility_update_value = 1.0
        self._client_update_value = 1.0

        self._client_inner_values = None     # PC (n,k)
        self._facility_inner_values = None   # PF (n,k)
        self._client_activation_values = None  # C (n,k) int
        self._facility_activation_values = None  # F (n,k) int

        self._distance_values = None  # D (n,n)

        self._active_facility_list = []  # length k: center index per cluster

        # CPU/GPU toggle
        self._use_gpu = use_gpu

        # If we select the GPU and cuda is not available, fail loudly.
        if use_gpu:
            self._device = 'cuda' if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = 'cpu'

        self._math_row_indices = None
        self._k_indices = None

        self._alpha = 0
        self._beta = 1.0
        self._gamma = 0
        self._max_iterations = 1000

    def set_seed(self, seed: int):
        self._rng = random.Random(seed)

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

    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        # Initialize distance values
        if self._use_gpu:
            self._distance_values = self._graph._gpu_normalized_distances
        else:
            self._distance_values = self._graph._normalized_distances.clone().detach()

        if self._distance_values.ndim != 2:
            raise ValueError(
                f"Expected a 2D distance matrix, got shape {tuple(self._distance_values.shape)}."
            )

        num_clients, num_facility_candidates = self._distance_values.shape
        if num_clients != num_facility_candidates:
            raise ValueError(
                "HopfieldParallelSolver expects a square distance matrix for k-median. "
                f"Got shape {tuple(self._distance_values.shape)}."
            )
        if self._k > num_facility_candidates:
            raise ValueError(
                f"k={self._k} cannot exceed number of facility candidates ({num_facility_candidates})."
            )

        if self._n != num_clients:
            print(
                "Warning: problem n does not match distance matrix size. "
                f"Using matrix size n={num_clients} for tensor ops "
                f"(problem reported n={self._n})."
            )
        self._matrix_n = num_clients

        self._num_rows = self._matrix_n
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)

        self._math_row_indices = torch.arange(0, self._matrix_n, device=self._device)
        self._k_indices = torch.arange(0, self._k, device=self._device)

        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._client_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)

    def solve(self, runNum=None, starter_facilities=None):
        self._initialize_per_run_arrays(starter_facilities)

        current_facilities, current_distance = self._calculate_facilities_and_distance()
        best_facilities = list(current_facilities)
        best_distance = current_distance
        iterations = 0

        while iterations < self._max_iterations:
            prev_C = self._client_activation_values.clone()
            prev_F = self._facility_activation_values.clone()
            prev_active_facility_list = list(self._active_facility_list)
            prev_distance = current_distance

            # Client updates
            self._calculate_client_values()
            self._update_client()

            # Facility updates
            self._calculate_facility_values()
            self._update_facility()

            stabilized = (
                torch.equal(prev_C, self._client_activation_values) and
                torch.equal(prev_F, self._facility_activation_values)
            )
            iterations += 1

            tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()

            if tmp_solution_value >= prev_distance:
                self._client_activation_values = prev_C
                self._facility_activation_values = prev_F
                self._active_facility_list = prev_active_facility_list
                self._selectedFacilities = list(best_facilities)
                self._solutionValue = best_distance
                print(
                    f"Rejected iteration {iterations} because distance did not improve "
                    f"({tmp_solution_value} >= {prev_distance})."
                )
                print(f"Best distance: {self._solutionValue}")
                return

            current_facilities = list(tmp_selected_facilities)
            current_distance = tmp_solution_value
            print("Iteration", iterations, "distance:", current_distance)

            if current_distance < best_distance:
                best_facilities = list(current_facilities)
                best_distance = current_distance

            if stabilized:
                break

        if iterations >= self._max_iterations:
            print(f"Stopped after reaching max iterations ({self._max_iterations}).")
        else:
            print(f"Converged in {iterations} iterations.")

        self._selectedFacilities = list(best_facilities)
        self._solutionValue = best_distance
        print(f"Best distance: {self._solutionValue}")

    def _initialize_per_run_arrays(self, starter_facilities):
        self._client_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)
        self._facility_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)

        self._active_facility_list = []

        # Use explicit warm start if provided
        if starter_facilities is not None and len(starter_facilities) > 0:
            if len(starter_facilities) != self._k:
                raise ValueError("starter_facilities must have length k.")
            if any(f < 0 or f >= self._matrix_n for f in starter_facilities):
                raise ValueError(f"starter_facilities must be within [0, {self._matrix_n - 1}].")
            initial_facilities = list(starter_facilities)
        else:
            initial_facilities = self._rng.sample([i for i in range(0, self._n)], k=self._k)

        index = 0
        for value in initial_facilities:
            self._facility_activation_values[value, index] = 1
            self._active_facility_list.append(value)
            index = index + 1

        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

        tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
        print("Initial distance:", tmp_solution_value)

    def _compute_dciq_and_Dminusq(self):
        d_ci_q = self._distance_values[:, self._active_facility_list]   # (n, k)

        D_minus_q = torch.empty(
            (self._matrix_n, self._k),
            dtype=self._distance_values.dtype,
            device=self._device
        )

        for q in range(self._k):
            other_centers = [self._active_facility_list[r] for r in range(self._k) if r != q]

            if len(other_centers) == 0:
                D_minus_q[:, q] = float("inf")
            else:
                D_minus_q[:, q] = torch.min(
                    self._distance_values[:, other_centers],
                    dim=1
                ).values

        return d_ci_q, D_minus_q
    
    def _calculate_client_values(self):
        d_ci_q, D_minus_q = self._compute_dciq_and_Dminusq()

        cond = D_minus_q > (1.0 + self._alpha) * d_ci_q

        self._client_inner_values = torch.where(
            cond,
            (1.0 - self._alpha) * d_ci_q,
            2.0 * d_ci_q - D_minus_q
        )

    def _update_client(self):
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        # Find the maximum value in each row of client_inner_values and set corresponding activation to 1
        max_indices = torch.argmin(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices, max_indices] = 1

    def _update_facility(self):
        self._facility_activation_values = torch.zeros(
        size=self._size, dtype=torch.int, device=self._device
        )

        chosen = []
        used_mask = torch.zeros(self._matrix_n, dtype=torch.bool, device=self._device)

        for q in range(self._k):
            costs = self._facility_inner_values[:, q].clone()

            # Prevent already chosen facilities from being selected again
            costs[used_mask] = float("inf")

            j = torch.argmin(costs).item()

            chosen.append(j)
            used_mask[j] = True
            self._facility_activation_values[j, q] = 1

        self._active_facility_list = chosen

    def _calculate_facility_values(self):
        _, D_minus_q = self._compute_dciq_and_Dminusq()
        D = self._distance_values  # shape (n, n), D[i, j] = d_ij

        self._facility_inner_values = torch.empty(
            (self._matrix_n, self._k),
            dtype=D.dtype,
            device=self._device
        )

        for q in range(self._k):
            # base_matrix[i, j] = min(d_ij, D_i^{-q})
            base_matrix = torch.minimum(
                D,
                D_minus_q[:, q].unsqueeze(1)
            )  # shape (n, n)

            # base_sum[j] = sum_i min(d_ij, D_i^{-q})
            base_sum = base_matrix.sum(dim=0)  # shape (n,)

            # penalty must be indexed by candidate facility j
            D_minus_q_at_candidate = D_minus_q[:, q]  # this contains D_i^{-q} for all nodes i
            # when interpreted as candidates, index j refers to the candidate node j

            threshold = self._beta * (self._matrix_n / self._k) * base_sum

            delta = torch.where(
                D_minus_q_at_candidate > threshold,
                torch.zeros_like(base_sum),
                self._gamma * D_minus_q_at_candidate
            )

            # pf_jq for all candidate facilities j
            self._facility_inner_values[:, q] = base_sum + delta

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._active_facility_list)
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        return selected_facilities, selected_distance