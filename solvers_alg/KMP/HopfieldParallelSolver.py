import random

import torch

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class HopfieldParallelSolver(KMPSolver):
    def __init__(self, use_gpu, seed=None):
        self._name = "Hopfield Parallel Solver"
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

    def set_seed(self, seed: int):
        self._rng = random.Random(seed)

    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        # Initialize distance values
        if self._use_gpu:
            self._distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            self._distance_values = (1 - self._graph._normalized_distances).clone().detach()

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

    def solve(self, runNum=None, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        self._initialize_per_run_arrays(starter_facilities)

        iterations = 0

        # Evaluate initial solution
        stabilized = False

        while not stabilized:
            prev_C = self._client_activation_values.clone()
            prev_F = self._facility_activation_values.clone()

            # Client updates
            self._calculate_client_values()
            self._update_client()

            # Facility updates
            self._calculate_facility_values()
            self._update_facility()

            # Fixed-point condition
            stabilized = (
                torch.equal(prev_C, self._client_activation_values) and
                torch.equal(prev_F, self._facility_activation_values)
            )
            iterations += 1

            tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
            print("Iteration", iterations, "distance:", tmp_solution_value)

        print(f"Converged in {iterations} iterations.")
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()
        print(f"Distance: {self._solutionValue}")


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



    # Greedy BUILD initializer
    def _warm_start_facilities_greedy_deterministic(self) -> list[int]:
        # Work on CPU for simple deterministic indexing and masking.
        D = self._distance_values.detach().to("cpu")
        num_candidates = D.shape[1]

        # D stores similarity (1 - normalized distance), so we maximize totals.
        # First facility: argmax_f sum_i D[i, f]
        col_sums = D.sum(dim=0)
        first_facility = torch.argmax(col_sums).item()

        selected = [first_facility]
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool)
        selected_mask[first_facility] = True

        # Maintain max similarity to selected set for each client i.
        current_max_sim = D[:, first_facility].clone()

        for _ in range(1, self._k):
            # For each candidate f, compute sum_i max(current_max_sim[i], D[i, f]).
            candidate_scores = torch.maximum(current_max_sim.unsqueeze(1), D).sum(dim=0)
            candidate_scores[selected_mask] = float("-inf")

            next_facility = torch.argmax(candidate_scores).item()
            selected.append(next_facility)
            selected_mask[next_facility] = True

            # Update maintained maximum similarities.
            current_max_sim = torch.maximum(current_max_sim, D[:, next_facility])

        return selected

    # Optional random first greedy initializer (currently not used).
    def _warm_start_facilities_greedy_random_first(self) -> list[int]:
        D = self._distance_values.detach().to("cpu")
        num_candidates = D.shape[1]

        # First facility: random, controlled by solver-local RNG.
        first_facility = self._rng.randrange(num_candidates)

        selected = [first_facility]
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool)
        selected_mask[first_facility] = True

        # Maintain max similarity to selected set for each client i.
        current_max_sim = D[:, first_facility].clone()

        for _ in range(1, self._k):
            # For each candidate f, compute sum_i max(current_max_sim[i], D[i, f]).
            candidate_scores = torch.maximum(current_max_sim.unsqueeze(1), D).sum(dim=0)
            candidate_scores[selected_mask] = float("-inf")

            next_facility = torch.argmax(candidate_scores).item()
            selected.append(next_facility)
            selected_mask[next_facility] = True

            # Update maintained maximum similarities.
            current_max_sim = torch.maximum(current_max_sim, D[:, next_facility])

        return selected

    def _calculate_client_values(self):
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _update_client(self):
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        # Find the maximum value in each row of client_inner_values and set corresponding activation to 1
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices, max_indices] = 1

    def _calculate_facility_values(self):
        C = self._client_activation_values.to(self._distance_values.dtype)    # (n,k)
        self._facility_inner_values = self._distance_values.t() @ C           # (n,n)^T @ (n,k) -> (n,k)

        # """
        # PF[j, q] = sum_{i assigned to cluster q} D[i, j]
        # """
        # D = self._distance_values  # (n, n)

        # # (n, k) float or same dtype as D
        # C = self._client_activation_values.to(D.dtype)

        # self._facility_inner_values = torch.zeros(
        #     (self._n, self._k), dtype=D.dtype, device=self._device
        # )

        # for q in range(self._k):
        #     # Find clients assigned to cluster q
        #     assigned_mask = C[:, q] > 0  # (n,) bool

        #     if torch.any(assigned_mask):
        #         # Sum distances from those clients to every candidate center j
        #         # D[assigned_mask, :] has shape (#assigned, n)
        #         self._facility_inner_values[:, q] = D[assigned_mask, :].sum(dim=0)
        #     else:
        #         # Edge case: no clients assigned to this cluster.
        #         # You can choose what you want here; this keeps costs at 0.
        #         self._facility_inner_values[:, q] = 0.0

    def _update_facility(self):
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        chosen = []

        for q in range(self._k):
            costs = self._facility_inner_values[:, q]  # (n,)
            j = torch.argmax(costs).item()
            chosen.append(j)
            self._facility_activation_values[j, q] = 1

        self._active_facility_list = chosen

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._active_facility_list)
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        return selected_facilities, selected_distance
