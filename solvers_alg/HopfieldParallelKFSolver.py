import random

import torch

from problems.KFProblem import KFProblem
from solvers.brute_solver import calculate_distance_with_facility_cost
from solvers_alg.KFSolver import KFSolver


class HopfieldParallelKFSolver(KFSolver):
    def __init__(self, use_gpu, seed=None):
        self._name = "Hopfield Parallel Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._matrix_n = None
        self._k = None
        self._graph = None

        self._costs = None

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

    def initialize(self, problem:KFProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        self._costs = problem.getCosts()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        # Initialize distance values.
        # For k-facility location, we use the normal normalized distances because
        # the objective is a minimization objective: assignment distance + opening cost.
        if self._use_gpu:
            self._distance_values = self._graph._gpu_normalized_distances
        else:
            self._distance_values = self._graph._normalized_distances.clone().detach()

        if self._distance_values.ndim != 2:
            raise ValueError(
                f"Expected a 2D distance matrix, got shape {tuple(self._distance_values.shape)}."
            )

        num_clients, num_facility_candidates = self._distance_values.shape

        if self._costs is None:
            raise ValueError("Facility costs must be set before calling initialize().")

        if isinstance(self._costs, torch.Tensor):
            self._cost_values = self._costs.clone().detach().to(self._device, dtype=self._distance_values.dtype)
        elif isinstance(self._costs, dict):
            self._cost_values = torch.tensor(
                [self._costs[i] for i in range(self._n)],
                dtype=self._distance_values.dtype,
                device=self._device
            )
        else:
            self._cost_values = torch.tensor(
                self._costs,
                dtype=self._distance_values.dtype,
                device=self._device
            )

        if self._cost_values.numel() != num_facility_candidates:
            raise ValueError(
                f"Expected {num_facility_candidates} facility costs, got {self._cost_values.numel()}."
            )

        cost_min = torch.min(self._cost_values)
        cost_max = torch.max(self._cost_values)
        if torch.isclose(cost_max, cost_min):
            self._normalized_cost_values = torch.zeros_like(self._cost_values)
        else:
            self._normalized_cost_values = (self._cost_values - cost_min) / (cost_max - cost_min)

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

    def setCosts(self, costs):
        self._costs = costs

    def solve(self, runNum=None, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = calculate_distance_with_facility_cost(self._graph, best_facilities, self._costs, self._n) if starter_facilities else None

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
        print("Number of selected facilities:", len(self._selectedFacilities))
        print("Selected facilities:", self._selectedFacilities)
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

    def _calculate_client_values(self):
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _update_client(self):
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        # Each client is assigned to the closest currently active facility.
        min_indices = torch.argmin(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices, min_indices] = 1

    def _calculate_facility_values(self):
        C = self._client_activation_values.to(self._distance_values.dtype)    # (n,k)
        self._facility_inner_values = self._distance_values.t() @ C           # (n,n)^T @ (n,k) -> (n,k)

        # Opening cost is a penalty in the k-facility objective, so it is added
        # to the assignment distance score for each candidate facility.
        self._facility_inner_values += self._normalized_cost_values.unsqueeze(1)

    def _update_facility(self):
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        chosen = []
        used_facilities = set()

        for q in range(self._k):
            costs = self._facility_inner_values[:, q].clone()  # (n,)

            # Prevent duplicate facilities across clusters.
            # Each cluster q should choose a different facility, otherwise the solution
            # may contain fewer than k unique open facilities.
            for used in used_facilities:
                costs[used] = float("inf")

            j = torch.argmin(costs).item()
            chosen.append(j)
            used_facilities.add(j)
            self._facility_activation_values[j, q] = 1

        self._active_facility_list = chosen

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._active_facility_list)
        selected_distance = calculate_distance_with_facility_cost(self._graph, selected_facilities, self._costs, self._n)
        return selected_facilities, selected_distance
