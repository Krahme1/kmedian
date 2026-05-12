import random

import torch

from problems.CKMProblem import CKMProblem
from solvers.brute_solver import calculate_capacitated_distance
from solvers_alg.CKMP.CKMPSolver import CKMPSolver


class HopfieldParallelCKMSolver(CKMPSolver):
    def __init__(self, use_gpu, max_iterations=300):
        self._name = "Hopfield Parallel Solver - Capacitated K-median"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._k = None
        self._graph = None
        self._capacity = None

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
        self._client_update_order = None
        self._max_iterations = max_iterations

    def initialize(self, problem:CKMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        self._capacity = problem.getCapacity()
        if self._graph is None or self._n is None or self._k is None or self._capacity is None:
            raise ValueError("Graph, n, k, and capacity must be set before calling initialize().")
        
        self._num_rows = self._n
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)

        self._math_row_indices = torch.arange(0, self._n, device=self._device)
        self._k_indices = torch.arange(0, self._k, device=self._device)

        # Initialize distance values
        if self._use_gpu:
            self._distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            self._distance_values = (1 - self._graph._normalized_distances).clone().detach()

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

    def setCapacity(self, capacity):
        self._capacity = capacity

    def solve(self, runNum=None, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = calculate_capacitated_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        self._initialize_per_run_arrays(starter_facilities)

        stabilized = False
        iterations = 0

        while not stabilized and iterations < self._max_iterations:
            prev_C = self._client_activation_values.clone()
            prev_F = self._facility_activation_values.clone()

            # Client update
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

        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()


    def _initialize_per_run_arrays(self, starter_facilities):
        self._client_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)
        self._facility_activation_values = torch.zeros(self._size, dtype=torch.int, device=self._device)

        self._active_facility_list = []
        # Use a fixed client update order per run to avoid oscillations.
        self._client_update_order = torch.randperm(self._n, device=self._device)

        # randomly pick k vertices as the starting facilities
        index = 0
        for value in random.sample([i for i in range(0, self._n)], k=self._k):
            self._facility_activation_values[value, index] = 1
            self._active_facility_list.append(value)
            index = index + 1

        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

    def _calculate_client_values(self):
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _update_client(self):
        # reset activations in-place to avoid reallocating every iteration
        self._client_activation_values.zero_()

        # track how many clients are assigned to each cluster
        cluster_loads = torch.zeros(self._k, dtype=torch.long, device=self._device)
        capacity = int(self._capacity)
        available_clusters = torch.ones(self._k, dtype=torch.bool, device=self._device)
        num_available_clusters = self._k
        neg_inf = torch.finfo(self._client_inner_values.dtype).min

        if self._client_update_order is None or self._client_update_order.numel() != self._n:
            client_order = torch.arange(self._n, device=self._device)
        else:
            client_order = self._client_update_order

        # Convert once to avoid per-iteration tensor->python sync inside the loop.
        client_order_list = client_order.tolist()

        # Assign clients one-by-one in order, choosing best currently available cluster.
        for i in client_order_list:
            if num_available_clusters > 0:
                scores = self._client_inner_values[i].masked_fill(~available_clusters, neg_inf)
                q = int(torch.argmax(scores).item())
            else:
                # Infeasible instance (k * capacity < n): assign to best cluster anyway.
                q = int(torch.argmax(self._client_inner_values[i]).item())

            self._client_activation_values[i, q] = 1
            cluster_loads[q] += 1
            if available_clusters[q] and cluster_loads[q] >= capacity:
                available_clusters[q] = False
                num_available_clusters -= 1
                
    def _calculate_facility_values(self):
        C = self._client_activation_values.to(self._distance_values.dtype)    # (n,k)
        self._facility_inner_values = self._distance_values.t() @ C           # (n,n)^T @ (n,k) -> (n,k)

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
        selected_distance = calculate_capacitated_distance(self._graph, self._client_activation_values, self._active_facility_list,)
        # print("Length:", len(self._active_facility_list), "Distance:", selected_distance)
        # print("Selected facilities:", selected_facilities)
        return selected_facilities, selected_distance
