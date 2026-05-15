import random

import torch

from problems.KCProblem import KCProblem
from solvers_alg.solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class HopfieldParallelKCPSolver(KCPSolver):
    def __init__(self, use_gpu):
        # Initialize Variables for Solver
        self._name = "Hopfield Parallel Solver - K-Center Problem"
        self._solutionValue = 0
        self._selectedFacilities = []

        self.verbose = False
        self._n = None
        self._k = None
        self._graph = None
        self._num_rows = None
        self._num_cols = None
        self._size = None
        self._facility_update_value = 1.0
        self._client_update_value = 1.0

        # CPU/GPU toggle
        self._use_gpu = use_gpu

        # If we select the GPU and cuda is not available, fail loudly.
        if use_gpu:
            self._device = 'cuda' if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = 'cpu' 
        
        self._full_distance_values = None
        self._distance_values = None
        
        # These are the two sets of 2nk neurons
        self._facility_inner_values = None
        self._client_inner_values = None
        self._facility_activation_values = None
        self._client_activation_values = None
        
        # This is required for evaluating the n-(k+1) candidate facilities
        self._candidatefacility_inner_values = None
        
        # These are 1D arrays that have been convenient so far (but not sure if needed)
        self._math_row_indices = None
        self._k_indices = None
        self._facilities = None
        self._active_facility_list = []
        
        # Caching the sorted list of a facility inner values in order to decrease the number of sort calls
        self._sorted_facility_inner_values = None
        self._sorted_facility_indices = None
        self._sorted_facility_indices = None

    def initialize(self, problem:KCProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        self._num_rows = self._n
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)

        # The graph should contain a normalized torch array of distances.
        # Subtract 1 in order to prioritize smaller distances
        if self._use_gpu:
            self._full_distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            self._full_distance_values = (1 - self._graph._normalized_distances).clone().detach()
        
        self._distance_values = self._full_distance_values

        # These are the two sets of 2nk neurons
        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._client_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        
        # This is required for evaluating the n-(k+1) candidate facilities
        self._candidatefacility_inner_values = torch.zeros(size=(self._n, self._n), device=self._device)
        
        # These are 1D arrays that have been convenient so far (but not sure if needed)
        self._math_row_indices = torch.arange(start=0, end=self._n, device=self._device)
        self._k_indices = torch.arange(start=0, end=self._k, device=self._device)
        self._facilities = torch.zeros(size=(1,self._num_rows), dtype=torch.int, device=self._device)

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
        best_facilities = starter_facilities
        best_radius = calculate_radius(self._graph, best_facilities, self._n) if starter_facilities else None

        self._initialize_per_run_arrays(starter_facilities)

        stabilized = False
        iterations = 0

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

        print(f"Stabilized after {iterations} iterations.")
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

    def _initialize_per_run_arrays(self, starter_facilities):
    
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        #self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facilities = torch.zeros(size=(1,self._num_rows), dtype=torch.int, device=self._device)
        self._active_facility_list = []

        if starter_facilities is not None:
            if len(starter_facilities) != self._k:
                raise ValueError("starter_facilities must have length k.")
            initial_facilities = list(starter_facilities)
        else:
            # initial_facilities = random.sample([i for i in range(0, self._n)], k=self._k)
            initial_facilities = self._farthest_first_initial_facilities()

        
        # randomly pick k vertices as the starting facilities
        index = 0
        for value in initial_facilities:
            self._facility_activation_values[value, index] = 1
            self._facilities[0,value] = 1
            self._active_facility_list.append(value)
            index = index + 1

        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

    def _farthest_first_initial_facilities(self):
        # self._distance_values is CLOSENESS in [0,1] (bigger = closer)
        closeness = self._distance_values  # (n,n)

        # First center: "most central" in distance-space == highest average closeness
        first_center = torch.argmax(closeness.sum(dim=1)).item()
        selected = [first_center]

        selected_mask = torch.zeros(self._n, dtype=torch.bool, device=self._device)
        selected_mask[first_center] = True

        # For each node i: closeness to its nearest selected center (nearest = max closeness)
        max_close_to_selected = closeness[:, first_center].clone()

        for _ in range(1, self._k):
            # Farthest = least close to selected set => minimize max_close_to_selected
            scores = max_close_to_selected.masked_fill(selected_mask, float("inf"))
            next_center = torch.argmin(scores).item()

            selected.append(next_center)
            selected_mask[next_center] = True

            # Update nearest-selected closeness for each node
            max_close_to_selected = torch.maximum(max_close_to_selected, closeness[:, next_center])

        return selected

    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []

        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected_facilities.append(i)

        radius = calculate_radius(self._graph, selected_facilities)

        return selected_facilities, radius
    
    def _calculate_client_values(self):
    
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _update_client(self):
        
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices,max_indices] = 1

    def _calculate_facility_values(self):

        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)

        D = self._distance_values

        # client assignments C: (n,k) int {0,1}
        C = self._client_activation_values.bool()

        for q in range(self._k):
            mask = C[:, q]  # which clients are in cluster q
            # distances from clients in cluster q to every candidate center j: (m,n)
            sub = D[mask, :]  # m x n

            # PF[:,q] = min over clients (worst client closeness) -> (n,)
            self._facility_inner_values[:, q] = sub.min(dim=0).values

    def _update_facility(self):

        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        self._facilities = torch.zeros(size=(1, self._num_rows), dtype=torch.int, device=self._device)

        chosen = torch.zeros(self._n, dtype=torch.bool, device=self._device)

        # Process clusters in fixed order: 0,1,...,k-1
        for q in range(self._k):
            scores = self._facility_inner_values[:, q].clone()

            # Disallow centers already chosen by earlier clusters
            scores[chosen] = -float("inf")

            j = torch.argmax(scores).item()

            self._facility_activation_values[j, q] = 1
            self._facilities[0, j] = 1
            chosen[j] = True

        # Keep active list aligned with cluster columns
        centers_by_cluster = torch.argmax(self._facility_activation_values, dim=0)  # (k,)
        self._active_facility_list = centers_by_cluster.tolist()