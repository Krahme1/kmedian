import random

import numpy as np
import torch
from problems.CKMProblem import CKMProblem
from solvers.brute_solver import calculate_distance

from solvers_alg.CKMP.CKMPSolver import CKMPSolver

# Constants for update type.
FACILITY = 1
CLIENT = 0

class HopfieldOriginal2nkCKMPSolver(CKMPSolver):
    def __init__(self, use_gpu):
        self._name = "Hopfield (Original2nk) - Capacitated K-Median Problem"
        self._solutionValue = 0
        self._selectedFacilities = []

        self.verbose = False
        self._n = None
        self._k = None
        self._graph = None
        self._capacity = None
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
        
        #self.facility_col = torch.zeros(self._n, device=self._device)
        #self.D_cols = torch.zeros((self._n, self._n - k + 1), device=self._device)
        #self.D_rr = torch.zeros((self._n - k + 1, self._n - k + 1), device=self._device)

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

    def setCapacity(self, capacity):
        self._capacity = capacity

    def solve(self, runNum=None, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        completed = 0
        import time
        start_time = time.time()
        max_time = 235

        current_time = time.time()

        self._initialize_per_run_arrays()
        facility_stabilized = False

        while not facility_stabilized:
            
            max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
            sumBefore = torch.sum(max_values).item()                
        
            # Determine which of the facilities has the lowest inner value and deactivate it
            self._sorted_facility_inner_values, self._sorted_facility_indices = torch.topk(max_values, self._k)
            worstFacility = self._sorted_facility_indices[self._k-1]
            
            self._facility_activation_values[worstFacility, max_indices[worstFacility]] = 0
            self._facilities[0,worstFacility] = 0 #This variable is useful in some matrix computations, and it tracks active facilities in a 1D array               
            
            #Set all the client inner values corresponding to the worst facility to 0
            #This is needed in order to find an accurate value for which clients are assigned to the k-1 remaining active facilities
            self._client_inner_values[:, max_indices[worstFacility]] = 0
        
            #Calculate the value for each client being served by their closest k-1 remaining active facility
            client_max_values, client_max_indices = torch.max(self._client_inner_values, dim=1)    
            #"""                
            facility_values = torch.reshape(self._facilities, (self._n, 1))
            facility_values = (facility_values - 1) * -1
            self._candidatefacility_inner_values = facility_values * self._distance_values
            facility_values = torch.reshape(facility_values, (1, self._n))
            self._candidatefacility_inner_values = facility_values * self._candidatefacility_inner_values          
            
            #self._candidatefacility_inner_values = torch.where(self._candidatefacility_inner_values > client_max_values, self._candidatefacility_inner_values - client_max_values, 0) 
            self._candidatefacility_inner_values = (self._candidatefacility_inner_values - client_max_values).clamp_min_(0) # This is faster
            #"""
            

            # Now we need to take the sum of each row from candidate inner value, and put it in correct spot for facility inner value
            # max_indices[worstFacility] is the cluster we’re updating
            self._facility_inner_values[:, max_indices[worstFacility]] = torch.sum(self._candidatefacility_inner_values, dim=1)
            
            # Sort the facilities again
            bestFacility = torch.argmax(self._facility_inner_values[:, max_indices[worstFacility]])    
            #best_local = torch.argmax(s_reduced)         # index in [0 .. (n-k)-1]
            #bestFacility = inactive_idx[best_local].item()
            self._active_facility_list[max_indices[worstFacility]] = bestFacility.item()            
            self._facility_activation_values[bestFacility, max_indices[worstFacility]] = 1

            self._calculate_client_values()
            self._update_client()
            self._calculate_facility_values()
            
            max_values_after, max_indices_after = torch.max(self._facility_inner_values, dim=1)
            sumAfter = torch.sum(max_values_after).item()
            self._facilities[0,bestFacility] = 1
            
            if sumBefore >= sumAfter:
                facility_stabilized = True
                self._facility_activation_values[bestFacility.item(),max_indices[worstFacility]] = 0
                #self._facility_activation_values[bestFacility,max_indices[worstFacility]] = 0
                self._facility_activation_values[worstFacility.item(),max_indices[worstFacility]] = 1
                self._facilities[0,bestFacility] = 0
                self._facilities[0,worstFacility] = 1
                self._active_facility_list[max_indices[worstFacility]] = worstFacility.item()
                
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

    def _initialize_per_run_arrays(self):
    
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        #self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facilities = torch.zeros(size=(1,self._num_rows), dtype=torch.int, device=self._device)
        self._active_facility_list = []
        
        # randomly pick k vertices as the starting facilities
        index = 0
        for value in random.sample([i for i in range(0, self._n)], k=self._k):
            self._facility_activation_values[value, index] = 1
            self._facilities[0,value] = 1
            self._active_facility_list.append(value)
            index = index + 1
        
        """
        if self.verbose is True:
            print("These are the initial random activation values")
            print(self._facility_activation_values)
            print("")
        """
        
        self._calculate_client_values()
        
        """
        if self.verbose is True:
            print("These are the initial client inner values")
            print(self._client_inner_values)
            print()
        """
        
        self._update_client()
        
        """
        if self.verbose is True:
            print("These are the inital client activation values")
            print(self._client_activation_values)
            print()
        """

        self._calculate_facility_values()
        
        """
        if self.verbose is True:
            print("These are the initial facility inner values")
            print(self._facility_inner_values)
            print()
        """

    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []
        
        for i in range(self._n):
        
            if self._facilities[0,i] == 1:
            
                selected_facilities.append(i)
                
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        
        return selected_facilities, selected_distance

    def _create_distance_array(self):
    
        return torch.tensor(self._graph.distance_cache, device=self._device)

    def _calculate_client_values(self):
    
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _calculate_facility_values(self):
    
        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_inner_values[self._active_facility_list,self._k_indices] = torch.sum(self._client_inner_values * self._client_activation_values, dim=0)

    def _update_client(self):

        # Reset client activations
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)

        # Track how many clients are assigned to each cluster
        cluster_loads = torch.zeros(self._k, device=self._device)

        # Randomize client update order to avoid bias
        client_order = torch.randperm(self._n, device=self._device)

        for i in client_order:

            # Clusters sorted by preference (largest inner value first)
            prefs = torch.argsort(self._client_inner_values[i], descending=True)

            assigned = False

            # Assign to best feasible cluster
            for q in prefs:
                if cluster_loads[q] < self._capacity:
                    self._client_activation_values[i, q] = 1
                    cluster_loads[q] += 1
                    assigned = True
                    break

            # Safety fallback (should almost never trigger)
            if not assigned:
                q = torch.argmin(cluster_loads)
                self._client_activation_values[i, q] = 1
                cluster_loads[q] += 1
