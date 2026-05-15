import random

import numpy as np
import torch

from problems.KCProblem import KCProblem
from solvers_alg.solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver

# Constants for update type.
FACILITY = 1
CLIENT = 0

class HopfieldOriginal2nkSolverKCenter(KCPSolver):
    def __init__(self, use_gpu):
        # Initialize Variables for Solver
        self._name = "Hopfield (original 2nk) - K-Center Problem"
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
        
        #self.facility_col = torch.zeros(self._n, device=self._device)
        #self.D_cols = torch.zeros((self._n, self._n - k + 1), device=self._device)
        #self.D_rr = torch.zeros((self._n - k + 1, self._n - k + 1), device=self._device)

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
        best_distance = calculate_radius(self._graph, best_facilities) if starter_facilities else None

        completed = 0
        import time
        start_time = time.time()
        max_time = 235
                
        current_time = time.time()
        
        #if current_time - start_time >= max_time:
        #    break
            
        # Initialize our per run variables.
        self._initialize_per_run_arrays()
        facility_stabilized = False

        iterations = 0
        
        while not facility_stabilized:
            
            max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
            radiusBefore = torch.min(max_values).item()
        
            # Determine which of the facilities has the lowest inner value and deactivate it
            #self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(max_values)
            #worstFacility = self._sorted_facility_indices[self._n - self._k]
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
            
            """
            # Is this faster?
            inactive_mask = (self._facilities == 0).view(-1)                                    # shape (n,)
            inactive_idx = torch.nonzero(inactive_mask, as_tuple=True)[0]                       # (n-k,)
            D_cols = self._distance_values.index_select(dim=1, index=inactive_idx)
            D_rr = D_cols.index_select(dim=0, index=inactive_idx)                          # reduced (n-k) x (n-k)
            client_max_reduced = client_max_values.index_select(0, inactive_idx).unsqueeze(0)   # (n-k) x 1
            C = (D_rr - client_max_reduced).clamp_min_(0)                                       # (n-k) x (n-k)
            s_reduced = torch.sum(C, dim=1)   
            self.facility_col.zero_()
            self.facility_col.index_copy_(0, inactive_idx, s_reduced)                                # (n-k,)
            """

            self._facility_inner_values[:, max_indices[worstFacility]] = torch.max(self._candidatefacility_inner_values, dim=1).values
            
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
            radiusAfter = torch.min(max_values_after).item()
            self._facilities[0,bestFacility] = 1
            
            if radiusBefore <= radiusAfter:
                facility_stabilized = True
                self._facility_activation_values[bestFacility.item(),max_indices[worstFacility]] = 0
                #self._facility_activation_values[bestFacility,max_indices[worstFacility]] = 0
                self._facility_activation_values[worstFacility.item(),max_indices[worstFacility]] = 1
                self._facilities[0,bestFacility] = 0
                self._facilities[0,worstFacility] = 1
                self._active_facility_list[max_indices[worstFacility]] = worstFacility.item()

            iterations += 1
                
        #selected_facilities, selected_distance = self._calculate_facilities_and_distance()

        # update our best value
        #if best_distance is None or selected_distance < best_distance:
        #    if best_distance is None:
        #        pass
        #    else:
        #        new_ratio = round(best_distance / selected_distance, 3)
                
        #    best_distance = selected_distance
        #    best_facilities = selected_facilities
            
    #print("Best facilities: ",best_facilities)
    #print("Active facilities: ",self._active_facility_list)
    #return best_facilities
    #current_time = time.time()
    #print("Energy: ",max(sumBefore,sumAfter)," Time: ",current_time - start_time)
        print(f"Stabilized after {iterations} iterations.")
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
        
        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()

    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []

        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected_facilities.append(i)

        radius = calculate_radius(self._graph, selected_facilities)

        return selected_facilities, radius

    def _create_distance_array(self):
    
        return torch.tensor(self._graph.distance_cache, device=self._device)

    def _calculate_client_values(self):
    
        self._client_inner_values[:,:] = self._distance_values[:,self._active_facility_list]

    def _calculate_facility_values(self):
    
        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)

        # The min fucntion could assume the inactive clients are what we are looking for, so mask them before looking for the minimum
        masked = self._client_inner_values.clone()
        masked[self._client_activation_values == 0] = float('inf')
        self._facility_inner_values[self._active_facility_list,self._k_indices] = torch.max(masked, dim=0).values

    def _update_client(self):
        
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices,max_indices] = 1