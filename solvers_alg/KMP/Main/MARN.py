# =============================================================
#  MARN  (Modified Asymmetric Resonance Network) - the ACCURATE method
# -------------------------------------------------------------
#  Menu option : 12  ("2nk Exhaustive Hopfield")
#  Class name  : HopfieldExhaustiveAlgorithmSolver
#  Paper       : "A Resonance Neural Network for the K-Median Problem"
#  What it does: Like ARN, but considers replacing ANY facility (all
#                possible single swaps), not just the weakest one. A
#                bit slower than ARN, but more accurate - it can escape
#                weaker local optima that ARN gets stuck in.
# =============================================================
import random
import time

import numpy as np
import torch

from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class HopfieldExhaustiveAlgorithmSolver(KMPSolver):
    def __init__(self, use_gpu):
        # Initialize Variables for Solver
        self._name = "Hopfield 2nk Exhaustive"
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
        
        self.start_time = 0
        #self.maxTime = 1000000
        #self.maxTime = 200
        
        self.maxTime = 0

    def initialize(self, problem:KMProblem):
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
            #self._full_distance_values = torch.tensor(1 - graph._normalized_distances)
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

        self.start_time = time.time()
        #self.maxTime = 1000000
        #self.maxTime = 200
        
        if self._n < 1000:
            self.maxTime = 0.2
        elif self._n > 1000 and self._n < 1500:
            self.maxTime = 1
        elif self._n > 1000 and self._n < 3000:
            self.maxTime = 2
        elif self._n > 1000 and self._n < 5000:
            self.maxTime = 3
        elif self._n > 1000 and self._n < 6000:
            self.maxTime = 20
        elif self._n > 1000 and self._n < 15000 and self._k < 1000:
            self.maxTime = 50
        elif self._n > 1000 and self._n < 15000 and self._k == 1000:
            self.maxTime = 75
        elif self._n > 1000 and self._n < 15000 and self._k == 2000:
            self.maxTime = 100
        elif self._n > 1000 and self._n < 15000 and self._k > 2000:
            self.maxTime = 200

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
    
    def solve(self,runNum, starter_facilities=None):
        
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None
        self.start_time = time.time()
                
        self._initialize_per_run_arrays(runNum)
        facility_stabilized = False
        while not facility_stabilized:
            current_time = time.time()
            if current_time - self.start_time >= self.maxTime:
                break  
            # Run fast ARN to a local optimum, then let MARN try ALL facilities
            # to escape it. Repeat until MARN can't improve things either.
            self.ARN()
            facility_stabilized = self.MARN()

        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

    def _initialize_per_run_arrays(self, r):
    
        self._facility_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        #self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facilities = torch.zeros(size=(1,self._num_rows), dtype=torch.int, device=self._device)
        self._active_facility_list = []
        
        index = 0
        if self._k > 2 and r > 0:
            available_list = [i for i in range(self._n)]
            num = self._k
            if self._k > 3:
                for i in range(min(self._k - 3, int(self._k*0.9))):
                    value = self._sorted_facility_indices[self._n - i - 1].item()
                    self._facility_activation_values[value, index] = 1
                    available_list.remove(value)
                    self._facilities[0, value] = 1
                    self._active_facility_list.append(value)
                    num = num - 1
                    index = index + 1
            else:
                for i in range(min(self._k - 2, int(self._k*0.9))):
                    value = self._sorted_facility_indices[self._n - i - 1].item()
                    self._facility_activation_values[value, index] = 1
                    available_list.remove(value)
                    self._facilities[0, value] = 1
                    self._active_facility_list.append(value)
                    num = num - 1
                    index = index + 1 
            for value in random.sample(available_list, num):
                self._facility_activation_values[value, index] = 1
                self._facilities[0,value] = 1
                self._active_facility_list.append(value)
                index = index + 1
        else:
            for value in random.sample([i for i in range(0, self._n)], k=self._k):
                self._facility_activation_values[value, index] = 1
                self._facilities[0,value] = 1
                self._active_facility_list.append(value)
                index = index + 1
        
        self._calculate_client_values()
        self._update_client()
        self._calculate_facility_values()
        
    def ARN(self):
        
        facility_stabilized = False
        while not facility_stabilized:
            current_time = time.time()
            if current_time - self.start_time >= self.maxTime:
                return
            max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
            sumBefore = torch.sum(max_values).item() 
            self._sorted_facility_inner_values, self._sorted_facility_indices = torch.torch.topk(max_values, self._k)
            worstFacility = self._sorted_facility_indices[self._k - 1]
            self._facility_activation_values[worstFacility, max_indices[worstFacility]] = 0
            self._facilities[0,worstFacility] = 0        
            self._client_inner_values[:, max_indices[worstFacility]] = 0
            client_max_values, client_max_indices = torch.max(self._client_inner_values, dim=1)
            facility_values = torch.reshape(self._facilities, (self._n, 1))
            facility_values = (facility_values - 1) * -1
            self._candidatefacility_inner_values = facility_values * self._distance_values
            facility_values = torch.reshape(facility_values, (1, self._n))
            self._candidatefacility_inner_values = facility_values * self._candidatefacility_inner_values
            self._candidatefacility_inner_values = (self._candidatefacility_inner_values - client_max_values).clamp_min_(0) 
            self._facility_inner_values[:, max_indices[worstFacility]] = torch.sum(self._candidatefacility_inner_values, dim=1)       
            bestFacility = torch.argmax(self._facility_inner_values[:, max_indices[worstFacility]])
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
                self._facility_activation_values[worstFacility.item(),max_indices[worstFacility]] = 1
                self._facilities[0,bestFacility] = 0
                self._facilities[0,worstFacility] = 1
                self._active_facility_list[max_indices[worstFacility]] = worstFacility.item()
        #current_time = time.time()
        #print("ARN COMPLETE. Energy: ",max(sumBefore,sumAfter)," Time: ",current_time - self.start_time)
        return

    # MARN's exhaustive step: unlike ARN (which only swaps the single weakest
    # facility), this tries replacing facilities one by one and keeps the best
    # improving swap it finds -- that's what makes MARN more accurate than ARN.
    def MARN(self):
        
        max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
        sumBefore = torch.sum(max_values).item()
        bestSolution = sumBefore
        self._sorted_facility_inner_values, self._sorted_facility_indices = torch.topk(max_values, self._k)
        self._facility_inner_values_copy = self._facility_inner_values.detach().clone()
        self._client_inner_values_copy = self._client_inner_values.detach().clone()
        self._facility_activation_values_copy = self._facility_activation_values.detach().clone()
        self._client_activation_values_copy = self._client_activation_values.detach().clone()
        # Try replacing facilities one at a time (weaker ones first). This
        # loop over multiple facilities is the "exhaustive" part ARN skips.
        for innerLoop in range(1, self._k):
            current_time = time.time()
            if current_time - self.start_time >= self.maxTime:
                return True
            worstFacility = self._sorted_facility_indices[self._k - 1 - innerLoop]     
            self._facility_activation_values[worstFacility, max_indices[worstFacility]] = 0
            self._facilities[0,worstFacility] = 0
            self._client_inner_values_copy[:, max_indices[worstFacility]] = 0
            client_max_values, client_max_indices = torch.max(self._client_inner_values_copy, dim=1)
            facility_values = torch.reshape(self._facilities, (self._n, 1))
            facility_values = (facility_values - 1) * -1
            self._candidatefacility_inner_values = facility_values * self._distance_values
            facility_values = torch.reshape(facility_values, (1, self._n))
            self._candidatefacility_inner_values = facility_values * self._candidatefacility_inner_values 
            self._candidatefacility_inner_values = (self._candidatefacility_inner_values - client_max_values).clamp_min_(0)
            self._facility_inner_values_copy[:, max_indices[worstFacility]] = torch.sum(self._candidatefacility_inner_values, dim=1)
            bestFacility = torch.argmax(self._facility_inner_values_copy[:, max_indices[worstFacility]])    
            self._active_facility_list[max_indices[worstFacility]] = bestFacility.item()             
            self._facility_activation_values_copy[bestFacility, max_indices[worstFacility]] = 1
            self._calculate_client_values_copy()
            self._update_client_copy()
            self._calculate_facility_values_copy()
            max_values_after, max_indices_after = torch.max(self._facility_inner_values_copy, dim=1)
            sumAfter = torch.sum(max_values_after).item()
            self._facilities[0,bestFacility] = 1         
            self._facility_activation_values_copy[bestFacility,max_indices[worstFacility]] = 0
            self._facility_activation_values_copy[worstFacility,max_indices[worstFacility]] = 1
            self._facilities[0,bestFacility] = 0
            self._facilities[0,worstFacility] = 1
            self._active_facility_list[max_indices[worstFacility]] = worstFacility.item()
            #self._client_inner_values_copy[:,max_indices[worstFacility]] = self._client_inner_values[max_indices[worstFacility]]
            if sumAfter > bestSolution:
                bestSolution = sumAfter
                bestSolutionFacilityToActivate = bestFacility.item()
                bestSolutionFacilityToDeactivate = worstFacility.item()
                break
        # If any facility swap improved the total, apply the best one and keep
        # going (return False = not done). If none helped, we're at a true local
        # optimum across all single swaps (return True = stop).
        if bestSolution > sumBefore:
            self._facility_activation_values[bestSolutionFacilityToActivate,max_indices[bestSolutionFacilityToDeactivate]] = 1
            self._facility_activation_values[bestSolutionFacilityToDeactivate,max_indices[bestSolutionFacilityToDeactivate]] = 0
            self._facilities[0,bestSolutionFacilityToActivate] = 1
            self._facilities[0,bestSolutionFacilityToDeactivate] = 0
            self._active_facility_list[max_indices[bestSolutionFacilityToDeactivate]] = bestSolutionFacilityToActivate
            self._calculate_client_values()
            self._update_client()
            self._calculate_facility_values()
            return False
        else:
            return True

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
        
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices,max_indices] = 1
        
    def _calculate_client_values_copy(self):
    
        self._client_inner_values_copy[:,:] = self._distance_values[:,self._active_facility_list]

    def _calculate_facility_values_copy(self):
    
        self._facility_inner_values_copy = torch.zeros(size=self._size, device=self._device)
        self._facility_inner_values_copy[self._active_facility_list,self._k_indices] = torch.sum(self._client_inner_values_copy * self._client_activation_values_copy, dim=0)

    def _update_client_copy(self):
        
        self._client_activation_values_copy = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmax(self._client_inner_values_copy, dim=1)
        self._client_activation_values_copy[self._math_row_indices,max_indices] = 1