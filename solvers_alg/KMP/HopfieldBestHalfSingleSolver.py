import random

import numpy as np
import torch

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class HopfieldBestHalfSingleSolver(KMPSolver):
    def __init__(self, use_gpu):
        # Initialize Variables for Solver
        self._name = "Hopfield 2nk Best Half Single"
        self._solutionValue = 0
        self._selectedFacilities = []

        self.verbose = False
        self._n = None
        self._k = None
        self._graph = None
        self._num_rows = None
        self._num_cols = None
        self._size = None

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

    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getN()
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
        """
        if self.verbose is True:
            print("These are the initial (normalized) distance values")
            print(self._full_distance_values)
            print()
        """
              
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
    
    def solve(self, runNum, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        #import time

        #max_time = 100000
                
        #start_time = time.time()

        # Initialize our per run variables.
        self._initialize_per_run_arrays(runNum)
        
        facility_stabilized = False
        #counter = 0
        
        while not facility_stabilized:
            
            #current_time = time.time()
            #if current_time - start_time >= max_time:
            #    facility_stabilized = True
            #    break
                
            max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
            sumBefore = torch.sum(max_values).item()
            #counter = counter + 1

            
            """
            if self.verbose is True:
                print("Here is the sum of facility inner values before deactivation")
                print(sumBefore)
                print()
            
                print("These are the facility activation values")
                print(self._facility_activation_values)
                print()
                print("These are the client activation values")
                print(self._client_activation_values)
                print()
            """
        
            # Determine which of the facilities has the lowest inner value and deactivate it
            self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(max_values)
            worstFacility = self._sorted_facility_indices[self._n - self._k]
            
            self._facility_activation_values[worstFacility, max_indices[worstFacility]] = 0
            self._facilities[0,worstFacility] = 0 #This variable is useful in some matrix computations, and it tracks active facilities in a 1D array
            
            """
            if self.verbose is True:
                print("Beginning of the loop")
                print("Here is the list of sorted facility inner values")
                print(self._sorted_facility_inner_values)
                print()
            
                print("Here is the list of sorted facility indices")
                print(self._sorted_facility_indices)
                print()
            
                print("The active facility with the worst inner value is: ", worstFacility.item())
                print()
            """
            
            #Set all the client inner values corresponding to the worst facility to 0
            #This is needed in order to find an accurate value for which clients are assigned to the k-1 remaining active facilities
            self._client_inner_values[:, max_indices[worstFacility]] = 0
            
            """
            if self.verbose is True:
                print("These are the client inner values after deactivating the worst facility")
                print(self._client_inner_values)
                print()
            """
        
            #Calculate the value for each client being served by their closest k-1 remaining active facility
            client_max_values, client_max_indices = torch.max(self._client_inner_values, dim=1)
            client_max_values =  client_max_values.unsqueeze(0)
            
            """
            if self.verbose is True:
                print("These are the max values in each column")
                print(client_max_values)
                print()
            """

            facility_values = torch.reshape(self._facilities, (self._n, 1))
            facility_values = (facility_values - 1) * -1
            
            """
            if self.verbose is True:
                print("Facility values")
                print(facility_values)
            """
            
            self._candidatefacility_inner_values = facility_values * self._distance_values
            
            """
            if self.verbose is True:
                print("These are the candidate facility inner values with remaining facility rows zeroed out")
                print(self._candidatefacility_inner_values)
                print()
            """
            
            facility_values = torch.reshape(facility_values, (1, self._n))
            
            """
            if self.verbose is True:
                print("Facility values")
                print(facility_values)
            """
            
            self._candidatefacility_inner_values = facility_values * self._candidatefacility_inner_values 
            
            """
            if self.verbose is True:
                print("These are the candidate facility inner values with remaining facility columns zeroed out")
                print(self._candidatefacility_inner_values)
                print()
            """
                
            self._candidatefacility_inner_values = torch.where(self._candidatefacility_inner_values > client_max_values, self._candidatefacility_inner_values - client_max_values, 0) 
                
            """
            if self.verbose is True:
                print("These are the candidate facility inner values that have been adjusted based on which clients they can actually serve")
                print(self._candidatefacility_inner_values)
                print()
            """
            
            # Now we need to take the sum of each row from candidate inner value, and put it in correct spot for facility inner value
            # max_indices[worstFacility] represents the cluster where we deactivated a facility
            self._facility_inner_values[:, max_indices[worstFacility]] = torch.sum(self._candidatefacility_inner_values, dim=1)
            
            """
            if self.verbose is True:
                print("Updated candidate facility inner values")
                print(self._facility_inner_values)
                print()
            """
            
            # Sort the facilities again
            bestFacility = torch.argmax(self._facility_inner_values[:, max_indices[worstFacility]])    
            self._active_facility_list[max_indices[worstFacility]] = bestFacility              
            self._facility_activation_values[bestFacility, max_indices[worstFacility]] = 1
            
            """
            if self.verbose is True:
                print("Comparing worstFacility to bestFacility")
                print("worstFacility", worstFacility.item())
                print("bestFacility", bestFacility.item())
                print()
            """
            
            self._calculate_client_values()              
            self._update_client()
            self._calculate_facility_values()
            
            max_values_after, max_indices_after = torch.max(self._facility_inner_values, dim=1)
            sumAfter = torch.sum(max_values_after).item()
            self._facilities[0,bestFacility] = 1
            
            """
            if self.verbose is True:
                print("Here is the sum of facility inner values")
                print("Beginning of loop", sumBefore)
                print("End of loop", sumAfter)
                print()
            """
            
            if sumBefore >= sumAfter:
            #if sumAfter > (1 + (1 / self._n)) * sumBefore:
                facility_stabilized = True
                self._facility_activation_values[bestFacility.item(),max_indices[worstFacility]] = 0
                self._facility_activation_values[worstFacility.item(),max_indices[worstFacility]] = 1
                self._facilities[0,bestFacility] = 0
                self._facilities[0,worstFacility] = 1
                self._active_facility_list[max_indices[worstFacility]] = worstFacility        
                
                """
                if self.verbose is True:
                    print("Stabilized")
                """
                
        #print("Number of iterations of the main loop: ", counter)
        selected_facilities, selected_distance = self._calculate_facilities_and_distance()
        
        """
        if self.verbose is True:
            print("Current distance", selected_distance)
            print("Best distance", best_distance)
            print()
        """

        # update our best value
        if best_distance is None or selected_distance < best_distance:
        
            if best_distance is None:
            
                pass
                
            else:
            
                new_ratio = round(best_distance / selected_distance, 3)
                #print(f"{r}: {selected_distance} - {new_ratio}")
                
            best_distance = selected_distance
            best_facilities = selected_facilities
            
            """
            if self.verbose is True:
                print("These are the activation values")
                print(self._activation_values)
                print()   
            """

        #completed += 1
        #print("Iterations: ",counter,", Cost: ",selected_distance)


        #import sys
        #sys.exit(0)
        #print(completed)
        self._selectedFacilities = best_facilities
        self._solutionValue = best_distance

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
        """
        if self.verbose is True:
            print("These are the client inner values")
            print(self._client_inner_values)
            print()
        """

    def _calculate_facility_values(self):
    
        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_inner_values[self._active_facility_list,self._k_indices] = torch.sum(self._client_inner_values * self._client_activation_values, dim=0)
        """
        if self.verbose is True:
            print("These are the active facilities")
            print(self._facility_activation_values)
            print("These are the facility inner values")
            print(self._facility_inner_values)
            print()
        """

    def _update_client(self):
        
        self._client_activation_values = torch.zeros(size=self._size, dtype=torch.int, device=self._device)
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        self._client_activation_values[self._math_row_indices,max_indices] = 1
        """
        if self.verbose is True:
            print("These are the client activation values")
            print(self._client_activation_values)
            print()
        """

