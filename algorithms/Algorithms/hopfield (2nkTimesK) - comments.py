import random
import numpy as np
import torch

from solvers_alg.solvers.brute_solver import calculate_distance

# Constants for update type.
FACILITY = 1
CLIENT = 0

class Hopfield:
    def __init__(self, n, k, graph, use_gpu):
    
        self.verbose = False
        self._n = n
        self._k = k
        self._graph = graph
        self._num_rows = n
        self._num_cols = k
        self._size = (self._num_rows, self._num_cols)
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
          
        # The graph should contain a normalized torch array of distances.
        # Subtract 1 in order to prioritize smaller distances
        if self._use_gpu:
            self._full_distance_values = 1 - graph._gpu_normalized_distances
        else:
            #self._full_distance_values = torch.tensor(1 - graph._normalized_distances)
            self._full_distance_values = (1 - graph._normalized_distances).clone().detach()
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
        self._active_facility_list = []
        
        # Caching the sorted list of a facility inner values in order to decrease the number of sort calls
        self._sorted_facility_inner_values = None
        self._sorted_facility_indices = None

    def run(self, runs, starter_facilities=None):
                   
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        completed = 0
        import time
        start_time = time.time()
        max_time = 235
        
        for r in range(runs):
        
            current_time = time.time()
            
            #if current_time - start_time >= max_time:
            #    break
                
            # Initialize our per run variables.
            self._initialize_per_run_arrays(r, runs)
            
            facility_stabilized = False
            #counter = 0
            
            while not facility_stabilized:
                    
                max_values, max_indices = torch.max(self._facility_inner_values, dim=1)
                sumBefore = torch.sum(max_values).item()
                bestSolution = sumBefore
                self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(max_values)
                
                self._facility_inner_values_copy = self._facility_inner_values.detach().clone()
                self._client_inner_values_copy = self._client_inner_values.detach().clone()
                self._facility_activation_values_copy = self._facility_activation_values.detach().clone()
                self._client_activation_values_copy = self._client_activation_values.detach().clone()
                
                for innerLoop in range(0, self._k):
                #for innerLoop in range(0, 3):
                    
                    #counter = counter + 1
                    #print("InnerLoop:", innerLoop)
                
                    # Determine which of the facilities has the lowest inner value and deactivate it
                    worstFacility = self._sorted_facility_indices[self._n - self._k + innerLoop]
                    
                    self._facility_activation_values[worstFacility, max_indices[worstFacility]] = 0
                    self._facilities[0,worstFacility] = 0 #This variable is useful in some matrix computations, and it tracks active facilities in a 1D array
                    
                    #Set all the client inner values corresponding to the worst facility to 0
                    #This is needed in order to find an accurate value for which clients are assigned to the k-1 remaining active facilities
                    self._client_inner_values_copy[:, max_indices[worstFacility]] = 0
                
                    #Calculate the value for each client being served by their closest k-1 remaining active facility
                    client_max_values, client_max_indices = torch.max(self._client_inner_values_copy, dim=1)
                    
                    facility_values = torch.reshape(self._facilities, (self._n, 1))
                    facility_values = (facility_values - 1) * -1
                    self._candidatefacility_inner_values = facility_values * self._distance_values
                    facility_values = torch.reshape(facility_values, (1, self._n))
                    self._candidatefacility_inner_values = facility_values * self._candidatefacility_inner_values 
                    self._candidatefacility_inner_values = torch.where(self._candidatefacility_inner_values > client_max_values, self._candidatefacility_inner_values - client_max_values, 0)  
                    
                    # Now we need to take the sum of each row from candidate inner value, and put it in correct spot for facility inner value
                    # max_indices[worstFacility] represents the cluster where we deactivated a facility
                    self._facility_inner_values_copy[:, max_indices[worstFacility]] = torch.sum(self._candidatefacility_inner_values, dim=1)
                    
                    # Sort the facilities again
                    bestFacility = torch.argmax(self._facility_inner_values_copy[:, max_indices[worstFacility]])    
                    self._active_facility_list[max_indices[worstFacility]] = bestFacility              
                    self._facility_activation_values_copy[bestFacility, max_indices[worstFacility]] = 1

                    self._calculate_client_values_copy()
                    self._update_client_copy()
                    self._calculate_facility_values_copy()
                    
                    max_values_after, max_indices_after = torch.max(self._facility_inner_values_copy, dim=1)
                    sumAfter = torch.sum(max_values_after).item()
                    self._facilities[0,bestFacility] = 1                  
                        
                    #Regardless of whether we found a best solution or not, revert to the starting point
                    self._facility_activation_values_copy[bestFacility,max_indices[worstFacility]] = 0
                    self._facility_activation_values_copy[worstFacility,max_indices[worstFacility]] = 1
                    self._facilities[0,bestFacility] = 0
                    self._facilities[0,worstFacility] = 1
                    self._active_facility_list[max_indices[worstFacility]] = worstFacility
                    self._client_inner_values_copy[max_indices[worstFacility]] = self._client_inner_values[max_indices[worstFacility]]
                    
                    #if sumAfter > (1 + (1 / self._n)) * bestSolution:
                    if sumAfter > bestSolution:
                        bestSolution = sumAfter
                        bestSolutionFacilityToActivate = bestFacility
                        bestSolutionFacilityToDeactivate = worstFacility
                        break
                    
                #If we found an improvement, apply it
                if bestSolution > sumBefore:
                    self._facility_activation_values[bestSolutionFacilityToActivate,max_indices[bestSolutionFacilityToDeactivate]] = 1
                    self._facility_activation_values[bestSolutionFacilityToDeactivate,max_indices[bestSolutionFacilityToDeactivate]] = 0
                    self._facilities[0,bestSolutionFacilityToActivate] = 1
                    self._facilities[0,bestSolutionFacilityToDeactivate] = 0
                    self._active_facility_list[max_indices[bestSolutionFacilityToDeactivate]] = bestSolutionFacilityToActivate
                    self._calculate_client_values()
                    self._update_client()
                    self._calculate_facility_values()
                else:
                    facility_stabilized = True      
                        
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

            completed += 1

        #import sys
        #sys.exit(0)
        #print(completed)
        return best_facilities

    def _initialize_per_run_arrays(self, r, runs):
    
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