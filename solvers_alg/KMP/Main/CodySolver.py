# =============================================================
#  Cody Rossiter's Hopfield network for k-median
#  Menu option : 1  ("Modified Hopfield Algorithm")
#  Class name  : HopfieldAlgorithmSolver
#  An earlier Hopfield-style network; predates ARN/MARN.
# =============================================================
import random

import numpy as np
import torch

from algorithms.search_tree import SearchAgent
from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver

# Constants for update type.
FACILITY = 1
CLIENT = 0

class HopfieldAlgorithmSolver(KMPSolver):
    def __init__(self, use_gpu, search_tree_config=None):
        # Initialize Variables for Solver
        self._name = "Hopfield"
        self._solutionValue = 0
        self._selectedFacilities = []

        # Variables for Hopfield Algorithm
        self._n = None
        self._k = None
        self._graph = None
        self._search_tree_config = search_tree_config

        self._num_rows = 0
        self._num_cols = 0
        self._size = None
        self._facility_update_value = 1.0
        self._client_update_value = 1.0

        self._facility_length = 0
        self._facility_list = None

        self._neighbour_sets = {}                   # This keeps track of the specific facility neighbour groups
        self._selected_row_cache = {}   

        self._use_gpu = use_gpu                     # CPU/GPU toggle 

        # If we select the GPU and cuda is not available, fail loudly.
        if use_gpu:                                 
            self._device = 'cuda' if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = 'cpu'

        self._full_distance_values = None
        self._distance_values = None

        # Value Tensors/Matrices
        self._full_activation_values = None
        self._activation_values = None
        self._full_inner_values = None
        self._inner_values = None
        
        # Constant Tensors/Matrices
        self._client_addition_values = None
        self._sparse_column_indices = None
        
        # we want the columns from 1 to the end of the array (i.e., only the client values)
        self._math_column_indices = None
        
        # Test tensors to see if making comparisons on the GPU is faster
        self._k_on_gpu = None
        self._n_on_gpu = None
        self._1_on_gpu = None

        # Caching the sorted list of a facility inner values in order to decrease the number of sort calls
        self._sorted_facility_inner_values = None
        self._sorted_facility_indices = None

        self._maxTime = 0

    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        self._num_rows = self._n
        self._num_cols = self._n + 1
        self._size = (self._num_rows, self._num_cols)
        
        self._facility_length = self._n                  # This variable keeps track of how many facilities we are evaluating. 
        self._facility_list = [i for i in range(self._n)] # this variable keeps track of the facilities we are evaluating
        
        # Cache the mapping from a facility to it's selected row group.
        # The graph should contain a normalized torch array of distances.
        # Subtract 1 in order to prioritize smaller distances
        if self._use_gpu:
            self._full_distance_values = 1 - self._graph._gpu_normalized_distances
        else:
            self._full_distance_values = torch.tensor(1 - self._graph._normalized_distances)
        
        # Value Tensors/Matrices
        self._full_activation_values = torch.empty(size=self._size, device=self._device)
        self._full_inner_values = torch.empty(size=self._size, device=self._device)
        
        # Constant Tensors/Matrices
        self._client_addition_values = torch.full(fill_value=self._client_update_value * 2, size=(self._n,), device=self._device)
        self._sparse_column_indices = torch.arange(start=0, end=self._n, device=self._device)
        
        # we want the columns from 1 to the end of the array (i.e., only the client values)
        self._math_column_indices = torch.arange(start=1, end=self._n + 1, device=self._device)
        
        # Test tensors to see if making comparisons on the GPU is faster
        self._k_on_gpu = torch.tensor(self._k, device=self._device)
        self._n_on_gpu = torch.tensor(self._n, device=self._device)
        self._1_on_gpu = torch.tensor(1.0, device=self._device)
        
        if self._n > 6000 and self._k > 2000:
            self._maxTime = 200
        elif self._n > 6000 and self._k <= 2000:
            self._maxTime = 100
        elif self._n < 1000:
            self._maxTime = 5
        elif self._n > 1000 and self._k < 1000:
            self._maxTime = 10
        elif self._n > 1000 and self._k >= 1000:
            self._maxTime = 15
        else:
            self._maxTime = 15

        
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

    def solve(self,runNum=None, starter_facilities=None):
    
        if self._search_tree_config is not None:
            search_agent = SearchAgent(
                self._n,
                self._k,
                self._search_tree_config['epsilon'],
                self._search_tree_config['exclude'],
                self._search_tree_config['fixed_size'])
        else:
            search_agent = SearchAgent(self._n, self._k)

        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        import time
        start_time = time.time()
        max_time = 235

        # ---- single run start ----
        next_facilities, level, search_size, exclude_original = search_agent.get_facility_group()
        self._initialize_per_run_arrays(next_facilities, search_size, exclude_original)

        # sort facilities at the start
        self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(self._inner_values[:, 0])
        self._sorted_facility_inner_values = self._sorted_facility_inner_values[-self._k:]
        self._sorted_facility_indices = self._sorted_facility_indices[-self._k:]

        bandit = ChangeBandit(self._facility_length, 0.05)
        facility_stabilized = False

        update_type = FACILITY

        while not facility_stabilized:
            current_time = time.time()
            if current_time - start_time >= self._maxTime:
                break

            if update_type == FACILITY:
                facility = bandit.get_action()
                has_changed = self._update_facility(facility)

                if has_changed:
                    bandit.update_action(facility, 1)
                    update_type = CLIENT
                    self._calculate_client_values()
                    facility_stabilized = self._is_stabilized_facility()
                else:
                    bandit.update_action(facility, 0)
                    update_type = FACILITY

            else:
                has_changed = self._update_client()

                if has_changed:
                    self._calculate_facility_values()
                    self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(self._inner_values[:, 0])
                    self._sorted_facility_inner_values = self._sorted_facility_inner_values[-self._k:]
                    self._sorted_facility_indices = self._sorted_facility_indices[-self._k:]
                update_type = FACILITY

        selected_facilities, selected_distance = self._calculate_facilities_and_distance()

        # update search agent and best solution
        search_agent.update_facility_group(next_facilities, level, selected_facilities, selected_distance)

        if best_distance is None or selected_distance < best_distance:

            if best_distance is None:
                
                pass
                    
            else:
                
                new_ratio = round(best_distance / selected_distance, 3)
                #print(f"{r}: {selected_distance} - {new_ratio}")

            best_distance = selected_distance
            best_facilities = selected_facilities

        # ---- single run end ----
        self._selectedFacilities = best_facilities
        self._solutionValue = best_distance

    def _initialize_per_run_arrays(self, best_facilities, search_size, exclude_original):
    
        # Quick and dirty arithmetic to get random numbers between .99 and .999.
        self._full_activation_values = (torch.rand(size=self._size, device=self._device) / 100) + 0.99
        torch.diag(self._full_activation_values[:, 1], 0).zero_()
        #print("These are the initial random activation values")
        #print(self._full_activation_values)
        #print("")
        self._full_inner_values = torch.zeros(size=self._size, device=self._device)

        if best_facilities is not None:
        
            full_locations = self._calculate_nearest_neighbours(best_facilities, search_size, exclude_original)

            self._activation_values = self._full_activation_values[full_locations]
            self._inner_values = self._full_inner_values[full_locations]
            self._distance_values = self._full_distance_values[full_locations]
            self._facility_list = list(full_locations)
            self._facility_length = len(full_locations)
            self._selected_row_cache = {}
            
        else:
        
            # use the full values and change nothing else
            self._activation_values = self._full_activation_values
            self._inner_values = self._full_inner_values
            self._distance_values = self._full_distance_values
            self._facility_list = [i for i in range(self._n)]
            self._facility_length = self._n
            self._neighbour_sets = {}

        self._calculate_facility_values()
        #print("These are the initial facility inner values")
        #print(self._inner_values)
        #print()
        self._calculate_client_values()
        #print("These are the initial client inner values")
        #print(self._inner_values)
        #print()

    def _calculate_nearest_neighbours(self, best_facilities, search_size, exclude_original):
    
        # Make the initial dict for holding the neighbour sets.
        neighbour_sets = {}
        
        for facility in best_facilities:
        
            neighbour_sets[facility] = set()

        # Get all potential neighbours.
        all_locations = set()
        
        for facility in best_facilities:
        
            # Remember to sort by descending because the distance array is built using (1 - distance).
            values, indices = torch.sort(self._full_distance_values[facility, :], descending=True)
            
            # add one to account for including the initial facility.
            all_locations = all_locations.union(indices[:search_size + 1].tolist())
            neighbour_sets[facility] = set(indices[:search_size + 1].tolist())

        # convert to a list so that it can be used to index tensors
        all_locations = sorted(list(all_locations))

        # assign potential neighbours to facility groups
        for i in range(len(best_facilities)):
        
            f_i = best_facilities[i]
            
            for j in range(i + 1, len(best_facilities)):
            
                f_j = best_facilities[j]
                intersect = neighbour_sets[f_i].intersection(neighbour_sets[f_j])
                
                for client in intersect:
                
                    # first check if the client is the same as the facility
                    if client == f_i:
                    
                        neighbour_sets[f_j].remove(client)
                        
                    elif client == f_j:
                    
                        neighbour_sets[f_i].remove(client)
                        
                    # else assign to the nearest facility
                    else:
                    
                        dist_i = self._graph.get_standard_distance(client, f_i)
                        dist_j = self._graph.get_standard_distance(client, f_j)
                        
                        if dist_i > dist_j:
                        
                            neighbour_sets[f_j].remove(client)
                            
                        else:
                        
                            neighbour_sets[f_i].remove(client)

        # Sort the sets
        for facility in best_facilities:
        
            neighbour_sets[facility] = sorted(neighbour_sets[facility])

        # Remove original facilities if toggled
        if exclude_original:
        
            for facility in best_facilities:
            
                if len(neighbour_sets[facility]) > 1:
                
                    neighbour_sets[facility].remove(facility)
                    all_locations.remove(facility)

        self._neighbour_sets = dict(neighbour_sets)

        return all_locations

    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []
        
        for i in range(self._facility_length):
        
            if self._activation_values[i, 0] == 1:
            
                selected_facilities.append(self._facility_list[i])
                
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        
        return selected_facilities, selected_distance

    def _create_distance_array(self):
    
        return torch.tensor(self._graph.distance_cache, device=self._device)

    def _is_stabilized_facility(self):
    
        """
        # check that k facilities are active
        if torch.ne(torch.sum(self._activation_values[:, 0], dim=0), self._k):
            return False
        return True
        """
        
        is_stab = torch.sum(self._activation_values[:, 0], dim=0) == self._k
        # Code for Bfloat16 checks - can remove for regular Float32 types
        
        if is_stab:
        
            x = self._activation_values[:, 0]
            non_indices = torch.nonzero(x)
            non_zero = x[non_indices]
            typecst = non_zero.type(torch.int64)
            typecst = torch.flatten(typecst)
            py_list = typecst.tolist()
            
            if sum(py_list) != self._k:
            
                return False

        return is_stab



    def _is_stabilized_client(self):
        """
        """
        # Start with a quick check that the matrix adds up to n (ideally a 1.0 value in every column)
        if self._activation_values[:, 1:].sum() == self._n:
        
            # Do a longer check to see if every column contains exactly one 1.0 value.
            # Right now the check is to see if the average and max of a column is equal to 1.0
            # TODO: if needed, check to see if there is a faster check.
            #column_sums = self._activation_values[:, 1:].sum(dim=0)
            column_max_values, _ = self._activation_values[:, 1:].max(dim=0)
            #return torch.eq(column_sums, column_max).all()
            return torch.mean(column_max_values) == 1.0

        return False


    def _calculate_client_values(self):
    
        facility_values = torch.reshape(self._activation_values[:, 0], (self._facility_length, 1))
        self._inner_values[:, 1:] = facility_values * self._distance_values

    def _calculate_facility_values(self):
    
        self._inner_values[:, 0] = torch.sum(self._activation_values[:, 1:] * self._distance_values, dim=1)

    def _update_facility_full(self, facility):
    
        # TODO - May go back to clone() instead of item()
        old_value = self._activation_values[facility, 0].item()
        self._activation_values[facility, 0] = torch.eq(self._sorted_facility_indices, facility).any().float()
        #print("Is this facility in the top k?: ", facility)
        #print(self._sorted_facility_indices)
        #print(torch.eq(self._sorted_facility_indices, facility))
        #print(torch.eq(self._sorted_facility_indices, facility).any())
        #print(torch.eq(self._sorted_facility_indices, facility).any().float())
        #if old_value == 0.0 and self._activation_values[facility, 0].item() == 1.0:
        #    print("Updated a 0 to a 1")
        #    import sys
        #    sys.exit(0)
        return not (old_value == self._activation_values[facility, 0].item())

    def return_maxes(self, row_values):
    
        max_index = torch.argmax(row_values)
        max_value = torch.max(row_values)

        return max_index, max_value

    def update_value(self, facility, facility_value, facility_index, max_value, max_index):
    
        # if torch.gt(facility_value, max_value) or torch.eq(max_index, facility_index):
        # TODO - may be worth testing if gt values is faster than eq indices
        if torch.eq(max_index, facility_index):
        
            self._activation_values[facility, 0] = 1.0
            
        else:
        
            self._activation_values[facility, 0] = 0.0

    def _full_facility_index_to_neighbour_group(self, facility_index):
        """
        Using the facility index of the full array, check all neighbour sets and return the group that it is a part of.
        """
        for group in self._neighbour_sets.values():
        
            if facility_index in group:
            
                indices = group
                indices = sorted(indices)
                return indices

    def _convert_neighbour_set_indices_into_sub_array_indices(self, full_indices):
    
        # convert the indices to a form that's applicable to the sub-array
        sub_indices = []
        
        for index in full_indices:
        
            sub_indices.append(self._facility_list.index(index))
            
        return sub_indices

    def _update_facility_sub(self, facility):
    
        """
        When updating the sub-array we have to do two things:
            - Convert the facility into it's full-array index so that we can get the right neighbour set.
            - Use the neighbour set and create a smaller sub-array of just the search area
        """
        # Get the full-array index and use it to get the correct neighbour set
        full_facility_index = self._facility_list[facility]
        full_facility_neighbours = self._full_facility_index_to_neighbour_group(full_facility_index)
        sub_array_indices = self._convert_neighbour_set_indices_into_sub_array_indices(full_facility_neighbours)

        # Make the smaller sub-array of just the neighbour group.
        try:
        
            row_values = self._selected_row_cache[str(sub_array_indices)]
            
        except KeyError:
        
            selected_rows = self._inner_values[list(sub_array_indices)]
            row_values = selected_rows[:, 0]
            self._selected_row_cache[str(sub_array_indices)] = row_values

        # Get the sub-array facility value and the min/max values for the update.
        sub_array_facility_index = full_facility_neighbours.index(full_facility_index)
        facility_value = row_values[sub_array_facility_index]
        max_index, max_value = self.return_maxes(row_values)

        """
        # We can skip updating the value if we're decreasing a 0.0 value or increasing a 1.0 value.
        if torch.eq(self._activation_values[facility, 0], 0.0) and torch.lt(self._inner_values[facility, 0], max_value):
            return False
        if torch.eq(self._activation_values[facility, 0], 1.0) and torch.eq(self._inner_values[facility, 0], max_value):
            return False
        """

        # Otherwise, we update the value normally
        old_value = self._activation_values[facility, 0].item()
        self.update_value(facility, facility_value, sub_array_facility_index, max_value, max_index)
        return not (self._activation_values[facility, 0].item() == old_value)

    def _update_facility(self, facility):
    
        if len(self._neighbour_sets) == 0:
        
            return self._update_facility_full(facility)
            
        else:
        
            return self._update_facility_sub(facility)

    def _update_client_math(self):
    
        """
        This method sees if using indexing for calculations is faster
        for updating the client values.
        """
        max_indices = torch.argmax(self._inner_values[:, 1:], dim=0)
        
        # decrease all values
        self._activation_values[:, 1:] = 0.0
        
        # increase the max values
        self._activation_values[max_indices, self._math_column_indices] = 1.0

    def _update_client_original(self):
    
        """
        Original method to be used as a baseline for speed testing
        """
        max_indices = torch.argmax(self._inner_values[:, 1:], dim=0)
        self._activation_values[:, 1:] -= self._client_update_value
        
        for place, i in enumerate(max_indices):
        
            self._activation_values[i, place + 1] += (self._client_update_value * 2)

    def _update_client(self):
    
        old_values = self._activation_values[:, 1:].clone()
        self._update_client_math()
        return torch.ne(old_values, self._activation_values[:, 1:]).any()



class NonStationaryBandit:

    def __init__(self, num_actions, epsilon=0.3, alpha=0.5):
    
        self._num_actions = num_actions
        self._Q = np.zeros(num_actions)
        self._N = np.zeros(num_actions)
        self._epsilon = epsilon
        self._alpha = alpha

    def get_action(self):
    
        if random.random() < self._epsilon:
        
            return random.randint(0, self._num_actions - 1)
            
        else:
        
            return np.argmax(self._Q)

    def update_action(self, action, reward):
    
        self._N[action] += 1
        self._Q[action] += self._alpha * (reward - self._Q[action])


class ChangeBandit:

    def __init__(self, num_actions, epsilon=0.3):
    
        self._num_actions = num_actions
        self._unlocked = set([i for i in range(num_actions)])
        self._locked = set()
        self._epsilon = epsilon

    def reset(self):
    
        self._unlocked = set([i for i in range(self._num_actions)])
        self._locked = set()

    def get_action(self):
    
        if random.random() < self._epsilon:
        
            if len(self._locked) > 0:
            
                return random.choice(list(self._locked))
                
            else:
            
                return random.choice(list(self._unlocked))
                
        else:
        
            if len(self._unlocked) > 0:
            
                return random.choice(list(self._unlocked))
                
            else:
            
                return random.choice(list(self._locked))

    def update_action(self, action, reward):
    
        # move locked to unlocked only if there is a change
        if action in self._locked and reward == 1:
        
            self._locked.remove(action)
            self._unlocked.add(action)
            
        elif action in self._unlocked and reward == 0:
        
            self._unlocked.remove(action)
            self._locked.add(action)