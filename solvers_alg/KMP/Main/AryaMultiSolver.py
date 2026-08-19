# =============================================================
#  Arya et al. local search for k-median (LS in the paper)
#  Menu option : 5  ("Arya Multi Swap")
#  Class name  : AryaMultiSolver
#  The well-known local-search algorithm with a provable
#  approximation guarantee; supports multi-swaps.
# =============================================================
import random
import time

import torch

from problems.KMProblem import KMProblem
from solvers_alg.KMP.KMPSolver import KMPSolver


class AryaMultiSolver(KMPSolver):
    def __init__(self, max_time, solution=None):
        # Initialize Variables for Solver
        self._name = "Arya Multi"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._max_time = max_time
        self._check_counter = 0
        self._swap_counter = 0
        self._maxTime = 0

        self._vertices = None
        self._solution = solution

    def initialize(self, problem:KMProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        if self._solution:
            vertices = []
            for i in range(self._n):
                if i in self._solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(self._n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, self._n)], k=self._k):
                vertices[value] = 1
        self._vertices = vertices

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
    
    def setN (self, n):
        self._n = n

    def setK (self, k):
        self._k = k

    def setGraph (self, graph):
        self._graph = graph
    
    """
    Randomly select a client and compare swapping it with each facility.
    If the distance is improved, we select the swap pair that decreases the distance the most.
    Otherwise, keep the original state.

    :return: None
    """
    def solve(self, runNum=None):

        #print("START")
        start_time = time.time()
        best_distance = self.calculate_distance(self._vertices)
        #print("Max time:", self.max_time)
        
        while True:
        
            #print("IN LOOP")
            open_locations = [i for i in range(self._n) if self._vertices[i] == 0]
            facilities = [i for i in range(self._n) if self._vertices[i] == 1]
            has_swapped = False
            
            while has_swapped is False:
            
                client1, client2 = random.sample(open_locations, k=2)
                facility1, facility2 = random.sample(facilities, k=2)
                self._check_counter += 1
                #facility = None
            
                temp_vertices = self._vertices.copy()
                temp_vertices[facility1] = 0
                temp_vertices[facility2] = 0
                temp_vertices[client1] = 1
                temp_vertices[client2] = 1
                new_distance = self.calculate_distance(temp_vertices)
            
                if new_distance < (1 - (1 / self._n)) * best_distance:
                        
                    #print("UPDATE SOLUTION")
                    best_distance = new_distance
                    self._vertices[facility1] = 0
                    self._vertices[facility2] = 0
                    self._vertices[client1] = 1
                    self._vertices[client2] = 1
                    has_swapped = True
                    self._swap_counter += 1
                
                if time.time() - start_time >= self._maxTime:
                    
                    #print("HIT TIME LIMIT")
                    break 
                    
            if time.time() - start_time >= self._maxTime:
                    
                #print("HIT TIME LIMIT")
                break       

        self._selectedFacilities = [i for i in range(self._n) if self._vertices[i] == 1]
        self._solutionValue = best_distance

    """
    Helper function to get the total distance of all clients to their nearest facility.
    :param vertices: The solution to check
    :return: The sum of distances from clients to their nearest facility.
    """
    def calculate_distance(self, vertices):
    
        facility_tensor = torch.tensor(vertices)
        max_values, _ = torch.max(facility_tensor * (1 - self._graph._normalized_distances), dim=1)
        return torch.sum(1 - max_values)