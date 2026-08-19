# =============================================================
#  Local Search for k-median
#  Menu option : 3  ("Local Search Algorithm")
#  Class name  : LocalSearchSolver
#  Classic swap-based method: repeatedly swap a facility for a
#  non-facility node while it improves the solution.
# =============================================================
import random
import sys
import time

import torch

from problems.KMProblem import KMProblem
from solvers_alg.KMP.KMPSolver import KMPSolver


class LocalSearchSolver(KMPSolver):
    def __init__(self, max_time, solution=None):
        # Initialize Variables for Solver
        self._name = "Local Search"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._max_time = max_time
        self._check_counter = 0
        self._swap_counter = 0

        self._solution = solution
        self._vertices = None

        self._maxTime = 0

    def initialize(self, problem:KMProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        # prepare the initial vertices
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

        self._vertices = torch.tensor(vertices)

        if self._n < 1000:
            self._maxTime = 5
        elif self._n > 1000 and self._n < 1500:
            self._maxTime = 1
        elif self._n > 1500 and self._n < 3000:
            self._maxTime = 2
        elif self._n > 3000 and self._n < 5000:
            self._maxTime = 3
        elif self._n > 5000 and self._n < 6000:
            self._maxTime = 20
        elif self._n > 6000 and self._n < 15000 and self._k < 1000:
            self._maxTime = 50
        elif self._n > 6000 and self._n < 15000 and self._k == 1000:
            self._maxTime = 75
        elif self._n > 6000 and self._n < 15000 and self._k == 2000:
            self._maxTime = 100
        elif self._n > 6000 and self._n < 15000 and self._k > 2000:
            self._maxTime = 200


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
    
    def solve(self, runNum=None):
        start_time = time.time()
        best_distance = self.calculate_distance()
        while True:
            has_swapped = False
            for client in range(self._n):
                if self._vertices[client] == 0:
                    self._check_counter += 1
                    # check every possible swap
                    for facility in range(self._n):
                        if time.time() - start_time >= self._maxTime:
                            break
                        if self._vertices[facility] == 1:
                            self._vertices[facility] = 0
                            self._vertices[client] = 1
                            new_distance = self.calculate_distance()
                            # if our change in distance reaches a certain threshold we stop.
                            # We use the formula from 'Effectiveness of Local Search for Geometric Optimization'
                            # (Cohen-Addad and Mathieu 2015).
                            if new_distance < (1 - (1 / self._n)) * best_distance:
                                best_distance = new_distance
                                has_swapped = True
                                break
                            else:
                                self._vertices[facility] = 1
                                self._vertices[client] = 0
                    if has_swapped is True:
                        self._swap_counter += 1
                        break
            # if we have no better candidate we stop
            if not has_swapped:
                break

        self._selectedFacilities = [i for i in range(self._n) if self._vertices[i] == 1]
        self._solutionValue = best_distance


    def calculate_distance(self):
        """
        Helper function to get the total distance of all clients to their nearest facility.

        :param vertices: The solution to check
        :return: The sum of distances from clients to their nearest facility.
        """
        # The graph contains a normalized NxN array of distance values. We use facility_tensor to isolate only the
        # distance values for the clients to facilities. Then instead of getting the min (which will include the
        # unused facilities) we use the max and flip the distance with 1 - distances.
        # The max values of this will be the minimum distance values of the actual graph.
        max_values, _ = torch.max(self._vertices * (1 - self._graph._normalized_distances), dim=1)        
        # For now, flip back to the min values by subtracting 1 again. This is superfluous and can be changed but
        # will require changing the check in the loop as well.
        return torch.sum(1 - max_values)

   
