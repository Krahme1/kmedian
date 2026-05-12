import random
import sys
import time

from solvers.brute_solver import calculate_radius
import torch

from problems.KCProblem import KCProblem
from solvers_alg.KCP.KCPSolver import KCPSolver


class LocalSearchSolverKCenter(KCPSolver):
    def __init__(self, run_time, solution=None):
        # Initialize Variables for Solver
        self._name = "Local Search K-Center"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._run_time = run_time
        self._check_counter = 0
        self._swap_counter = 0

        self._solution = solution
        self._vertices = None

        self._maxTime = 0

        self._facilities = None 

    def initialize(self, problem:KCProblem):
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

        # keep facilities consistent with initial solution too
        self._facilities = self._vertices.unsqueeze(0).clone()

        if self._n < 1000:
            self._maxTime = 5
        elif self._n > 1000 and self._n < 1500:
            self._maxTime = 1
        elif self._n > 1500 and self._n < 3000:
            self._maxTime = 2
        elif self._n > 3000 and self._n < 5000:
            self._maxTime = 3

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
        best_radius = self.calculate_radius()

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

                            new_radius = self.calculate_radius()

                            # accept only sufficiently improving swaps
                            if new_radius < (1 - (1 / self._n)) * best_radius:
                                best_radius = new_radius
                                has_swapped = True
                                break
                            else:
                                self._vertices[facility] = 1
                                self._vertices[client] = 0

                    if has_swapped:
                        self._swap_counter += 1
                        break

            # if we have no better candidate we stop
            if not has_swapped:
                break

        # copy final solution into facilities representation
        self._facilities = self._vertices.unsqueeze(0).clone()

        # derive selected facilities and radius from helper
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

    def calculate_radius(self):
        """
        True k-center objective:
        - For each client: distance to closest facility (MIN)
        - Then take the WORST client (MAX)
        """

        # Keep only facility distances
        facility_mask = self._vertices.bool()
        distances = self._graph._normalized_distances[:, facility_mask]

        # Closest facility for each client
        min_dist_per_client, _ = torch.min(distances, dim=1)

        # Worst-covered client → radius
        radius = torch.max(min_dist_per_client)

        return radius.item()
    
    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []

        for i in range(self._n):
            if self._facilities[0, i] == 1:
                selected_facilities.append(i)

        radius = calculate_radius(self._graph, selected_facilities)

        return selected_facilities, radius