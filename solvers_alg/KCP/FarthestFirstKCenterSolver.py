import torch

from problems.KCProblem import KCProblem
from solvers.brute_solver import calculate_radius
from solvers_alg.KCP.KCPSolver import KCPSolver


class FarthestFirstKCenterSolver(KCPSolver):
    def __init__(self, run_time=None):
        self._name = "Farthest-First K-Center"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None

        self._distances = None
        self._vertices = None
        self._facilities = None

    def initialize(self, problem: KCProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        if self._k < 0 or self._k > self._n:
            raise ValueError("k must satisfy 0 <= k <= n.")

        self._distances = self._graph._normalized_distances

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
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Solver must be initialized before calling solve().")

        if self._k == 0:
            self._selectedFacilities = []
            self._solutionValue = float("inf")
            self._vertices = torch.zeros(self._n, dtype=torch.int)
            self._facilities = torch.zeros((1, self._n), dtype=torch.int)
            return

        # Initial arbitrary head
        heads = [0]

        # dist_to_head[i] = distance from node i to the head of its current cluster
        dist_to_head = self._distances[:, heads[0]].clone()

        for _ in range(1, self._k):
            # pick node farthest from its current assigned head
            new_head = torch.argmax(dist_to_head).item()
            heads.append(new_head)

            # distance from every node to the new head
            dist_to_new_head = self._distances[:, new_head]

            # reassign nodes if new head is closer
            move_mask = dist_to_new_head <= dist_to_head
            dist_to_head[move_mask] = dist_to_new_head[move_mask]

        # Store facilities in 1 x n binary tensor
        self._facilities = torch.zeros((1, self._n), dtype=torch.int)
        self._facilities[0, heads] = 1

        # Keep _vertices too if other parts of your code use it
        self._vertices = self._facilities[0].clone()

        # Derive selected facilities and radius from _facilities
        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

        print("length: ", len(self._selectedFacilities))
        print(f"Farthest-First K-Center Solution Value: {self._solutionValue}")


    def calculate_radius(self, selected_indices):
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