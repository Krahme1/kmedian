from docplex.mp.model import Model

from problems.CKMProblem import CKMProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.CKMP.CKMPSolver import CKMPSolver


class CPLEXCKMPSolver(CKMPSolver):
    def __init__(self):
        self._name = "CPLEX ILP Solver - Capacitated K-Median (Uniform Capacity)"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._capacity = None

        self._model = Model(name="Capacitated_K_Median")

        self._clients = None
        self._facilities = None
        self._d = {}

        self._X = {}
        self._F = {}

        self._mipGap = None

    def initialize(self, problem: CKMProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        self._capacity = problem.getCapacity()

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set.")

        self._clients = [str(i) for i in range(self._n)]
        self._facilities = [str(i) for i in range(self._n)]

        # distances
        raw_distances = self.calculate_distances()
        for i, u in enumerate(self._clients):
            for j, v in enumerate(self._facilities):
                distance_value = raw_distances[i][j]
                if hasattr(distance_value, 'item'):
                    self._d[(u, v)] = distance_value.item() # Use .item() to extract the Python number
                else:
                    self._d[(u, v)] = distance_value # If it's already a number, just assign it

        # variables
        for j in self._facilities:
            self._F[j] = self._model.binary_var(name=f"F_{j}")

        for i in self._clients:
            for j in self._facilities:
                self._X[(i, j)] = self._model.binary_var(name=f"X_{i}_{j}")
        # objective: minimize total assignment cost
        self._model.minimize(
            self._model.sum(
                self._d[(i, j)] * self._X[(i, j)]
                for i in self._clients
                for j in self._facilities
            )
        )

        # exactly k facilities open
        self._model.add_constraint(
            self._model.sum(self._F[j] for j in self._facilities) == self._k,
            ctname="Open_k"
        )

        # each client assigned exactly once
        for i in self._clients:
            self._model.add_constraint(
                self._model.sum(self._X[(i, j)] for j in self._facilities) == 1,
                ctname=f"Assign_{i}"
            )

        # assign only to open facilities
        for i in self._clients:
            for j in self._facilities:
                self._model.add_constraint(
                    self._X[(i, j)] <= self._F[j],
                    ctname=f"OpenLink_{i}_{j}"
                )

        # uniform capacity constraint
        for j in self._facilities:
            self._model.add_constraint(
                self._model.sum(
                    self._X[(i, j)] for i in self._clients
                ) <= self._capacity,
                ctname=f"Capacity_{j}"
            )

    def getName(self):
        return self._name
    
    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities
    
    def getMIPGap(self):
        return self._mipGap

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def setCapacity(self, capacity):
        self._capacity = capacity

    def solve(self, max_time=None):
        if max_time is not None:
            self._model.set_time_limit(max_time)

        solution = self._model.solve(log_output=False)

        if solution is None:
            self._solutionValue = None
            self._selectedFacilities = []
            return

        selected_facilities = []
        for facility in self._facilities:
            val = self._F[facility].solution_value
            # numeric tolerances can vary; use 0.5 threshold for binary
            if val is not None and val >= 0.5:
                selected_facilities.append(int(facility))

        self._selectedFacilities = selected_facilities
        # compute objective value using your helper
        self._solutionValue = calculate_distance(self._graph, selected_facilities, self._n)
        self._mipGap = self._model.get_solve_details().mip_relative_gap

        self._model.end()

    def calculate_distances(self):
        distances = []
        for i in range(self._n):
            row = []
            for j in range(self._n):
                row.append(self._graph.get_standard_distance(i, j))
            distances.append(row)
        return distances

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def getSolutionValue(self):
        return self._solutionValue

    def getMIPGap(self):
        return self._mipGap

    def isOptimal(self):
        return self._model.is_optimal()
