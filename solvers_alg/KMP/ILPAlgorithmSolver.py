import pulp

from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class ILPAlgorithmSolver(KMPSolver):
    def __init__(self, problem:KMProblem):
        # Initialize Variables for Solver
        self._name = "ILPSolver"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        self._model = pulp.LpProblem(name="facility-location-problem", sense=pulp.LpMinimize)
        # store the client and facility indices to make future lines easier to read
        self._clients = None
        self._facilities = None
        # this dictionary contains the values for the distance from u to v
        self._d = None
        # Connection variables - X[u][v] is 1 if client u is connected to facility v
        #self.X = pulp.LpVariable.dict("Connections", (self.clients, self.facilities), 0, 1, "Integer")
        self._X = None
        # Facility variables - F[i] is 1 if facility i is selected
        #self.F = pulp.LpVariable.dict("Facilities", self.facilities, 0, 1, "Integer")
        self._F = None
        # add the objective function
        #self.model += pulp.lpSum(self.d[u][v] * self.X[(u, v)] for u in self.clients for v in self.facilities if self.F[v] == 1.0)

    def initialize(self):
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        # store the client and facility indices to make future lines easier to read
        self._clients = [str(i) for i in range(self._n)]
        self._facilities = [str(i) for i in range(self._n)]
        # this dictionary contains the values for the distance from u to v
        self._d = pulp.makeDict([self._clients, self._facilities], self.calculate_distances())
        # Connection variables - X[u][v] is 1 if client u is connected to facility v
        #self.X = pulp.LpVariable.dict("Connections", (self.clients, self.facilities), 0, 1, "Integer")
        self._X = pulp.LpVariable.dict("Connections", (self._clients, self._facilities), cat="Binary")
        # Facility variables - F[i] is 1 if facility i is selected
        #self.F = pulp.LpVariable.dict("Facilities", self.facilities, 0, 1, "Integer")
        self._F = pulp.LpVariable.dict("Facilities", self._facilities, cat="Binary")
        # add the objective function
        #self.model += pulp.lpSum(self.d[u][v] * self.X[(u, v)] for u in self.clients for v in self.facilities if self.F[v] == 1.0)
        self._model += pulp.lpSum(self._d[u][v] * self._X[(u, v)] for u in self._clients for v in self._facilities)
        # add constraints
        self._model += (pulp.lpSum(self._F) <= self._k, "Up to K facilities")
        for u in self._clients:
            self._model += (pulp.lpSum(self._X[(u, v)] for v in self._facilities) == 1), f"Client {u} is connected"
        for u in self._clients:
            for v in self._facilities:
                self._model += self._X[(u, v)] <= self._F[v], f"test {u}_{v}"

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

    def solve(self, max_time=None):
        self.model.solve(solver=pulp.PULP_CBC_CMD(msg=False, timeLimit=max_time))

        selected_facilities = []
        for facility in self._facilities:
            if self._F[facility].value() == 1.0:
                selected_facilities.append(int(facility))
       
        self._selectedFacilities = selected_facilities
        self._solutionValue = calculate_distance(self._graph, selected_facilities, self._n)

    def warm_start(self, solution, client_connections):
        for facility in solution:
            self._F[str(facility)].setInitialValue(1)

        for i in range(self._n):
            for j in range(self._n):
                self._X[(str(i), str(j))].setInitialValue(0)

        for pair in client_connections:

            self._X[(str(pair[0]), str(pair[1]))].setInitialValue(1)

    def is_optimal(self):
        """
        Helper function to see if the solution is optimal.

        :return: True if the solution is optimal, False otherwise.
        """
        # We assert that a solution has been found
        #assert(self.model.sol_status == 1 or self.model.sol_status == 2)
        #return self.model.sol_status == 1
        return pulp.LpStatus[self.model.status] == 'Optimal'

    def calculate_distances(self):
        """
        Construct a distance array where each row contains the distance
        from i to locations j.
        """
        distances = []
        for i in range(self._n):
            distance_row = []
            for j in range(self._n):
                distance_row.append(self.graph.get_standard_distance(i, j))
            distances.append(distance_row)

        return distances

