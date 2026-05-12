from docplex.mp.model import Model

from problems.KFProblem import KFProblem
from solvers.brute_solver import calculate_distance
from solvers_alg.KF.KFSolver import KFSolver


class CPLEXKFSolver(KFSolver):
    def __init__(self):
        self._name = "CPLEX ILP Solver - K-Facility Location Problem"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None

        # facility opening costs
        self._facilityCosts = None

        # DOcplex model
        self._model = Model(name="K-Facility Location Problem")

        # indices and data placeholders
        self._clients = None
        self._facilities = None

        # distance values
        self._d = {}

        # facility opening costs keyed by facility id string
        self._c = {}

        # variables
        self._X = {}
        self._F = {}

        self._mipGap = None

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

    def setFacilityCosts(self, facility_costs):
        self._facilityCosts = facility_costs

    def initialize(self, problem: KFProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        self._facilityCosts = problem.getCosts()

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        if self._facilityCosts is None:
            raise ValueError("facility_costs must be provided.")

        if len(self._facilityCosts) != self._n:
            raise ValueError("facility_costs must have length n.")

        self._clients = [str(i) for i in range(self._n)]
        self._facilities = [str(i) for i in range(self._n)]

        # distances
        raw_distances = self.calculate_distances()
        for i, u in enumerate(self._clients):
            for j, v in enumerate(self._facilities):
                distance_value = raw_distances[i][j]
                if hasattr(distance_value, "item"):
                    self._d[(u, v)] = distance_value.item()
                else:
                    self._d[(u, v)] = distance_value

        # facility costs
        for j, v in enumerate(self._facilities):
            cost_value = self._facilityCosts[j]
            if hasattr(cost_value, "item"):
                self._c[v] = cost_value.item()
            else:
                self._c[v] = cost_value

        # create binary variables
        for v in self._facilities:
            self._F[v] = self._model.binary_var(name=f"F_{v}")

        for u in self._clients:
            for v in self._facilities:
                self._X[(u, v)] = self._model.binary_var(name=f"X_{u}_{v}")

        # objective:
        # minimize assignment cost + facility opening cost
        self._model.minimize(
            self._model.sum(
                self._d[(u, v)] * self._X[(u, v)]
                for u in self._clients
                for v in self._facilities
            )
            +
            self._model.sum(
                self._c[v] * self._F[v]
                for v in self._facilities
            )
        )

        # open exactly k facilities
        self._model.add_constraint(
            self._model.sum(self._F[v] for v in self._facilities) == self._k,
            ctname="Exactly_K_facilities"
        )

        # each client connected to exactly one facility
        for u in self._clients:
            self._model.add_constraint(
                self._model.sum(self._X[(u, v)] for v in self._facilities) == 1,
                ctname=f"Client_{u}_connected"
            )

        # connection implies facility opened
        for u in self._clients:
            for v in self._facilities:
                self._model.add_constraint(
                    self._X[(u, v)] <= self._F[v],
                    ctname=f"Link_{u}_{v}"
                )

    def solve(self, max_time=None):
        if max_time is not None:
            self._model.set_time_limit(max_time)

        solution = self._model.solve(log_output=False)

        if solution is None:
            self._selectedFacilities = []
            self._solutionValue = float("inf")
            return

        selected_facilities = []
        for facility in self._facilities:
            val = self._F[facility].solution_value
            if val is not None and val >= 0.5:
                selected_facilities.append(int(facility))

        self._selectedFacilities = selected_facilities
        self._solutionValue = self.calculate_objective_value(selected_facilities)
        self._mipGap = self._model.get_solve_details().mip_relative_gap
        self._model.end()

    def warm_start(self, solution, client_connections):
        for facility in self._facilities:
            var = self._F[facility]
            for attr in ("start", "start_value"):
                try:
                    setattr(var, attr, 0)
                except Exception:
                    pass

        for i in range(self._n):
            for j in range(self._n):
                var = self._X[(str(i), str(j))]
                for attr in ("start", "start_value"):
                    try:
                        setattr(var, attr, 0)
                    except Exception:
                        pass

        for facility in solution:
            var = self._F[str(facility)]
            for attr in ("start", "start_value"):
                try:
                    setattr(var, attr, 1)
                except Exception:
                    pass

        for pair in client_connections:
            c, f = pair
            var = self._X[(str(c), str(f))]
            for attr in ("start", "start_value"):
                try:
                    setattr(var, attr, 1)
                except Exception:
                    pass

    def isOptimal(self):
        return self._model.is_optimal()

    def calculate_distances(self):
        distances = []
        for i in range(self._n):
            distance_row = []
            for j in range(self._n):
                distance_row.append(self._graph.get_standard_distance(i, j))
            distances.append(distance_row)
        return distances
    
    def calculate_objective_value(self, selected_facilities):
        if len(selected_facilities) == 0:
            return float("inf")

        total_distance = 0

        for i in range(self._n):
            best_distance = float("inf")
            for facility in selected_facilities:
                d = self._graph.get_standard_distance(i, facility)
                if hasattr(d, "item"):
                    d = d.item()
                if d < best_distance:
                    best_distance = d
            total_distance += best_distance

        total_opening_cost = 0

        for facility in selected_facilities:
            cost = self._facilityCosts[facility]
            if hasattr(cost, "item"):
                cost = cost.item()
            total_opening_cost += cost

        return total_distance + total_opening_cost

    def write_lp(self, path: str):
        self._model.export_as_lp(path)