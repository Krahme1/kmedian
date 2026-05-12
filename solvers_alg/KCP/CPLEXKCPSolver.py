from docplex.mp.model import Model

from problems.KCProblem import KCProblem
from solvers_alg.KCP.KCPSolver import KCPSolver


class CPLEXKCPSolver(KCPSolver):
    def __init__(self):
        self._name = "CPLEX ILP Solver - K-Median Problem"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None

        # DOcplex model
        self._model = Model(name="K-Median Problem")

        # indices and data placeholders
        self._clients = None
        self._facilities = None

        # d[(u,v)] distance values (keys are strings to match your original usage)
        self._d = {}

        # variables
        self._X = {}
        self._F = {}
        self._R = None

        self._mipGap = None

    def initialize(self, problem: KCProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        self._clients = [str(i) for i in range(self._n)]
        self._facilities = [str(i) for i in range(self._n)]

        # Build distance dictionary
        raw = self.calculate_distances()
        for i, u in enumerate(self._clients):
            for j, v in enumerate(self._facilities):
                distance_value = raw[i][j]
                if hasattr(distance_value, 'item'):
                    self._d[(u, v)] = distance_value.item() # Use .item() to extract the Python number
                else:
                    self._d[(u, v)] = distance_value # If it's already a number, just assign it

        # Create variables
        for v in self._facilities:
            self._F[v] = self._model.binary_var(name=f"F_{v}")

        for u in self._clients:
            for v in self._facilities:
                self._X[(u, v)] = self._model.binary_var(name=f"X_{u}_{v}")

        self._R = self._model.continuous_var(lb=0, name="R")

        # Objective: minimize radius
        self._model.minimize(self._R)

        # Constraints
        # 1. At most k facilities
        self._model.add_constraint(self._model.sum(self._F[v] for v in self._facilities) <= self._k, ctname="Up_to_K_facilities")

        # 2. Every client must be assigned to exactly one facility
        for u in self._clients:
            self._model.add_constraint(self._model.sum(self._X[(u, v)] for v in self._facilities) == 1, ctname=f"Client_{u}_connected")

        # 3. Cannot assign a client to a closed facility
        for u in self._clients:
            for v in self._facilities:
                self._model.add_constraint(self._X[(u, v)] <= self._F[v], ctname=f"Link_{u}_{v}")

        # 4. Radius definition: d(u,v) * X[u,v] <= R
        for u in self._clients:
            for v in self._facilities:
                self._model.add_constraint(self._d[(u, v)] * self._X[(u, v)] <= self._R, ctname=f"radius_{u}_{v}")

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

    def solve(self, max_time=None):
        if max_time is not None:
            self._model.set_time_limit(max_time)

        solution = self._model.solve(log_output=False)

        if solution is None:
            self._selectedFacilities = []
            self._solutionValue = None
            return
        
        selected_facilities = []
        for facility in self._facilities:
            val = self._F[facility].solution_value
            # numeric tolerances can vary; use 0.5 threshold for binary
            if val is not None and val >= 0.5:
                selected_facilities.append(int(facility))

        self._selectedFacilities = selected_facilities
        self._solutionValue = solution.get_value(self._R)

        self._mipGap = self._model.solve_details.mip_relative_gap
        self._model.end()     # Frees all CPLEX internal memory

    def warm_start(self, solution, client_connections):
        """Warm-start the k-center solver by setting start values on variables.

        `solution` should be an iterable of facility indices to open.
        `client_connections` should be iterable of (client, facility) pairs.
        """
        # --- 1. Reset all starts to 0 ---
        for facility in self._facilities:
            var = self._F[facility]
            for attr in ('start', 'start_value'):
                try:
                    setattr(var, attr, 0)
                except Exception:
                    pass

        for i in range(self._n):
            for j in range(self._n):
                var = self._X[(str(i), str(j))]
                for attr in ('start', 'start_value'):
                    try:
                        setattr(var, attr, 0)
                    except Exception:
                        pass

        # --- 2. Set provided warm-start facilities ---
        for facility in solution:
            var = self._F[str(facility)]
            for attr in ('start', 'start_value'):
                try:
                    setattr(var, attr, 1)
                except Exception:
                    pass

        # --- 3. Set warm-start client assignments ---
        for pair in client_connections:
            c, f = pair
            var = self._X[(str(c), str(f))]
            for attr in ('start', 'start_value'):
                try:
                    setattr(var, attr, 1)
                except Exception:
                    pass

        # --- 4. Set R to max assigned distance to be feasible ---
        if solution and client_connections:
            R_start = 0
            for c, f in client_connections:
                R_start = max(R_start, self._d[(str(c), str(f))])
            for attr in ('start', 'start_value'):
                try:
                    setattr(self._R, attr, R_start)
                except Exception:
                    pass

    def isOptimal(self):
        return self._model.is_optimal()
    
    def calculate_distances(self):
        """
        Reuse your original approach for constructing the distance matrix.
        """
        distances = []
        for i in range(self._n):
            distance_row = []
            for j in range(self._n):
                distance_row.append(self._graph.get_standard_distance(i, j))
            distances.append(distance_row)

        return distances
    
    def write_lp(self, path: str):
        """Write the underlying CPLEX .lp file so you can inspect or load it in CPLEX/Studio.

        Example: solver.write_lp('/tmp/facility.lp')
        """
        self._model.export_as_lp(path)