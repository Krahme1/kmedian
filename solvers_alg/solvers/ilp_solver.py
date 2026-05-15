import pulp

class FacilityLocationSolver:
    def __init__(self, graph, n, k):
        self.graph = graph
        self.n = n
        self.k = k
        self.model = pulp.LpProblem(name="facility-location-problem", sense=pulp.LpMinimize)
        # store the client and facility indices to make future lines easier to read
        self.clients = [str(i) for i in range(n)]
        self.facilities = [str(i) for i in range(n)]
        # this dictionary contains the values for the distance from u to v
        self.d = pulp.makeDict([self.clients, self.facilities], self.calculate_distances())
        # Connection variables - X[u][v] is 1 if client u is connected to facility v
        #self.X = pulp.LpVariable.dict("Connections", (self.clients, self.facilities), 0, 1, "Integer")
        self.X = pulp.LpVariable.dict("Connections", (self.clients, self.facilities), cat="Binary")
        # Facility variables - F[i] is 1 if facility i is selected
        #self.F = pulp.LpVariable.dict("Facilities", self.facilities, 0, 1, "Integer")
        self.F = pulp.LpVariable.dict("Facilities", self.facilities, cat="Binary")
        # add the objective function
        #self.model += pulp.lpSum(self.d[u][v] * self.X[(u, v)] for u in self.clients for v in self.facilities if self.F[v] == 1.0)
        self.model += pulp.lpSum(self.d[u][v] * self.X[(u, v)] for u in self.clients for v in self.facilities)
        # add constraints
        self.model += (pulp.lpSum(self.F) <= k, "Up to K facilities")
        for u in self.clients:
            self.model += (pulp.lpSum(self.X[(u, v)] for v in self.facilities) == 1), f"Client {u} is connected"
        for u in self.clients:
            for v in self.facilities:
                self.model += self.X[(u, v)] <= self.F[v], f"test {u}_{v}"

    def solve(self, max_time=None):
        self.model.solve(solver=pulp.PULP_CBC_CMD(msg=False, timeLimit=max_time))

        selected_facilities = []
        for facility in self.facilities:
            if self.F[facility].value() == 1.0:
                selected_facilities.append(int(facility))
       
        return selected_facilities

    def warm_start(self, solution, client_connections):
        for facility in solution:
            self.F[str(facility)].setInitialValue(1)

        for i in range(self.n):
            for j in range(self.n):
                self.X[(str(i), str(j))].setInitialValue(0)

        for pair in client_connections:

            self.X[(str(pair[0]), str(pair[1]))].setInitialValue(1)

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
        for i in range(self.n):
            distance_row = []
            for j in range(self.n):
                distance_row.append(self.graph.get_standard_distance(i, j))
            distances.append(distance_row)

        return distances
