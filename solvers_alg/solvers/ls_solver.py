import pulp

from solvers_alg.solvers.brute_solver import calculate_distance


class LocalSearchSolver:
    """
    This solver will try to find the best swap for a given solution of the facility location problem such that
    the swap reduces the total distance.
    """
    def __init__(self, graph, n, k, selected_facilities):
        self.graph = graph
        self.n = n
        self.k = k
        self.original_distance = calculate_distance(self.graph, selected_facilities, self.n)
        self.model = pulp.LpProblem(name="hopfield-huge.txt-search-swap", sense=pulp.LpMaximize)
        # store the client and facility indices to make future lines easier to read
        self.clients = [str(i) for i in range(n)]
        self.facilities = [str(i) for i in range(n)]
        # this dictionary contains the values for the distance from u to v
        self.d = pulp.makeDict([self.clients, self.facilities], self.calculate_distances())
        # Connection variables - X[u][v] is 1 if client u is connected to facility v
        self.X = self.initialize_connections(selected_facilities)
        # The connections for the swapped variables
        self.X_prime = pulp.LpVariable.dict("Connections_prime", (self.clients, self.facilities), 0, 1, "Integer")
        # Facility variables - F[i] is 1 if facility i is selected
        self.F = self.initialize_original_facilities(selected_facilities)
        # The swapped set of facilities
        self.F_prime = pulp.LpVariable.dict("Facilities_prime", self.facilities, 0, 1, "Integer")

        # add the objective function
        #self.model += pulp.lpSum(self.d[u][v] * self.X[(u, v)] for u in self.clients for v in self.facilities if self.F[v] == 1.0)
        self.model += pulp.lpSum(self.original_distance - (self.d[u][v] * self.X_prime[(u, v)]) for u in self.clients for v in self.facilities if self.F_prime[v] == 1.0)

        # add constraints
        # must be k facilities
        self.model += (pulp.lpSum(self.F_prime) == k, "K facilities in the swapped set")
        # clients must be connected to facilities
        for u in self.clients:
            self.model += (pulp.lpSum(self.X_prime[(u, v)] for v in self.facilities) >= 1), f"Client {u} is connected in the swapped set"
        for u in self.clients:
            for v in self.facilities:
                self.model += self.X_prime[(u, v)] <= self.F_prime[v], f"test {u}_{v} in swapped"
        # must have only swapped 1 facility
        self.model += (pulp.lpSum(self.F[v] * self.F_prime[v] for v in self.facilities) == self.k - 1, "Only one facility has been swapped")

    def initialize_original_facilities(self, selected_facilities):
        facilities = {}
        for i in range(self.n):
            facilities[str(i)] = 1 if i in selected_facilities else 0
        return facilities

    def initialize_connections(self, initial_facilities):
        connections = {}
        # 0 out the initial connections
        for i in range(self.n):
            for j in range(self.n):
                connections[(str(i), str(j))] = 0

        # connect each client to their closest facility
        for client in range(self.n):
            min_distance = None
            selected_facility = None
            for facility in initial_facilities:
                distance = self.graph.get_distance(client, facility)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    selected_facility = facility

            connections[(str(client), str(selected_facility))] = 1
        return connections

    def solve(self):
        """
        Select the best variables to swap.
        :return: A pair where the first node is to be turned on and the second node is to be turned off.
        """
        self.model.solve(solver=pulp.PULP_CBC_CMD(msg=False))
        swap = {}

        for facility in self.facilities:
            if int(self.F[facility]) == 0 and int(self.F_prime[facility].value()) == 1:
                swap["client"] = int(facility)
            elif int(self.F[facility]) == 1 and int(self.F_prime[facility].value()) == 0:
                swap["facility"] = int(facility)

        return swap

    def is_optimal(self):
        """
        Helper function to see if the solution is optimal.

        :return: True if the solution is optimal, False otherwise.
        """
        # We assert that a solution has been found
        assert(self.model.sol_status == 1 or self.model.sol_status == 2)
        return self.model.sol_status == 1

    def calculate_distances(self):
        """
        Construct a distance array where each row contains the distance
        from i to locations j.
        """
        distances = []
        for i in range(self.n):
            distance_row = []
            for j in range(self.n):
                distance_row.append(self.graph.get_distance(i, j))
            distances.append(distance_row)

        return distances
