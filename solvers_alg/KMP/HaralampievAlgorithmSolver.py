import math
import random

import numpy as np

from problems.KMProblem import KMProblem
from solvers.brute_solver import calculate_distance, get_facilities
from solvers_alg.KMP.KMPSolver import KMPSolver


def calculate_temperature_decay(T, iterations):
    """
    Perform a simple binary search to find a value a that will make
    T ~= 0.001 after a number of iterations
    :param T: The temperature value.
    :param iterations: The number of times to decrease T
    :return: The decay value a that will reduce T to ~0.001
    """
    start = 0.0
    end = 1.0
    a = 0.5
    for _ in range(10):
        dist = end - start
        a = start + (dist / 2)
        value = T * pow(a, iterations)
        if value < 0.001:
            start = a
        else:
            end = a

    return a


class HaralampievAlgorithmSolver(KMPSolver):
    def __init__(self, temperature, epoch_length, decay_interval, solution=None):
        # Initialize Variables for Solver
        self._name = "Haralampiev Network"
        self._solutionValue = 0
        self._selectedFacilities = []

        # Variables to be used by the solve function
        self._temperature = temperature
        self._epoch_length = epoch_length  # Set to none in order to dynamically get n at runtime
        self._alpha = calculate_temperature_decay(self._temperature, decay_interval)

        self._n = None
        self._k = None
        self._G = None

        # caches to speed up getting the on group members
        self._on_client_cache = {}
        self._on_facility_group_cache = {}
        self._on_facility_location_cache = {}

        # This cache keeps track of which clients are on for facility calculations
        # i.e., the facility f[j,s] will get the group of on c[i,j] clients.
        # Note: not an explicit on group.
        self._on_client_location_cache = {}

        # caches to speed-up/avoid unnecessary calculate unit value calls
        self._client_unit_value_cache = {}
        self._facility_unit_value_cache = {}

        self._total_vertices = None
        self._V = None
        self._distances = None
        self._maxTime = None

    def initialize(self, problem:KMProblem):

        self._n = problem.getN()
        self._k = problem.getK()
        self._G = problem.getGraph()

        if self._G is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        self._total_vertices = 2 * self._n * self._k
        self._V = np.random.randint(0, 2, self._total_vertices)
        #self.V = torch.randint(low=0, high=2, size=(self.total_vertices,), device="cuda")
        #self.group = [[] for _ in range(self.total_vertices)]
        #self.con = [{} for _ in range(self.total_vertices)]
        #self.generate_groups(self.n, self.k)
        #self.generate_weights(self.n, self.k)

        self._distances = self._G._distances.numpy()
        #self.distances = self.G._distances.to(device="cuda", copy=True)
        
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

    def warm_start(self, solution):
        self._V = np.zeros(self._total_vertices)
        # Set the facilities first
        for i, value in enumerate(solution):
            facility_start = (self._n * self._k) + (i * self._n)
            self._V[facility_start + value] = 1

        # Connect clients to the closest facility
        for client in range(self._n):
            min_distance = None
            min_facility = None
            for facility in solution:
                distance = self._distances[client, facility]
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_facility = facility

            facility_index = solution.index(min_facility)
            client_group = self._k * client
            self._V[client_group + facility_index] = 1


    def cache_on_groups(self):
        # Go through each client group and cache the on members
        for client_group in range(self._n):
            client_start = client_group * self._k
            client_end = client_start + self._k
            on_clients = np.flatnonzero(self._V[client_start:client_end])
            # flatnonzero returns the indices for the slice, so we need to add the group start to get the actual
            # unit value.
            on_clients = np.add(on_clients, client_start)
            self._on_client_cache[client_group] = sorted(on_clients.tolist())

        # Group on clients by the facility group that serves them
        for facility_group in range(self._k):
            client_start = facility_group
            client_indices = np.array([i for i in range(client_start, self._n * self._k, self._k)])
            client_values = self._V[client_indices]
            on_indices = np.flatnonzero(client_values)
            self._on_client_location_cache[facility_group] = client_indices[on_indices].tolist()

        # We need to cache the n-sized groups and the location groups separately.
        # Start with the n-sized groups
        for facility_group in range(self._k):
            facility_start = (self._n * self._k) + (facility_group * self._n)
            facility_end = facility_start + self._n
            on_facilities = np.flatnonzero(self._V[facility_start:facility_end])
            on_facilities = np.add(on_facilities, facility_start)
            self._on_facility_group_cache[facility_group] = sorted(on_facilities.tolist())

        # Then get the facilities across groups that share the same location
        for location in range(self._n):
            start_index = (self._n * self._k) + location
            location_indices = [i for i in range(start_index, self._total_vertices, self._n)]
            np_location_indices = np.array(location_indices)
            on_location_indices = np.flatnonzero(self._V[location_indices])
            location_members = np_location_indices[on_location_indices]
            self._on_facility_location_cache[location] = sorted(location_members.tolist())

    def cache_unit_value_groups(self):
        # Put dummy data into the client groups so that the value is calculated when needed
        for client in range(self._n * self._k):
            self._client_unit_value_cache[client] = (None, 0)

        for facility in range(self._n * self._k, self._total_vertices):
            self._facility_unit_value_cache[facility] = (None, 0)

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
        self._G = graph
    
    def solve(self, runNum=None):
        warm_instance = RandomChoiceAlgorithm()
        warm_solution = warm_instance.run(self._G, self._n, self._k)
        self.warm_start(warm_solution)
        # Epoch length can be variable so we allow a None assignment in order to signal that we should use n
        if self._epoch_length is None:
            epoch_length = self._n
        else:
            epoch_length = self._epoch_length

        # Set up the cache here instead of the init function because the network can be reused with different
        # starting values
        self.cache_on_groups()
        self.cache_unit_value_groups()

        units = [i for i in range(self._total_vertices)]
        has_stabilized = False
        while not has_stabilized:
            random.shuffle(units)
            for unit in units:
                self.update(self._temperature, unit)

            # The max is here to avoid too small of a value being treated as a 0 (we divide by T later)
            self._temperature = max(0.0000001, self._temperature * self._alpha)
            has_stabilized = self.is_valid()

            facilities = get_facilities(self, self._n, self._k)

        self._selectedFacilities = facilities
        self._solutionValue = calculate_distance(self._G, facilities, self._n)
    
    def update(self, temperature, unit):
        """
        Perform an update on a random unit X.
        Start by getting the ON set of connected units who have a value of 1.
        If this set is empty, set X to 1 and exit.

        Otherwise, compute value[X] and the minimum value[s] for each s in group[X].
        Then X is set to 1 if value[X] is less than the minimum value[s] and 0 if not.
        Finally, flip X with a probability calculated from the temperature.
        :param temperature: The value used to calculate the probability to flip X.
        :return: None
        """
        # select a binary variable at random
        X = unit

        ON = self.get_on_group(X)

        # TODO temporarily convert to python list - full implementation should stick to numpy
        if type(ON) != type(list()):
            ON = ON.tolist()
        # Remove self if in the group
        if X in ON:
            ON.remove(X)
        # calculate the BEST value from the on group
        on_values = [self.calculate_unit_value(unit) for unit in ON]
        # If we receive an empty list then BEST is None
        if not on_values:
            # if there are no ON values in a group, then flip the variable on
            self._V[X] = 1
        else:
            BEST = min(on_values)
            X_value = self.calculate_unit_value(X)
            delta = abs(BEST - X_value)

            if X_value < BEST:
                self._V[X] = 1
                if self.accept(delta, temperature):
                    self._V[X] = 0
            else:
                self._V[X] = 0
                if self.accept(delta, temperature):
                    self._V[X] = 1

        self.cache_unit_on_status(X)

    def cache_unit_on_status(self, X):
        # make sure the on group cache is updated correctly
        if X < self._n * self._k:
            # Cache the k-sized competing client groups
            client_group = X // self._k
            # If X is on and not in the group, add it
            if self._V[X] == 1 and (X not in self._on_client_cache[client_group]):
                self._on_client_cache[client_group].append(X)
                self._on_client_cache[client_group] = sorted(self._on_client_cache[client_group])
            # If X is off and in the group remove it.
            elif self._V[X] == 0 and (X in self._on_client_cache[client_group]):
                self._on_client_cache[client_group].remove(X)

            # Cache the clients based on the serving facility group i.e group 0 contains C(0,0), C(1,0), ...
            facility_group = X % self._k
            if self._V[X] == 1 and (X not in self._on_client_location_cache[facility_group]):
                self._on_client_location_cache[facility_group].append(X)
                self._on_client_location_cache[facility_group] = sorted(self._on_client_location_cache[facility_group])
            elif self._V[X] == 0 and (X in self._on_client_location_cache[facility_group]):
                self._on_client_location_cache[facility_group].remove(X)
        else:
            facility_group = (X - (self._n * self._k)) // self._n
            if self._V[X] == 1 and (X not in self._on_facility_group_cache[facility_group]):
                self._on_facility_group_cache[facility_group].append(X)
                self._on_facility_group_cache[facility_group] = sorted(self._on_facility_group_cache[facility_group])
            elif self._V[X] == 0 and (X in self._on_facility_group_cache[facility_group]):
                self._on_facility_group_cache[facility_group].remove(X)

            location_group = X % self._n
            if self._V[X] == 1 and (X not in self._on_facility_location_cache[location_group]):
                self._on_facility_location_cache[location_group].append(X)
                self._on_facility_location_cache[location_group] = sorted(self._on_facility_location_cache[location_group])
            elif self._V[X] == 0 and (X in self._on_facility_location_cache[location_group]):
                self._on_facility_location_cache[location_group].remove(X)

    def accept(self, delta, temperature):
        """
        Flip based on temperature
        """
        # if our delta is 0, then our probability becomes 50%, we skip in this case
        if math.isclose(delta, 0.0, abs_tol=0.01):
            return False


        # We have to watch out for overflow errors with the (Best-value)/T power
        try:
            e_term = np.exp(float(delta) / float(temperature))
            #e_term = torch.exp(torch.FloatTensor([float(delta) / float(temperature)]))
        except OverflowError:
            # In this case the e_term will cause the probability to essentially be 0
            print("Overflow")
            return False

        probability = 1 / (1 + e_term)
        return random.random() < probability


    def calculate_unit_value(self, X):
        # Client receiving Facility values
        if X < self._n * self._k:
            return self.calculate_client_unit_value(X)
        # Facility receiving Client values
        else:
            return self.calculate_facility_unit_value(X)

    def calculate_facility_unit_value(self, X):
        # Try the cache first; check if the on values for the connected clients group have changed;
        # if not, use the cached value
        facility_group = (X - (self._n * self._k)) // self._n
        client_on_values = self._on_client_location_cache[facility_group]
        cached_on_clients, cached_unit_value = self._facility_unit_value_cache[X]
        if cached_on_clients == client_on_values:
            return cached_unit_value
        # Calculate the unit value and cache it
        else:
            # get the clients that connect to this location. Note they are not located in V in sequence.
            if len(client_on_values) == 0:
                unit_value = 0.0
            else:
                on_to_indices = np.array(client_on_values) // self._k
                location = X % self._n
                unit_value = np.sum(self._distances[location, on_to_indices])

            # cache the value before returning
            self._facility_unit_value_cache[X] = (client_on_values.copy(), unit_value)
            return unit_value

    def calculate_client_unit_value(self, X):
        # Try the cache first; check if the on values for the connected facility group have changed;
        # if not, use the cached value
        facility_group = X % self._k
        facility_on_values = self._on_facility_group_cache[facility_group]
        cached_on_facilities, cached_unit_value = self._client_unit_value_cache[X]
        if cached_on_facilities == facility_on_values:
            return cached_unit_value
        # Calculate the unit value and cache it
        else:
            # TODO - might be able to use the on values directly here
            facility_start = (self._n * self._k) + (facility_group * self._n)
            facility_end = facility_start + self._n
            facility_values = self._V[facility_start:facility_end]
            client_location = X // self._k
            distances = self._distances[client_location, :]
            unit_value = np.sum(facility_values * distances)
            # cache the value before returning
            self._client_unit_value_cache[X] = (facility_on_values.copy(), unit_value)
            return unit_value

    def get_unit_group(self, unit):
        """
        Helper function to return a unit's competing group with itself included.
        :param unit: The unit whose group we are getting.
        :return: The competing group including the unit.
        """
        unit_group = self.group[unit]
        unit_group.append(unit)
        return sorted(unit_group)

    def get_best_group_member(self, group):
        selected = group[0]
        best = self.calculate_unit_value(group[0])
        for unit in group[1:]:
            value = self.calculate_unit_value(unit)
            if value < best:
                selected = unit
                best = value
        return selected

    def get_on_group(self, X):
        """
        Generate a set of units that are connected with X and have a value of 1.
        :param X: The unit to build the ON set for.
        :return: A set of unit indices representing units connected to X with a value of 1.
        """
        # if the unit is a client, select competitors from the first half of V
        if X < self._n * self._k:
            """
            client_group = X // self.k
            client_start = self.k * client_group
            client_end = client_start + self.k  # End value; Do not include
            on_members = np.flatnonzero(self.V[client_start:client_end])
            on_members = np.add(on_members, client_start)
            #on_members = torch.nonzero(self.V[client_start:client_end])
            #on_members = torch.add(on_members, client_start)
            return on_members #torch.flatten(on_members)
            """
            client_group = X // self._k
            return self._on_client_cache[client_group]
        # else select from the second half of V
        else:
            """
            facility_group = (X - (self.n * self.k)) // self.n
            facility_start = (self.n * self.k) + (self.n * facility_group)
            facility_end = facility_start + self.n  # End value; Do not include
            on_members = np.flatnonzero(self.V[facility_start:facility_end])
            on_members = np.add(on_members, facility_start)
            #on_members = torch.nonzero(self.V[facility_start:facility_end])
            #on_members = torch.add(on_members, facility_start)

            # additionally, add competitors that use the same location from other facility groups
            location = (X - (self.n * self.k)) % self.n
            location_start = (self.n * self.k) + (location)
            location_indices = [i for i in range(location_start, self.total_vertices, self.n)]
            np_location_indices = np.array(location_indices)
            on_location_indices = np.flatnonzero(self.V[location_indices])
            location_members = np_location_indices[on_location_indices]

            full_on_members = np.concatenate((on_members, location_members), axis=0)
            # Make sure the elements are unique as the two arrays could share members
            full_on_members = np.unique(full_on_members, axis=0)
            return full_on_members
            #return on_members #torch.flatten(on_members)
            """
            facility_group = (X - (self._n * self._k)) // self._n
            facility_on = self._on_facility_group_cache[facility_group]

            location = (X - (self._n * self._k)) % self._n
            location_on = self._on_facility_location_cache[location]

            full_on = facility_on + location_on
            # convert to a set to ensure elements are unique
            full_on = set(full_on)
            # reconvert to a list as that's what is expected
            return list(full_on)


    def is_valid(self):
        """
        Helper function that determines if all neurons have been assigned in a correct fashion
        e.g., 1 client neuron per client, 1 facility per location, ...
        """

        # Check that there is exactly one active neuron per client group
        for i in range(self._n):
            client_start = i * self._k
            client_end = client_start + self._k
            total = np.sum(self._V[client_start:client_end])
            #total = torch.sum(self.V[client_start:client_end])
            if int(total) != 1:
                return False

        # Check that there is exactly one active neuron per facility group
        for i in range(self._k):
            facility_start = (self._n * self._k) + (i * self._n)
            facility_end = facility_start + self._n
            total = np.sum(self._V[facility_start:facility_end])
            #total = torch.sum(self.V[facility_start:facility_end])
            if int(total) != 1:
                return False

        # Check that each facility is assigned to a unique location
        used_locations = set()
        for i in range(self._k):
            facility_start = (self._n * self._k) + (i * self._n)
            facility_end = facility_start + self._n
            on_facilities = np.flatnonzero(self._V[facility_start:facility_end])
            if len(on_facilities) != 1:
                return False

            on_facility = on_facilities.item(0)
            if on_facility in used_locations:
                return False
            else:
                used_locations.add(on_facility)


        return True

    def get_best_on_group_member(self, group):
        """
        Helper function to return the best 'ON' member of a group.
        :param group: The group to check.
        :return: The index of the best 'ON' unit.
        """
        assert (sum(self._V[unit] for unit in group) >= 1)
        selected = None
        best = None
        for unit in group:
            if self._V[unit] == 1:
                value = self.calculate_unit_value(unit)
                if best is None or value < best:
                    selected = unit
                    best = value
        return selected

    def convert_facility_groups_to_location_groups(self, facility_groups):
        """
        Helper function to transpose the facility groups 2D array in order to make a group based around a
        single location.
        :param facility_groups: The facilities groups that will be converted into location groups.
        :return: A 2D array representing facilities competing for one location.
        """
        location_groups = [[] for _ in range(self._n)]
        for i in range(self._n):
            for j in range(self._k):
                location_groups[i].append(facility_groups[j][i])
        return location_groups

    def get_facility_location_count(self, facility_groups):
        location_counts = [0 for _ in range(self._n)]
        # Count how many facilities are in each location
        for group in facility_groups:
            activated_variables = self._V[group[0]:group[0] + len(group)]
            location_counts = np.add(location_counts, activated_variables)
        return location_counts

    def generate_groups(self, n, k):
        """
        Create the group[X] sets. Here each unit is grouped with competing units
        with the CF group defined by {CF[i,t] | t in 1...k} and the FL group defined
        by {FL[j,t] | Vt in V}.
        :param n: The number of locations.
        :param k: The number of facilities to place.
        :return: None
        """
        # Get the competing groups for CF units
        for i in range(n):
            for j in range(k):
                X = (k * i) + j
                for j_prime in range(k):
                    if j_prime != j:
                        competitor = (k * i) + j_prime
                        self.group[X].append(competitor)

        # Get the competing groups for FL units
        for j in range(k):
            for s in range(n):
                X = (n * k) + s + (n * j)
                for s_prime in range(n):
                    if s_prime != s:
                        competitor = (n * k) + s_prime + (n * j)
                        self.group[X].append(competitor)

    def generate_weights(self, n, k):
        """
        Connect Each CF[i,j] unit with a FL[j,s] unit. The edge weight is the
        distance between the two locations.
        :param n: The number of locations.
        :param k: The number of facilities to place.
        :return: None
        """
        for i in range(n):
            for j in range(k):
                cf = (k * i) + j
                for s in range(n):
                    fl = (n * k) + s + (n * j)
                    self.con[cf][str(fl)] = self._G.get_standard_distance(i, s)
                    self.con[fl][str(cf)] = self._G.get_standard_distance(i, s)

    def cf(self, i, j):
        """
        Calculate the expression CF[i,j] which denotes if a client i is being
        served by a facility j.
        :param i: The index of the client.
        :param j: The index of the facility.
        :return: 1 or 0 if the binary variable representing this connection is
                 activated.
        """
        variable_index = (i * self._k) + j
        return self._V[variable_index]

    def fl(self, j, s):
        """
        Calculate the expression FL[j,s] which denotes if a facility j is in location
        s.
        The binary variables for location are stored in the back half of the array and
        separated into j groups.
        :param j: The index of the facility.
        :param s: The index of the location.
        :return: 1 or 0 if the binary variable representing this connection is activated.
        """
        variable_index = s + (self._n * j) + int(self._total_vertices / 2)
        return self._V[variable_index]


class HaralampievSolutionChecker:
    """
    This class accepts an invalid solution from a Haralampiev network and corrects it such that it becomes
    a valid solution.
    """

    def __init__(self, haralampiev_network):
        self._network = haralampiev_network

    def check(self):
        # check facilities first
        facility_groups = [self._network.get_unit_group(unit) for unit in
                           range((self._network.n * self._network.k), self._network.total_vertices, self._network.n)]
        self.select_best_facility_in_group(facility_groups)
        self.remove_duplicate_facility_locations(facility_groups)

        # check client groups
        client_groups = [self._network.get_unit_group(unit) for unit in
                         range(0, (self._network.n * self._network.k), self._network.k)]
        for client_group in client_groups:
            # check that each client group has only one connection
            if sum(self._network.V[unit] for unit in client_group) != 1:
                selected = self._network.get_best_group_member(client_group)
                for unit in client_group:
                    self._network.V[unit] = 1 if unit == selected else 0

            # check that each client group is connected to a facility
            if sum(self._network.V[unit] for unit in client_group) == 0:
                selected = self._network.get_best_group_member(client_group)
                for unit in client_group:
                    self._network.V[unit] = 1 if unit == selected else 0

    def select_best_facility_in_group(self, facility_groups):
        # For groups with too many or no facilities, choose the location with the best value
        for group in facility_groups:
            # No facilities placed
            if sum(self._network.V[unit] for unit in group) == 0:
                selected = self._network.get_best_group_member(group)
            # 1 or more facilities placed
            else:
                selected = self._network.get_best_on_group_member(group)
            for unit in group:
                self._network.V[unit] = 1 if unit == selected else 0

    def remove_duplicate_facility_locations(self, facility_groups):
        location_groups = self._network.convert_facility_groups_to_location_groups(facility_groups)
        # get all open facility spots
        open_locations = []
        for location, location_group in enumerate(location_groups):
            if sum(self._network.V[unit] for unit in location_group) == 0:
                open_locations.append(location)

        # keep the best facility on each location
        for group in location_groups:
            # if more than 1 facility, keep the best
            num_facilities_on_location = sum(self._network.V[unit] for unit in group)
            if num_facilities_on_location > 1:
                # find the best 'on' group member and turn off the rest
                selected = self._network.get_best_on_group_member(group)
                for unit in group:
                    if unit != selected:
                        self._network.V[unit] = 0

        # for empty facility-groups, place them on the best open spot that they have access to
        for facility_group in facility_groups:
            if sum(self._network.V[unit] for unit in facility_group) == 0:
                # turn on all potential facilities and then select the best
                for i in open_locations:
                    unit = facility_group[i]
                    self._network.V[unit] = 1
                selected = self._network.get_best_on_group_member(facility_group)
                for unit in facility_group:
                    self._network.V[unit] = 0 if unit != selected else 1
                # remove the open location from the list
                open_locations.remove(facility_group.index(selected))
    
# I noticed it was only used by this solver, so I assumed it is best fit here
class RandomChoiceAlgorithm(KMPSolver):
    """
    This class exists for two purposes:
    1. See if an approximation is doing well or just appearing to do well.
    2. This class runs quickly, so it can be used to test that datasets have been constructed properly.
    """

    def name(self):
        return "Random Choice"

    def run(self, graph, n, k, solution=None):
        selection = random.sample([i for i in range(n)], k=k)
        return selection