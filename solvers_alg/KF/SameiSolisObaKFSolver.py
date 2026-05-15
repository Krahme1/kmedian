import itertools

from problems.KFProblem import KFProblem
from solvers_alg.solvers.brute_solver import calculate_distance_with_facility_cost
from solvers_alg.KF.KFSolver import KFSolver


class SameiSolisObaKFSolver(KFSolver):
    def __init__(self, swap_size=1, random_seed=None):
        self._name = "Samei-Solis-Oba KF Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._k = None
        self._graph = None
        self._costs = None

        self._swap_size = swap_size
        self._random_seed = random_seed

    def initialize(self, problem: KFProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        self._costs = problem.getCosts()

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        if self._costs is None:
            raise ValueError("Facility costs must be set before calling initialize().")

        if self._swap_size < 1:
            raise ValueError("swap_size must be at least 1.")

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

    def setCosts(self, costs):
        self._costs = costs

    def solve(self, runNum=None):
        """
        Paper-style implementation:

        1. S <- a random set of k facilities
        2. for i = 1 to k:
              S' <- a random subset of i facilities
              while there exists an improving multi-swap for S':
                  S' <- (S' \\ A) U B
              if cost(S) > cost(S'):
                  S <- S'
        3. return S
        """

        if self._random_seed is not None:
            import random
            random.seed(self._random_seed)

        # Best-so-far solution S starts with a random set of k facilities
        best_overall_facilities = self._random_initialize(self._k)
        best_overall_value = calculate_distance_with_facility_cost(
            self._graph, best_overall_facilities, self._costs, self._n
        )

        print("Initial best distance (size k):", best_overall_value)

        # For each i = 1, ..., k
        for i in range(1, self._k + 1):
            print(f"\nRunning local search for size {i}")

            current_facilities = self._random_initialize(i)
            current_value = calculate_distance_with_facility_cost(
                self._graph, current_facilities, self._costs, self._n
            )

            improved = True
            iterations = 0

            while improved:
                improved = False
                iterations += 1

                open_set = set(current_facilities)
                closed_facilities = [j for j in range(self._n) if j not in open_set]

                best_local_value = current_value
                best_local_facilities = None

                max_swap_size = min(self._swap_size, len(current_facilities), len(closed_facilities))

                # Search all improving multi-swaps up to max_swap_size
                for s in range(1, max_swap_size + 1):
                    for facilities_out in itertools.combinations(current_facilities, s):
                        for facilities_in in itertools.combinations(closed_facilities, s):
                            trial_facilities = [f for f in current_facilities if f not in facilities_out]
                            trial_facilities.extend(facilities_in)

                            trial_value = calculate_distance_with_facility_cost(
                                self._graph, trial_facilities, self._costs, self._n
                            )

                            if trial_value < best_local_value:
                                best_local_value = trial_value
                                best_local_facilities = list(trial_facilities)

                if best_local_facilities is not None:
                    current_facilities = list(best_local_facilities)
                    current_value = best_local_value
                    improved = True

                if improved:
                    print("Iteration", iterations, "distance:", current_value)

            print(f"Local optimum for size {i}: {current_value}")

            if current_value < best_overall_value:
                best_overall_facilities = list(current_facilities)
                best_overall_value = current_value

        self._selectedFacilities = list(best_overall_facilities)
        self._solutionValue = best_overall_value

        print("\nBest overall solution:")
        print("Selected facilities:", self._selectedFacilities)
        print("Distance:", self._solutionValue)

    def _random_initialize(self, num_facilities):
        """
        Random initialization for any requested solution size.
        The Samei-Solis-Oba algorithm only requires "any" starting set, so a
        random subset is a valid starting point for the local search benchmark.
        """
        import random

        if num_facilities < 1:
            raise ValueError("num_facilities must be at least 1.")

        if num_facilities > self._n:
            raise ValueError("num_facilities cannot be larger than the number of facilities.")

        selected_facilities = random.sample(range(self._n), num_facilities)

        initial_value = calculate_distance_with_facility_cost(
            self._graph, selected_facilities, self._costs, self._n
        )
        print(f"Initial random distance for size {num_facilities}: {initial_value}")

        return selected_facilities

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._selectedFacilities)
        selected_distance = calculate_distance_with_facility_cost(
            self._graph, selected_facilities, self._costs, self._n
        )
        return selected_facilities, selected_distance