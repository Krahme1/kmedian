import itertools
import math
import random
import time

from problems.KFProblem import KFProblem
from solvers.brute_solver import calculate_distance_with_facility_cost
from solvers_alg.KF.KFSolver import KFSolver


class ZhangKFSolver(KFSolver):
    def __init__(self, swap_size=1, epsilon_prime=0.01, random_seed=None):
        self._name = "Zhang KF Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._k = None
        self._graph = None
        self._costs = None
        self._scaled_costs = None

        self._swap_size = swap_size
        self._epsilon_prime = epsilon_prime
        self._random_seed = random_seed

    def initialize(self, problem: KFProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        self._costs = problem.getCosts()

        p = self._swap_size
        delta = 1 + (1 / p) + math.sqrt(3 + (2 / p) + (1 / (p * p)))

        self._scaled_costs = [delta * cost for cost in self._costs]

    def getName(self):
        return self._name

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def solve(self, runNum=None):
        if self._random_seed is not None:
            random.seed(self._random_seed)

        start_time = time.time()

        current_facilities = random.sample(range(self._n), self._k)
        current_value = calculate_distance_with_facility_cost(
            self._graph, current_facilities, self._scaled_costs, self._n
        )

        q = self._n * self._n + self._n
        threshold_factor = 1 - (self._epsilon_prime / q)

        improved = True
        iteration = 0

        while improved:
            improved = False
            iteration += 1

            best_value = current_value
            best_facilities = None
            best_move = None

            current_set = set(current_facilities)
            closed_facilities = list(set(range(self._n)) - current_set)

            # ADD operation
            if len(current_facilities) < self._k:
                for facility in closed_facilities:
                    trial_facilities = list(current_set | {facility})
                    trial_value = calculate_distance_with_facility_cost(
                        self._graph, trial_facilities, self._scaled_costs, self._n
                    )

                    if trial_value < best_value:
                        best_value = trial_value
                        best_facilities = trial_facilities
                        best_move = f"add {facility}"

            # DROP operation
            if len(current_facilities) > 1:
                for facility in current_facilities:
                    trial_facilities = list(current_set - {facility})
                    trial_value = calculate_distance_with_facility_cost(
                        self._graph, trial_facilities, self._scaled_costs, self._n
                    )

                    if trial_value < best_value:
                        best_value = trial_value
                        best_facilities = trial_facilities
                        best_move = f"drop {facility}"

            # SWAP operation
            max_swap_size = min(
                self._swap_size,
                len(current_facilities),
                len(closed_facilities)
            )

            for size in range(1, max_swap_size + 1):
                for A in itertools.combinations(current_facilities, size):
                    for B in itertools.combinations(closed_facilities, size):
                        trial_facilities = list((current_set - set(A)) | set(B))
                        trial_value = calculate_distance_with_facility_cost(
                            self._graph, trial_facilities, self._scaled_costs, self._n
                        )

                        if trial_value < best_value:
                            best_value = trial_value
                            best_facilities = trial_facilities
                            best_move = f"swap out {list(A)} in {list(B)}"

            # Zhang threshold condition:
            # cost(S') <= (1 - epsilon'/q) cost(S)
            if best_facilities is not None and best_value <= threshold_factor * current_value:
                current_facilities = list(best_facilities)
                current_value = best_value
                improved = True

                print(
                    "Iteration",
                    iteration,
                    "move:",
                    best_move,
                    "scaled distance:",
                    current_value
                )

        self._selectedFacilities = list(current_facilities)

        self._solutionValue = calculate_distance_with_facility_cost(
            self._graph, self._selectedFacilities, self._costs, self._n
        )

        self._runningTime = time.time() - start_time

        print("Final Zhang selected facilities:", self._selectedFacilities)
        print("Final Zhang original distance:", self._solutionValue)