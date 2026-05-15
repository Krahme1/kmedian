import csv
import os
import re
import time

import numpy as np

from solvers_alg.solvers.brute_solver import (calculate_capacitated_distance,
                                  calculate_distance,
                                  calculate_distance_with_facility_cost,
                                  calculate_radius)
from solvers_alg.KMP.AryaMultiSolver import AryaMultiSolver
from solvers_alg.KMP.CohenAddadMultiSolver import CohenAddadMultiSolver
from solvers_alg.KMP.CohenAddadSolver import CohenAddadSolver
from solvers_alg.KMP.DominguezAlgorithmSolver import DominguezAlgorithmSolver
from solvers_alg.KCP.DropWorstFacilityKCenterSolver import \
    DropWorstFacilityKCenterSolver
from solvers_alg.KCP.FarthestClientReassignmentKCenterSolver import \
    FarthestClientReassignmentKCenterSolver
from solvers_alg.KCP.FarthestFirstKCenterSolver import FarthestFirstKCenterSolver
from solvers_alg.KCP.GreedyAddRemoveKCenterSolver import \
    GreedyAddRemoveKCenterSolver
from solvers_alg.KMP.HaralampievAlgorithmSolver import HaralampievAlgorithmSolver
from solvers_alg.KMP.HopfieldAlgorithmSolver import HopfieldAlgorithmSolver
from solvers_alg.KMP.HopfieldBestHalfMultiAlgorithmSolver import \
    HopfieldBestHalfMultiSolver
from solvers_alg.KMP.HopfieldBestHalfSecondClosestAlgorithmSolver import \
    HopfieldBestHalfSecondClosestAlgorithmSolver
from solvers_alg.KMP.HopfieldBestHalfSingleSolver import \
    HopfieldBestHalfSingleSolver
from solvers_alg.KMP.HopfieldExhaustiveAlgorithmSolver import \
    HopfieldExhaustiveAlgorithmSolver
from solvers_alg.CKMP.HopfieldOriginal2nkCKMPSolver import \
    HopfieldOriginal2nkCKMPSolver
from solvers_alg.KMP.HopfieldOriginal2nkSolver import HopfieldOriginalSolver
from solvers_alg.CKMP.HopfieldParallelCKMSolver import HopfieldParallelCKMSolver
from solvers_alg.KCP.HopfieldParallelKCPSolver import HopfieldParallelKCPSolver
from solvers_alg.KF.HopfieldParallelKFSolver import HopfieldParallelKFSolver
from solvers_alg.KMP.HopfieldSecondParallelSolver import \
    HopfieldSecondParallelSolver
from solvers_alg.KMP.HopfieldThirdParallelSolver import HopfieldThirdParallelSolver
from solvers_alg.KMP.InterchangeAlgorithmSolver import InterchangeAlgorithmSolver
from solvers_alg.KCP.LocalRecenterKCenterSolver import LocalRecenterKCenterSolver
from solvers_alg.KMP.LocalSearchSolver import LocalSearchSolver
from solvers_alg.KCP.LocalSearchSolverKCenter import LocalSearchSolverKCenter
from solvers_alg.KMP.PAMSolver import PAMSolver
from solvers_alg.KCP.RandomizedSwapKCenterSolver import RandomizedSwapKCenterSolver
from solvers_alg.KF.SameiSolisObaKFSolver import SameiSolisObaKFSolver
from solvers_alg.KMP.ZhuAlgorithmSolver import ZhuAlgorithmSolver


class ExperimentManager():
    def __init__(self, problems, solver, problem_family, num_runs=None):
        self._problems = problems
        self._solver = solver
        self._problem_family = problem_family
        self._latest_results = []
        self._num_runs = 1 if num_runs is None else num_runs

    def run(self, dataset_key):
        self._latest_results = []

        if dataset_key in ["4", "5"]:
            # Case 1: list of test sets
            self._run_from_list(dataset_key)
        elif dataset_key in ["1", "2", "3"]:
            # Case 2: dict of directories (e.g., {"pmed": tests_object})
            self._run_from_directory_dict(dataset_key)
        elif dataset_key == "6":
            self._run_special(dataset_key)
        else:
            raise TypeError("Did not pass a list of problems to Experiment Manage")

        # Save results
        self._save_results_to_csv(self._latest_results, dataset_key)

    def _run_single_test(self, problem):
        """Helper function to run a single problem instance multiple times."""
        optimal_distance = problem.getOptimal()

        ratios = []
        minRatio = 100.000
        maxRatio = 0.000
        
        times = []
        minTime = 99999.999
        maxTime = 0.000
        
        # Initialize the solver for the *current* problem instance
        self._solver.initialize(problem)

        for iteration in range(self._num_runs):
            start_time = time.time()
            self._solver.solve(iteration)
            facilities = self._solver.getSelectedFacilities()
            end_time = time.time()
            elapsedTime = end_time - start_time
            times.append(elapsedTime)

            # if self._problem_family == "1":  # k-median
            #     verify_kmedian_solution(
            #         graph=problem.getGraph(),
            #         selected_facilities=facilities,
            #         k=problem.getK(),
            #         reported_cost=self._solver.getSolutionValue()
            #     )

            # elif self._problem_family == "2":  # k-center
            #     verify_kcenter_solution(
            #         graph=problem.getGraph(),
            #         selected_facilities=facilities,
            #         k=problem.getK(),
            #         reported_radius=self._solver.getSolutionValue()
            #     )

            if elapsedTime > maxTime:
                maxTime = elapsedTime
            if elapsedTime < minTime:
                minTime = elapsedTime
            
            approximationRatio = self._solver.getSolutionValue() / optimal_distance
            ratios.append(approximationRatio)

            if approximationRatio > maxRatio:
                maxRatio = approximationRatio
            if approximationRatio < minRatio:
                minRatio = approximationRatio
                
        # Calculate stats for the run
        average_ratio = sum(ratios) / len(ratios)
        average_time = sum(times) / len(times)
        standard_deviation_ratio = np.std(ratios, dtype=np.float64)
        standard_deviation_time = np.std(times, dtype=np.float64)
        
        return (problem.getName(), problem.getN(), problem.getK(), 
                minRatio, average_ratio, maxRatio, minTime, average_time, 
                maxTime, standard_deviation_ratio, standard_deviation_time)


    def _run_from_list(self, dataset_key):
        for problem in self._problems:
            test_name = problem.getName()
            n = problem.getN()
            k = problem.getK()
            try:
                # 1. Run the test and get the results tuple
                # This calls the solver initialization once per problem instance
                result_tuple = self._run_single_test(problem)
                
                # 2. Append the result
                self._latest_results.append(result_tuple)
                
                print(f"n={n} k={k} Completed")
            except Exception as e:
                print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
            finally:
                # 3. Explicitly delete the local reference to the problem object
                # This frees up the memory for the large graph/tensors sooner.
                del problem
    

    def _run_from_directory_dict(self, dataset_key):
        for key, problem_list in self._problems.items():
            for problem in problem_list:
                test_name = problem.getName()
                n = problem.getN()
                k = problem.getK()
                try:
                    # 1. Run the test and get the results tuple
                    # Unpack the result, ignoring the name since it's not needed for this format
                    _, n, k, minR, avgR, maxR, minT, avgT, maxT, stdR, stdT = self._run_single_test(problem)
                    
                    # 2. Append the result (excluding the name which is not used here)
                    self._latest_results.append((n, k, minR, avgR, maxR, minT, avgT, maxT, stdR, stdT))
                    
                    print(f"n={n} k={k} Completed")
                except Exception as e:
                    print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
                finally:
                    # 3. Explicitly delete the local reference to the problem object
                    del problem
    
    def _run_special(self, dataset_key):
        for problem in self._problems:
            test_name = problem.getName()
            n = problem.getN()
            k = problem.getK()
            try:
                optimal_distance = problem.getOptimal()
                self._solver.initialize(problem)

                start_time = time.time()
                self._solver.solve()
                facilities = self._solver.getSelectedFacilities()
                end_time = time.time()
                total_time = end_time - start_time

                distance = self._solver.getSolutionValue()
                
                ratio = distance / optimal_distance
                self._latest_results.append((test_name, n, k, ratio, total_time, distance))
                print(f"n={n} k={k} Completed")
            except Exception as e:
                print(f"⚠️ Skipped {test_name} (n={n}, k={k}): {type(e).__name__}: {e}")
            finally:
                # Add explicit deletion here as well for consistency
                del problem


    def _save_results_to_csv(self, results, dataset_key, output_path=None, announce=True):
        if len(results) == 0:
            return

        if dataset_key in ["4", "5", "6"]:
            # Original value
            full_name = results[0][0]
            # Keep only the letters at the start
            match = re.match(r"[a-zA-Z]+", full_name)
            first_name = match.group(0) if match else "dataset"
            # Replace spaces just in case (optional)
            first_name = first_name.replace(" ", "_")
            output_filename = f"results_{first_name}.csv"
            has_name = True
        elif dataset_key in ["1", "2", "3"]:
            # Use timestamp if no names
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"results_{timestamp}.csv"
            has_name = False

        # Get the directory where this file (ExperimentManager) is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if output_path is None:
            output_path = os.path.join(base_dir, output_filename)

        # Build appropriate header
        if has_name and dataset_key in ["4", "5"]:
            header = [
                "Instance", "N", "K", "Min Ratio", "Average Ratio", "Max Ratio",
                "Min Time", "Average Time", "Max Time",
                "Standard Deviation (Ratio)", "Standard Deviation (Time)"
            ]
        elif dataset_key == "6":
            header = [
                "Instance", "N", "K", "Ratio", "Total Time", "Distance"
        ]
        else:
            header = [
                "N", "K", "Min Ratio", "Average Ratio", "Max Ratio",
                "Min Time", "Average Time", "Max Time",
                "Standard Deviation (Ratio)", "Standard Deviation (Time)"
            ]

        results.sort(key=lambda r: ((r[1], r[2]) if len(r) == 11 else (r[0], r[1])))

        # Write the CSV
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for r in results:
                if dataset_key in ["4", "5"]:
                    name, n, k, min_ratio, avg_ratio, max_ratio, min_time, avg_time, max_time, std_ratio, std_time = r
                    writer.writerow([
                        name, n, k,
                        f"{min_ratio:.3f}", f"{avg_ratio:.3f}", f"{max_ratio:.3f}",
                        f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}",
                        f"{std_ratio:.3f}", f"{std_time:.3f}"
                    ])
                elif dataset_key == "6":
                    writer.writerow([
                        r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}"
                ])
                else:
                    n, k, min_ratio, avg_ratio, max_ratio, min_time, avg_time, max_time, std_ratio, std_time = r
                    writer.writerow([
                        n, k,
                        f"{min_ratio:.3f}", f"{avg_ratio:.3f}", f"{max_ratio:.3f}",
                        f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}",
                        f"{std_ratio:.3f}", f"{std_time:.3f}"
                    ])

        if announce:
            print(f"\n✅ Results saved to {output_path}")



import math


def verify_selected_facilities(selected_facilities, k):
    """
    Basic checks common to k-median and k-center.
    """
    if len(selected_facilities) != k:
        raise ValueError(
            f"Wrong number of selected facilities: got {len(selected_facilities)}, expected {k}"
        )

    if len(set(selected_facilities)) != k:
        raise ValueError(
            f"Duplicate facilities detected: {selected_facilities}"
        )


def recompute_kmedian_cost(graph, selected_facilities):
    """
    Independently recompute the uncapacitated k-median objective.

    For each client node j:
        cost += min distance from j to any selected facility
    """
    distance_matrix = graph._distances
    n = distance_matrix.shape[0]

    total_cost = 0.0

    for client in range(n):
        best_distance = math.inf
        for facility in selected_facilities:
            d = float(distance_matrix[facility, client])
            if d < best_distance:
                best_distance = d
        total_cost += best_distance

    return total_cost


def recompute_kcenter_radius(graph, selected_facilities):
    """
    Independently recompute the k-center objective.

    For each client node j:
        nearest = min distance from j to any selected facility

    Return:
        max_j nearest(j)
    """
    distance_matrix = graph._distances
    n = distance_matrix.shape[0]

    radius = 0.0

    for client in range(n):
        best_distance = math.inf
        for facility in selected_facilities:
            d = float(distance_matrix[facility, client])
            if d < best_distance:
                best_distance = d
        radius = max(radius, best_distance)

    return radius


def verify_kmedian_solution(graph, selected_facilities, k, reported_cost, tolerance=1e-6):
    """
    Verify a k-median solution independently.
    """
    verify_selected_facilities(selected_facilities, k)

    recomputed_cost = recompute_kmedian_cost(graph, selected_facilities)
    difference = abs(recomputed_cost - reported_cost)

    print("=== K-MEDIAN VERIFICATION ===")
    print("Selected facilities:", selected_facilities)
    print("Facility count:", len(selected_facilities))
    print("Unique facility count:", len(set(selected_facilities)))
    print("Reported cost:", reported_cost)
    print("Recomputed cost:", recomputed_cost)
    print("Difference:", difference)

    if difference > tolerance:
        raise ValueError(
            f"k-median verification failed: reported={reported_cost}, "
            f"recomputed={recomputed_cost}, diff={difference}"
        )

    print("Verification passed.")
    return recomputed_cost


def verify_kcenter_solution(graph, selected_facilities, k, reported_radius, tolerance=1e-6):
    """
    Verify a k-center solution independently.
    """
    verify_selected_facilities(selected_facilities, k)

    recomputed_radius = recompute_kcenter_radius(graph, selected_facilities)
    difference = abs(recomputed_radius - reported_radius)

    print("=== K-CENTER VERIFICATION ===")
    print("Selected facilities:", selected_facilities)
    print("Facility count:", len(selected_facilities))
    print("Unique facility count:", len(set(selected_facilities)))
    print("Reported radius:", reported_radius)
    print("Recomputed radius:", recomputed_radius)
    print("Difference:", difference)

    if difference > tolerance:
        raise ValueError(
            f"k-center verification failed: reported={reported_radius}, "
            f"recomputed={recomputed_radius}, diff={difference}"
        )

    print("Verification passed.")
    return recomputed_radius
    print("Verification passed.")
    return recomputed_radius