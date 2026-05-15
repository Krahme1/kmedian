"""
This script runs an algorithm handler on the tsp dataset.
"""
import cProfile
import json
import time

import torch

from algorithms import algorithm_interface
from datasets.tsplib.load_tsplib_test import load_test
from solvers_alg.solvers.brute_solver import calculate_distance
from utils.graph import DistanceGraph, CoordinateGraph
from utils.seeding import seed_all

# Using the list of tests that Garcia et al used in:
# Solving Large p-Median Problems with a Radius Formulation
TEST_LIST = [
    'rl1304',
    'fl1400',
    'u1432',
    'vm1748',  # appears to be a typo in the original paper
    'd2103',
    'pcb3038',
    'fl3795',
    'rl5934',
    'usa13509',
    # 'sw24978', not in the bulk download, will have to look for - looks like this https://www.math.uwaterloo.ca/tsp/world/swpoints.html
    # 'ch71009', also appears to be missing - looks like https://www.math.uwaterloo.ca/tsp/world/chpoints.html
    'pla85900'
]

if __name__ == '__main__':
    # Seed the RNGs.
    seed_all()

    torch.set_default_dtype(torch.bfloat16)

    # instance = algorithm_interface.GridHopfieldAlgorithm(1)
    #search_config = {
    #    'epsilon': 0.1,
    #    'exclude': bool(True),
    #    'fixed_size': bool(False)
    #}
    #instance = algorithm_interface.SearchingGridHopfieldAlgorithm(1, search_config=search_config, use_gpu=False)
    #instance = algorithm_interface.LocalSearchAlgorithm(max_time=60)
    #instance = algorithm_interface.RandomChoiceAlgorithm()
    #instance = algorithm_interface.ILPSolverAlgorithm()
    instance = algorithm_interface.ILPSolverAlgorithm()
    """
    instance = algorithm_interface.SequentialCombinationAlgorithm(
        [
            algorithm_interface.SearchingGridHopfieldAlgorithm(3),
            algorithm_interface.LocalSearchAlgorithm(max_time=30, epsilon=0.01)
        ])
    """
    test_name = "pcb3038"
    #k_values = [300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
    #k_values = [300]
    k_values = [5, 10, 20, 50, 100, 200, 300, 400, 500]

    output_string = ""

    for k in k_values:

        filepath = f"tests/{test_name}_k{k}.json"
        with open(filepath, 'r') as f:
            test_data = json.loads(f.read())

        x = test_data['x']
        y = test_data['y']
        optimal_distance = test_data['distance']
        n = test_data['n']
        graph = CoordinateGraph(x, y, use_gpu=False)

        start_time = time.time()
        #with cProfile.Profile() as pr:
        facilities = instance.run(graph, n, k)
        end_time = time.time()
        total_time = end_time - start_time

        distance = calculate_distance(graph, facilities, n)
        ratio = distance / optimal_distance
        print("K:", k)
        print("Total Time:", total_time)
        print("Distance:", distance)
        print("Ratio:", ratio)
        print()

        output_string += str(k) + " " + str(total_time) + " " + str(distance) + " " + str(ratio) + "\n"
        #pr.print_stats()

    with open("results.txt", "w") as output:
        output.write(output_string)



