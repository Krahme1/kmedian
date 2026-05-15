"""
This script runs an algorithm handler on the p-med dataset.
"""
import json
import time

from algorithms import algorithm_interface
from solvers_alg.solvers.brute_solver import calculate_distance
from utils.graph import DistanceGraph
from utils.seeding import seed_all

if __name__ == '__main__':
    # Seed the RNGs.
    seed_all()

    # instance = algorithm_interface.GridHopfieldAlgorithm(1)
    #instance = algorithm_interface.SearchingGridHopfieldAlgorithm(3)
    # instance = algorithm_interface.GridAndLocalSearchAlgorithm(max_time=10, runs=1)
    #instance = algorithm_interface.LocalSearchAlgorithm(max_time=10, epsilon=0.01)
    #instance = algorithm_interface.RandomChoiceAlgorithm()
    instance = algorithm_interface.ILPSolverAlgorithm()
    """
    instance = algorithm_interface.SequentialCombinationAlgorithm(
        [
            algorithm_interface.SearchingGridHopfieldAlgorithm(3),
            algorithm_interface.LocalSearchAlgorithm(max_time=30, epsilon=0.01)
        ])
    """

    print("Running the special dataset with:", instance.name())
    print()

    # Print the csv headers
    print("test,ratio,time,distance")

    # 40 test files in the pmed dataset
    for i in range(1, 3):
        with open(f'tests/special{i}.json', 'r') as f:
            test_data = json.loads(f.read())

        n = test_data['n']
        k = test_data['k']
        distances = test_data['distances']
        graph = DistanceGraph(distances)
        optimal_distance = test_data['distance']

        start_time = time.time()
        facilities = instance.run(graph, n, k)
        end_time = time.time()

        distance = calculate_distance(graph, facilities, n)
        ratio = distance / optimal_distance
        print(f"special{i},{ratio},{end_time - start_time},{distance}")
