import json
import os
import time

import numpy as np

from solvers_alg.solvers.brute_solver import calculate_distance
from utils.graph import DistanceGraph


def run_tests(dataset_folder_path, algorithm_instance, use_gpu):
    results = []
    # 40 test files in the pmed dataset
    for i in range(1, 41):
        filepath = os.path.join(dataset_folder_path, "pmed", "tests", f"pmed{i}.json")
        with open(filepath, 'r') as f:
            test_data = json.loads(f.read())

        n = test_data['n']
        k = test_data['k']
        distances = test_data['distances']
        graph = DistanceGraph(distances, use_gpu)
        optimal_distance = test_data['distance']
        
        ratios = []
        minRatio = 100.000
        maxRatio = 0.000
        
        times = []
        minTime = 99999.999
        maxTime = 0.000

        algorithm_instance.setN(n)
        algorithm_instance.setK(k)
        algorithm_instance.setGraph(graph)
        algorithm_instance.initialize()
        
        for iteration in range(10):
            
            start_time = time.time()
            algorithm_instance.solve()
            facilities = algorithm_instance.getSelectedFacilities()
            end_time = time.time()
            elapsedTime = end_time - start_time
            times.append(elapsedTime)
            
            if elapsedTime > maxTime:
                maxTime = elapsedTime
            if elapsedTime < minTime:
                minTime = elapsedTime

            distance = calculate_distance(graph, facilities, n)
            approximationRatio = distance / optimal_distance
            ratios.append(approximationRatio)
            
            if approximationRatio > maxRatio:
                maxRatio = approximationRatio
            if approximationRatio < minRatio:
                minRatio = approximationRatio
            
        average_ratio = sum(ratios) / len(ratios)
        average_time = sum(times) / len(times)
        standard_deviation_ratio = np.std(ratios, dtype=np.float64)
        standard_deviation_time = np.std(times, dtype=np.float64)
        test_name = f"pmed{i}"
        results.append((test_name, n, k, minRatio, average_ratio, maxRatio, minTime, average_time, maxTime, standard_deviation_ratio, standard_deviation_time))    
        #results.append((test_name, n, k, ratio, total_time, distance))
        print(f"n={n} k={k} Completed")

    return results