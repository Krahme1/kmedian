import csv
import json
import os
import time

import numpy as np

from solvers_alg.solvers.brute_solver import calculate_distance
from utils.graph import CoordinateGraph, DistanceGraph


def run_tests(dataset_folder_path, algorithm_instance, use_gpu):
    # Get the value data from the CSV file
    # The format will be: test name, n, k, optimal (or best) value
    test_values = []
    csv_path = os.path.join(dataset_folder_path, "tsplib", "test_values.csv")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            test_values.append(row)

        # remove the header row
        test_values.pop(0)

    results = []
    # 40 test files in the pmed dataset
    for test in test_values:
        test_name, n, k, optimal_distance = test
        n = int(n)
        k = int(k)
        #if n != 13509:
        #    continue
        #if k < 3000: 
        #    continue
        #if k == 4000:
        #    continue
        #if n < 13509:
            #continue
        #if n != 1432:
        #    continue
        #if k != 50:
        #    continue
        optimal_distance = int(optimal_distance)

        # some names appear to have typos:
        if test_name == 'v1748':
            test_name = 'vm1748'

        filepath = os.path.join(dataset_folder_path, "tsplib", "tests", f"{test_name}_k{k}.json")
        with open(filepath, 'r') as f:
            test_data = json.loads(f.read())

        x = test_data['x']
        y = test_data['y']
        graph = CoordinateGraph(x, y, use_gpu)
        
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
        test_name = f"{test_name}"
        results.append((test_name, n, k, minRatio, average_ratio, maxRatio, minTime, average_time, maxTime, standard_deviation_ratio, standard_deviation_time))    
        #results.append((test_name, n, k, ratio, total_time, distance))
        print(f"n={n} k={k} Completed")

    return results
