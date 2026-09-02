"""
Converts pmed k-median instances into k-facility instances by adding
random facility costs. Cost recipe: solve k-median for value V (via ARN),
then each node's facility cost is random in [0, 2V/k].
Run from the repo root:  python pmed_to_kf.py
"""
import json
import os
import random
from utils.graph import DistanceGraph
from problems.KMProblem import KMProblem
from solvers_alg.KMP.Main.ARN import HopfieldOriginalSolver   # ARN

IN_DIR = "datasets/pmed/tests"
OUT_DIR = "datasets/pmed/kf_tests"
os.makedirs(OUT_DIR, exist_ok=True)

# process pmed1 through pmed40
for i in range(1, 41):
    name = f"pmed{i}"
    in_path = os.path.join(IN_DIR, f"{name}.json")

    with open(in_path) as f:
        data = json.load(f)
    n = data["n"]
    k = data["k"]
    distances = data["distances"]

    # compute V by solving k-median with ARN
    graph = DistanceGraph(distances, False)
    problem = KMProblem(name, graph, n, k, None)
    solver = HopfieldOriginalSolver(False)
    solver.initialize(problem)
    solver.solve()
    V = solver.getSolutionValue()

    # generate random facility costs in [0, 2V/k]; seed by instance for repeatability
    random.seed(i)
    upper = 2 * V / k
    costs = [random.uniform(0, upper) for _ in range(n)]

    # save as a k-facility JSON instance
    kf_instance = {
        "format": 9,
        "name": name,
        "n": n,
        "k": k,
        "distances": distances,
        "optimal_solution": None,
        "costs": costs,
    }
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as out_file:
        json.dump(kf_instance, out_file)

    print(f"{name}: n={n} k={k} V={V} cost_range=[0,{upper:.1f}] -> saved")

print("\nDone. All k-facility instances written to", OUT_DIR)
