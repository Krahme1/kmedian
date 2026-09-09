"""
Diagnose KUFL's objective on pmed18 and pmed19 (the below-1.0 ratio cases).
Run KUFL, capture its selected facilities and reported value, then
independently recompute the true objective and compare.
"""
import json
from utils.graph import DistanceGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.KUFL import HopfieldOriginalSolver as KUFLSolver

def true_objective(distances, costs, facilities):
    n = len(distances)
    assign = sum(min(distances[u][f] for f in facilities) for u in range(n))
    opening = sum(costs[f] for f in facilities)
    return assign, opening, assign + opening

for name in ["pmed18", "pmed19"]:
    d = json.load(open(f"datasets/pmed/kf_tests/{name}.json"))
    starts = json.load(open(f"datasets/pmed/starts/{name}.json"))
    opt = json.load(open("datasets/pmed/optimums.json"))[name]["optimum"]
    S = starts[0]

    g = DistanceGraph(d["distances"], False)
    prob = KFProblem(name, g, d["n"], d["k"], None, d["costs"])
    solver = KUFLSolver(use_gpu=False)
    solver.initialize(prob)
    solver.solve(starter_facilities=S)

    facilities = [int(f) for f in solver.getSelectedFacilities()]
    reported = solver.getSolutionValue()
    assign, opening, recomputed = true_objective(d["distances"], d["costs"], facilities)

    print(f"\n=== {name} (k={d['k']}) ===")
    print(f"  CPLEX optimum:        {opt:.2f}")
    print(f"  KUFL reported value:  {reported:.2f}   (ratio {reported/opt:.3f})")
    print(f"  KUFL opened {len(facilities)} facilities")
    print(f"  Recomputed from those facilities:")
    print(f"    assignment (distances): {assign:.2f}")
    print(f"    opening costs:          {opening:.2f}")
    print(f"    TRUE total:             {recomputed:.2f}   (ratio {recomputed/opt:.3f})")
    print(f"  Difference (reported vs true): {abs(reported - recomputed):.2f}")
