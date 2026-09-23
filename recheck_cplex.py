"""
Re-solve pmed18 and pmed19 with CPLEX and report the solve status + gap,
to see why CPLEX returned a suboptimal value.
"""
import json
from utils.graph import DistanceGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.CPLEXKFSolver import CPLEXKFSolver

for name in ["pmed18", "pmed19"]:
    d = json.load(open(f"datasets/pmed/kf_tests/{name}.json"))
    g = DistanceGraph(d["distances"], False)
    prob = KFProblem(name, g, d["n"], d["k"], None, d["costs"])
    solver = CPLEXKFSolver()
    solver.initialize(prob)
    solver.solve()   # no time limit

    facilities = sorted(int(f) for f in solver.getSelectedFacilities())
    value = solver.getSolutionValue()
    details = solver._model.get_solve_details()

    print(f"\n=== {name} (k={d['k']}) ===")
    print(f"  value:  {value:.2f}")
    print(f"  #facilities opened: {len(facilities)}")
    print(f"  solve status: {details.status}")
    print(f"  mip relative gap: {details.mip_relative_gap}")
