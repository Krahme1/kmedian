"""
Runs CPLEX on each k-facility instance to compute the exact optimum.
Saves results incrementally to datasets/pmed/optimums.json so a crash
doesn't lose progress (it skips instances already done).
Run from the repo root:  python run_cplex.py
"""
import json, os, time
from utils.graph import DistanceGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.CPLEXKFSolver import CPLEXKFSolver

KF_DIR = "datasets/pmed/kf_tests"
OUT_PATH = "datasets/pmed/optimums.json"

# load any optimums already computed (so we can resume)
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results = json.load(f)
else:
    results = {}

for i in range(1, 41):
    name = f"pmed{i}"
    if name in results:
        print(f"{name}: already done (optimum={results[name]['optimum']}), skipping")
        continue

    with open(os.path.join(KF_DIR, f"{name}.json")) as f:
        data = json.load(f)
    n, k = data["n"], data["k"]
    distances, costs = data["distances"], data["costs"]

    graph = DistanceGraph(distances, False)
    problem = KFProblem(name, graph, n, k, None, costs)
    solver = CPLEXKFSolver()
    solver.initialize(problem)

    t0 = time.time()
    solver.solve()
    elapsed = time.time() - t0

    optimum = solver.getSolutionValue()
    facilities = sorted(int(f) for f in solver.getSelectedFacilities())

    # only save if CPLEX actually found a real solution
    if optimum == float("inf") or len(facilities) == 0:
        print(f"{name}: SOLVE FAILED (optimum={optimum}) -> NOT saved, will retry")
        continue

    results[name] = {"n": n, "k": k, "optimum": optimum,
                     "facilities": facilities, "solve_time": elapsed}

    # save after EACH instance (checkpoint)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{name}: n={n} k={k} optimum={optimum:.1f} "
          f"({len(facilities)} facilities, {elapsed:.1f}s) -> saved")

print(f"\nDone. Optimums saved to {OUT_PATH}")
