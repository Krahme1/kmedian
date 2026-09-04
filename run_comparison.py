"""
Runs the 3 KF algorithms (Samei, Zhang, KUFL) on each instance, 10 times each
(once per shared starting solution), and reports approximation ratios and
runtimes against the CPLEX optimum.

- Saves results incrementally to datasets/pmed/comparison.json (resume on restart)
- Writes the final table to results/kf_comparison.csv
- Silences the solvers' internal print output so the log stays readable

Run from the repo root:  python run_comparison.py
"""
import json, os, time, statistics, sys, io, contextlib
from utils.graph import DistanceGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.SameiSolisObaKFSolver import SameiSolisObaKFSolver
from solvers_alg.KF.ZhangKFSolver import ZhangKFSolver
from solvers_alg.KF.KUFL import HopfieldOriginalSolver as KUFLSolver

OPT_PATH  = "datasets/pmed/optimums.json"
KF_DIR    = "datasets/pmed/kf_tests"
START_DIR = "datasets/pmed/starts"
OUT_JSON  = "datasets/pmed/comparison.json"
OUT_CSV   = "results/kf_comparison.csv"
ALGOS = ["Samei", "Zhang", "KUFL"]

optimums = json.load(open(OPT_PATH))

def new_solver(name):
    if name == "Samei": return SameiSolisObaKFSolver(swap_size=1)
    if name == "Zhang": return ZhangKFSolver(swap_size=1, epsilon_prime=0.001)
    if name == "KUFL":  return KUFLSolver(use_gpu=False)

@contextlib.contextmanager
def quiet():
    """Hide the solvers' internal print() output."""
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved

def run_instance(name):
    d = json.load(open(os.path.join(KF_DIR, f"{name}.json")))
    starts = json.load(open(os.path.join(START_DIR, f"{name}.json")))
    opt = optimums[name]["optimum"]
    n, k = d["n"], d["k"]
    row = {"instance": name, "n": n, "k": k, "optimum": opt}

    for algo in ALGOS:
        ratios, times = [], []
        for S in starts:
            g = DistanceGraph(d["distances"], False)
            prob = KFProblem(name, g, n, k, None, d["costs"])
            solver = new_solver(algo)
            solver.initialize(prob)
            t0 = time.time()
            with quiet():
                solver.solve(starter_facilities=S)
            times.append(time.time() - t0)
            ratios.append(solver.getSolutionValue() / opt)
        row[algo] = {
            "min_ratio": min(ratios),
            "avg_ratio": sum(ratios)/len(ratios),
            "max_ratio": max(ratios),
            "avg_time":  sum(times)/len(times),
            "std_time":  statistics.pstdev(times),
        }
        r = row[algo]
        print(f"    {algo}: ratio {r['min_ratio']:.3f}/{r['avg_ratio']:.3f}/{r['max_ratio']:.3f}  "
              f"time {r['avg_time']:.2f}s (std {r['std_time']:.2f})", flush=True)
    return row

def write_csv(results):
    os.makedirs("results", exist_ok=True)
    header = ["Instance", "n", "k", "Optimum"]
    for a in ALGOS:
        header += [f"{a} MinR", f"{a} AvgR", f"{a} MaxR", f"{a} AvgTime", f"{a} StdTime"]
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(1, 41):
            name = f"pmed{i}"
            if name not in results: continue
            r = results[name]
            line = [name, r["n"], r["k"], f"{r['optimum']:.1f}"]
            for a in ALGOS:
                s = r[a]
                line += [f"{s['min_ratio']:.3f}", f"{s['avg_ratio']:.3f}", f"{s['max_ratio']:.3f}",
                         f"{s['avg_time']:.3f}", f"{s['std_time']:.3f}"]
            w.writerow(line)
    print(f"\nWrote table to {OUT_CSV}")

# resume from any previous progress
results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}

for i in range(1, 41):
    name = f"pmed{i}"
    if name in results:
        print(f"{name}: already done, skipping", flush=True)
        continue
    print(f"{name}: n={optimums[name]['n']} k={optimums[name]['k']}", flush=True)
    results[name] = run_instance(name)
    json.dump(results, open(OUT_JSON, "w"), indent=2)   # checkpoint after each
    write_csv(results)                                   # refresh CSV as we go

print("\nDone. All instances processed.")
