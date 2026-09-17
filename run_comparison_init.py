import json, os, time, statistics, sys, io, contextlib, csv
from utils.graph import DistanceGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.SameiSolisObaKFSolver import SameiSolisObaKFSolver
from solvers_alg.KF.ZhangKFSolver import ZhangKFSolver
from solvers_alg.KF.KUFL import HopfieldOriginalSolver as KUFLSolver
from initializers.greedy_facility_cost_initializer import GreedyFacilityCostInitializer
from initializers.farthest_first_initializer import FarthestFirstInitializer

if len(sys.argv) < 2 or sys.argv[1] not in ("greedy", "farthest"):
    print("Usage: python run_comparison_init.py [greedy|farthest]")
    sys.exit(1)
INIT = sys.argv[1]

KF_DIR = "datasets/pmed/kf_tests"
OUT_JSON = "datasets/pmed/comparison_" + INIT + ".json"
OUT_CSV = "results/kf_comparison_" + INIT + ".csv"
NUM_RUNS = 10
ALGOS = ["Samei", "Zhang", "KUFL"]
optimums = json.load(open("datasets/pmed/optimums.json"))

def new_solver(name):
    if name == "Samei":
        return SameiSolisObaKFSolver(swap_size=1)
    if name == "Zhang":
        return ZhangKFSolver(swap_size=1, epsilon_prime=0.001)
    if name == "KUFL":
        return KUFLSolver(use_gpu=False)

@contextlib.contextmanager
def quiet():
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved

def make_start(d, g):
    if INIT == "greedy":
        init = GreedyFacilityCostInitializer(d["costs"])
    else:
        init = FarthestFirstInitializer(seed=0)
    return init.initialize(g, d["n"], d["k"])

def run_instance(name):
    d = json.load(open(os.path.join(KF_DIR, name + ".json")))
    opt = optimums[name]["optimum"]
    n = d["n"]
    k = d["k"]
    row = {"instance": name, "n": n, "k": k, "optimum": opt}
    g0 = DistanceGraph(d["distances"], False)
    S = make_start(d, g0)
    for algo in ALGOS:
        ratios = []
        times = []
        for _ in range(NUM_RUNS):
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
            "avg_ratio": sum(ratios) / len(ratios),
            "max_ratio": max(ratios),
            "avg_time": sum(times) / len(times),
            "std_time": statistics.pstdev(times),
        }
        r = row[algo]
        print("    " + algo + ": avg ratio " + str(round(r["avg_ratio"], 3)) + ", avg time " + str(round(r["avg_time"], 2)) + "s", flush=True)
    return row

def write_csv(results):
    os.makedirs("results", exist_ok=True)
    header = ["Instance", "n", "k", "Optimum"]
    for a in ALGOS:
        header.append(a + " MinR")
        header.append(a + " AvgR")
        header.append(a + " MaxR")
        header.append(a + " AvgTime")
        header.append(a + " StdTime")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(1, 41):
            name = "pmed" + str(i)
            if name not in results:
                continue
            r = results[name]
            line = [name, r["n"], r["k"], round(r["optimum"], 1)]
            for a in ALGOS:
                s = r[a]
                line.append(round(s["min_ratio"], 3))
                line.append(round(s["avg_ratio"], 3))
                line.append(round(s["max_ratio"], 3))
                line.append(round(s["avg_time"], 3))
                line.append(round(s["std_time"], 3))
            w.writerow(line)

results = {}
if os.path.exists(OUT_JSON):
    results = json.load(open(OUT_JSON))

print("Running comparison with the " + INIT + " initializer.")
for i in range(1, 41):
    name = "pmed" + str(i)
    if name in results:
        print(name + ": already done, skipping", flush=True)
        continue
    print(name + ": k=" + str(optimums[name]["k"]), flush=True)
    results[name] = run_instance(name)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    write_csv(results)
print("Done. Table written to " + OUT_CSV)
