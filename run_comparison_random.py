import json, os, time, statistics, sys, io, contextlib, csv
from utils.graph import CoordinateGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.SameiSolisObaKFSolver import SameiSolisObaKFSolver
from solvers_alg.KF.ZhangKFSolver import ZhangKFSolver
from solvers_alg.KF.KUFL import HopfieldOriginalSolver as KUFLSolver

KF_DIR = "datasets/random/kf_tests"
START_DIR = "datasets/random/starts"
OPT_PATH = "datasets/random/optimums.json"
OUT_JSON = "datasets/random/comparison_random.json"
OUT_CSV = "results/kf_comparison_random.csv"
K_VALUES = [2, 5, 10]
N_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
ALGOS = ["Samei", "Zhang", "KUFL"]

optimums = json.load(open(OPT_PATH))

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

def run_config(n, k):
    config = "n" + str(n) + "k" + str(k)
    instances = json.load(open(os.path.join(KF_DIR, config + ".json")))
    starts_all = json.load(open(os.path.join(START_DIR, config + ".json")))
    opts = optimums[config]
    row = {"n": n, "k": k}
    for algo in ALGOS:
        ratios = []
        times = []
        for idx, inst in enumerate(instances):
            opt = opts[idx]
            for S in starts_all[idx]:
                g = CoordinateGraph(inst["x_values"], inst["y_values"], False)
                prob = KFProblem(config, g, n, k, None, inst["costs"])
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
        print("    " + algo + ": avg ratio " + str(round(r["avg_ratio"], 3)) + ", avg time " + str(round(r["avg_time"], 3)) + "s", flush=True)
    return row

def write_csv(results):
    os.makedirs("results", exist_ok=True)
    header = ["n", "k"]
    for a in ALGOS:
        header.append(a + " MinR")
        header.append(a + " AvgR")
        header.append(a + " MaxR")
        header.append(a + " AvgTime")
        header.append(a + " StdTime")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for n in N_VALUES:
            for k in K_VALUES:
                key = "n" + str(n) + "k" + str(k)
                if key not in results:
                    continue
                r = results[key]
                line = [n, k]
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

for n in N_VALUES:
    for k in K_VALUES:
        config = "n" + str(n) + "k" + str(k)
        if config in results:
            print(config + ": already done, skipping", flush=True)
            continue
        print(config + ":", flush=True)
        results[config] = run_config(n, k)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        write_csv(results)

print("Done. Table written to " + OUT_CSV)
