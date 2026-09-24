import json, os, time
from utils.graph import CoordinateGraph
from problems.KFProblem import KFProblem
from solvers_alg.KF.CPLEXKFSolver import CPLEXKFSolver

KF_DIR = "datasets/random/kf_tests"
OUT_PATH = "datasets/random/optimums.json"
K_VALUES = [2, 5, 10]
N_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
TIME_LIMIT = 30.0     # seconds per instance; report best-so-far if it doesn't finish

# load any progress so we can resume
results = {}
if os.path.exists(OUT_PATH):
    results = json.load(open(OUT_PATH))

for n in N_VALUES:
    for k in K_VALUES:
        config = "n" + str(n) + "k" + str(k)
        if config in results:
            print(config + ": already done, skipping", flush=True)
            continue

        instances = json.load(open(os.path.join(KF_DIR, config + ".json")))
        config_opts = []
        t_config = time.time()
        for idx, inst in enumerate(instances):
            g = CoordinateGraph(inst["x_values"], inst["y_values"], False)
            prob = KFProblem(config, g, n, k, None, inst["costs"])
            solver = CPLEXKFSolver()
            solver.initialize(prob)
            solver.solve(max_time=TIME_LIMIT)      # time-limited; returns best found
            config_opts.append(solver.getSolutionValue())

        results[config] = config_opts
        json.dump(results, open(OUT_PATH, "w"))
        print(config + ": solved " + str(len(config_opts)) + " instances in " + str(round(time.time()-t_config,1)) + "s", flush=True)

print("Done. Optimums written to " + OUT_PATH)
