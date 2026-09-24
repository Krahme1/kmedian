import json, os, random
from utils.graph import CoordinateGraph
from problems.KMProblem import KMProblem
from solvers_alg.KMP.Main.ARN import HopfieldOriginalSolver

IN_DIR = "datasets/random"
OUT_DIR = "datasets/random/kf_tests"
K_VALUES = [2, 5, 10]          # Roberto's list, limited to what the dataset has
N_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
NUM_GRAPHS = 50                # 50 graphs per (n,k)

os.makedirs(OUT_DIR, exist_ok=True)

for n in N_VALUES:
    for k in K_VALUES:
        in_path = os.path.join(IN_DIR, "n" + str(n) + "k" + str(k) + ".json")
        if not os.path.exists(in_path):
            print("skip (missing): n" + str(n) + "k" + str(k))
            continue
        instances = json.load(open(in_path))        # a list of 100 instances
        instances = instances[:NUM_GRAPHS]           # take the first 50

        kf_instances = []
        for idx, inst in enumerate(instances):
            x = inst["x_values"]
            y = inst["y_values"]
            g = CoordinateGraph(x, y, False)

            # solve k-median with ARN to get V
            problem = KMProblem("rand", g, n, k, None)
            solver = HopfieldOriginalSolver(False)
            solver.initialize(problem)
            solver.solve()
            V = solver.getSolutionValue()

            # generate costs in [0, 2V/k]; seed by (n,k,idx) for repeatability
            random.seed(n * 100000 + k * 1000 + idx)
            upper = 2 * V / k
            costs = [random.uniform(0, upper) for _ in range(n)]

            kf_instances.append({
                "x_values": x,
                "y_values": y,
                "n": n,
                "k": k,
                "costs": costs,
            })

        out_path = os.path.join(OUT_DIR, "n" + str(n) + "k" + str(k) + ".json")
        json.dump(kf_instances, open(out_path, "w"))
        print("n" + str(n) + " k" + str(k) + ": saved " + str(len(kf_instances)) + " KF instances")

print("Done. KF instances written to " + OUT_DIR)
