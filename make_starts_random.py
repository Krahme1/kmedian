import json, os, random

KF_DIR = "datasets/random/kf_tests"
OUT_DIR = "datasets/random/starts"
K_VALUES = [2, 5, 10]
N_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
NUM_STARTS = 5      # Roberto: 5 random starts, not 10

os.makedirs(OUT_DIR, exist_ok=True)

for n in N_VALUES:
    for k in K_VALUES:
        config = "n" + str(n) + "k" + str(k)
        instances = json.load(open(os.path.join(KF_DIR, config + ".json")))
        all_starts = []
        for idx in range(len(instances)):
            random.seed(n * 100000 + k * 1000 + idx)
            starts = [random.sample(range(n), k) for _ in range(NUM_STARTS)]
            all_starts.append(starts)
        json.dump(all_starts, open(os.path.join(OUT_DIR, config + ".json"), "w"))
        print(config + ": saved 5 starts for " + str(len(all_starts)) + " instances", flush=True)

print("Done. Starts written to " + OUT_DIR)
