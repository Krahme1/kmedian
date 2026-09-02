"""
Generates 10 random starting solutions for each k-facility instance and saves
them, so all three algorithms can start from the SAME 10 starts (fair comparison)
and the experiments can be reproduced.

A starting solution = a list of k distinct node indices (0 .. n-1).
Output: datasets/pmed/starts/pmed<i>.json  (a list of 10 lists)

Run from the repo root:  python make_starts.py
"""
import json
import os
import random

KF_DIR = "datasets/pmed/kf_tests"     # the k-facility instances (have n, k)
OUT_DIR = "datasets/pmed/starts"      # where the starting solutions go
NUM_STARTS = 10                        # 10 starting solutions per instance

os.makedirs(OUT_DIR, exist_ok=True)

for i in range(1, 41):
    name = f"pmed{i}"
    with open(os.path.join(KF_DIR, f"{name}.json")) as f:
        data = json.load(f)
    n = data["n"]
    k = data["k"]

    # seed per-instance so the starts are reproducible if we ever rerun
    random.seed(i)
    starts = [random.sample(range(n), k) for _ in range(NUM_STARTS)]

    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as out:
        json.dump(starts, out)

    print(f"{name}: n={n} k={k} -> saved {len(starts)} starting solutions")

print(f"\nDone. Starting solutions written to {OUT_DIR}")
