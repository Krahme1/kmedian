"""
Verifies every CPLEX optimum in optimums.json is feasible and correctly valued.
For each instance, checks:
  1. at most k facilities are selected (and no duplicates)
  2. the reported value == independently recomputed objective
     (sum over nodes of distance to nearest open facility + opening costs)
"""
import json

optimums = json.load(open("datasets/pmed/optimums.json"))

def recompute_objective(distances, costs, facilities):
    n = len(distances)
    # assignment cost: each node to its nearest open facility
    assign = 0.0
    for u in range(n):
        best = min(distances[u][f] for f in facilities)
        assign += best
    # opening cost of the chosen facilities
    open_cost = sum(costs[f] for f in facilities)
    return assign + open_cost

print(f"{'instance':10} {'k':>4} {'#fac':>5} {'reported':>12} {'recomputed':>12} {'diff':>10}  status")
problems = []
for i in range(1, 41):
    name = f"pmed{i}"
    rec = optimums[name]
    k = rec["k"]
    facilities = rec["facilities"]
    reported = rec["optimum"]

    d = json.load(open(f"datasets/pmed/kf_tests/{name}.json"))
    distances, costs = d["distances"], d["costs"]

    recomputed = recompute_objective(distances, costs, facilities)
    diff = abs(recomputed - reported)

    # feasibility checks
    too_many = len(facilities) > k
    dupes = len(set(facilities)) != len(facilities)
    mismatch = diff > 1e-4

    status = "OK"
    if too_many:  status = "FAIL: >k facilities"
    elif dupes:   status = "FAIL: duplicates"
    elif mismatch: status = "FAIL: value mismatch"
    if status != "OK":
        problems.append(name)

    print(f"{name:10} {k:>4} {len(facilities):>5} {reported:>12.2f} {recomputed:>12.2f} {diff:>10.4f}  {status}")

print()
if problems:
    print("PROBLEM INSTANCES:", problems)
else:
    print("All 40 CPLEX optimums are feasible and correctly valued.")
