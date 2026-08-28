"""
Hand-verification test for CPLEXKFSolver (k-facility location).

Run this from the root of your kmedian folder on Nibi:
    module load python/3.11
    source ~/kmedian-env/bin/activate
    cd ~/scratch/kmedian
    python verify_cplex_kf.py

It builds two tiny instances whose optimal answers we can work out by hand,
runs CPLEX on them, and checks the answers match.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.KFProblem import KFProblem
from solvers_alg.KF.CPLEXKFSolver import CPLEXKFSolver


class MockGraph:
    """Minimal graph: just returns the distance between two nodes."""
    def __init__(self, dist_matrix):
        self._d = dist_matrix
    def get_standard_distance(self, i, j):
        return self._d[i][j]


def run_case(name, dist, costs, k, expected_cost, expected_num_open):
    n = len(dist)
    graph = MockGraph(dist)
    problem = KFProblem(name, graph, n, k, optimal=None, costs=costs)

    solver = CPLEXKFSolver()
    solver.initialize(problem)
    solver.solve()

    value = solver.getSolutionValue()
    facilities = solver.getSelectedFacilities()

    print(f"\n=== {name} ===")
    print(f"  k (budget)          : {k}")
    print(f"  facility costs       : {costs}")
    print(f"  CPLEX total cost     : {value}")
    print(f"  CPLEX opened         : {sorted(int(f) for f in facilities)}  ({len(facilities)} facilities)")
    print(f"  EXPECTED total cost  : {expected_cost}")
    print(f"  EXPECTED # opened    : {expected_num_open}")
    cost_ok = abs(value - expected_cost) < 1e-6
    num_ok = len(facilities) == expected_num_open
    print(f"  RESULT               : {'PASS' if (cost_ok and num_ok) else 'CHECK - does not match'}")


# ---------------------------------------------------------------------------
# CASE 1: two far-apart clusters, cheap facilities.
# Nodes {0,1} are close; nodes {2,3} are close; the two pairs are far apart.
# Facility cost = 2 each, k = 2.
# By hand: open one facility per cluster (e.g. 0 and 2).
#   distances: 0->0=0, 1->0=1, 2->2=0, 3->2=1  => 2
#   opening  : 2 + 2                            => 4
#   TOTAL = 6, opening 2 facilities.
# ---------------------------------------------------------------------------
dist1 = [
    [0,  1, 10, 10],
    [1,  0, 10, 10],
    [10, 10, 0,  1],
    [10, 10, 1,  0],
]
run_case("Case 1: two clusters, cheap facilities", dist1, [2, 2, 2, 2], k=2,
         expected_cost=6, expected_num_open=2)

# ---------------------------------------------------------------------------
# CASE 2: one tight cluster, EXPENSIVE facilities  -> tests "at most k".
# 3 nodes, all pairwise distance 1. Facility cost = 5 each, k = 2.
# By hand:
#   open 1 facility (e.g. node 0): distances 0+1+1 = 2, opening 5 => TOTAL 7
#   open 2 facilities:             distances 0+0+1 = 1, opening 10 => TOTAL 11
# So the optimum opens only 1 facility (total 7). This only comes out right
# if the model allows AT MOST k (the fix); "exactly k" would force 11.
# ---------------------------------------------------------------------------
dist2 = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]
run_case("Case 2: expensive facilities (checks at-most-k)", dist2, [5, 5, 5], k=2,
         expected_cost=7, expected_num_open=1)

print("\nDone. Both cases should say PASS.")
