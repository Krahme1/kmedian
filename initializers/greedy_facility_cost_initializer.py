"""
Greedy Facility Selection Start, modified to account for facility opening costs.
Based on Algorithm 1 in Nour's report, with Roberto's two changes:
  - first facility chosen by (total distance to all clients) + (its opening cost)
  - each added facility chosen by (assignment cost) + (its opening cost)
Uses RAW distances so distances and costs are on the same scale.
"""
import torch
from initializers.base_initializer import BaseInitializer


class GreedyFacilityCostInitializer(BaseInitializer):

    def __init__(self, costs):
        super().__init__("greedy_facility_cost")
        # costs[f] = opening cost of facility f
        self._costs = torch.tensor(costs, dtype=torch.float)

    def initialize(self, graph, n, k):
        # use RAW distances (same scale as the costs)
        D = graph._distances.detach().to("cpu").float()
        costs = self._costs

        # --- step 5: first facility = argmin( totalDistance[f] + cost[f] ) ---
        col_sums = D.sum(dim=0)                 # totalDistance[f] for every f
        first_scores = col_sums + costs         # add its opening cost
        first_facility = torch.argmin(first_scores).item()

        selected = [first_facility]
        selected_mask = torch.zeros(n, dtype=torch.bool)
        selected_mask[first_facility] = True

        # bestDistance[i]: each client's distance to nearest chosen facility so far
        current_min_dist = D[:, first_facility].clone()

        # --- steps 10-18: repeatedly add the best facility ---
        for _ in range(1, k):
            # cost[f] = sum over clients of min(bestDistance, D(i,f))
            candidate_scores = torch.minimum(
                current_min_dist.unsqueeze(1),
                D
            ).sum(dim=0)

            # step 14 change: add the opening cost of each candidate
            candidate_scores = candidate_scores + costs

            # don't reselect an already-chosen facility
            candidate_scores[selected_mask] = float("inf")

            next_facility = torch.argmin(candidate_scores).item()
            selected.append(next_facility)
            selected_mask[next_facility] = True

            current_min_dist = torch.minimum(
                current_min_dist,
                D[:, next_facility]
            )

        return selected
