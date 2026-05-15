import torch
from initializers.base_initializer import BaseInitializer

class PAMBuildInitializer(BaseInitializer):

    def __init__(self):
        super().__init__("pam_build")

    def initialize(self, graph, n, k):
        D = graph._normalized_distances.detach().to("cpu")
        num_candidates = D.shape[1]

        col_sums = D.sum(dim=0)
        first_facility = torch.argmin(col_sums).item()

        selected = [first_facility]
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool)
        selected_mask[first_facility] = True

        current_min_dist = D[:, first_facility].clone()

        for _ in range(1, k):
            candidate_scores = torch.minimum(
                current_min_dist.unsqueeze(1),
                D
            ).sum(dim=0)

            candidate_scores[selected_mask] = float("inf")

            next_facility = torch.argmin(candidate_scores).item()
            selected.append(next_facility)
            selected_mask[next_facility] = True
            
            current_min_dist = torch.minimum(
                current_min_dist,
                D[:, next_facility]
            )

        return selected