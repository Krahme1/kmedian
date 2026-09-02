"""
Farthest-first initialization.
Roberto's description: choose the first facility at random, then repeatedly
add the facility that is farthest from the currently selected facilities.
(A facility's distance to the selected set = distance to its NEAREST selected
facility; we pick the one whose nearest-selected distance is largest.)
"""
import random
import torch
from initializers.base_initializer import BaseInitializer


class FarthestFirstInitializer(BaseInitializer):

    def __init__(self, seed=None):
        super().__init__("farthest_first")
        self._seed = seed

    def initialize(self, graph, n, k):
        D = graph._distances.detach().to("cpu").float()

        rng = random.Random(self._seed)

        # first facility: random
        first_facility = rng.randrange(n)
        selected = [first_facility]

        # dist_to_selected[i] = distance from node i to its nearest selected facility
        dist_to_selected = D[:, first_facility].clone()

        for _ in range(1, k):
            # the farthest node from the selected set = largest nearest-distance
            next_facility = torch.argmax(dist_to_selected).item()
            selected.append(next_facility)

            # update each node's distance to the (now larger) selected set
            dist_to_selected = torch.minimum(
                dist_to_selected,
                D[:, next_facility]
            )

        return selected
