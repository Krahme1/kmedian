import random
from initializers.base_initializer import BaseInitializer


class DQSamplingInitializer(BaseInitializer):
    def __init__(self, q):
        super().__init__(f"D^{q}_sampling")
        self.q = q

    def initialize(self, graph, n, k):
        facilities = [random.randrange(n)]

        while len(facilities) < k:
            weights = []

            for u in range(n):
                if u in facilities:
                    weights.append(0)
                    continue

                d_u = min(
                    graph.get_standard_distance(u, c)
                    for c in facilities
                )

                weights.append(d_u ** self.q)

            total = sum(weights)

            if total == 0:
                remaining = [u for u in range(n) if u not in facilities]
                facilities.append(random.choice(remaining))
            else:
                next_facility = random.choices(
                    range(n),
                    weights=weights,
                    k=1
                )[0]

                if next_facility not in facilities:
                    facilities.append(next_facility)

        return facilities