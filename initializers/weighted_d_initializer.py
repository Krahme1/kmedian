import random
from initializers.base_initializer import BaseInitializer


class WeightedDInitializer(BaseInitializer):
    def __init__(self, weight_type):
        super().__init__(f"weighted_D_{weight_type}")
        self.weight_type = weight_type

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

                structural_weight = self.get_weight(graph, n, u)
                weights.append(structural_weight * d_u)

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

    def get_weight(self, graph, n, u):
        if self.weight_type == "degree":
            return self.degree_weight(graph, n, u)

        if self.weight_type == "closeness":
            return self.closeness_weight(graph, n, u)

        if self.weight_type == "density":
            return self.density_weight(graph, n, u)

        return 1

    def degree_weight(self, graph, n, u):
        count = 0

        for v in range(n):
            if u != v and graph.get_standard_distance(u, v) < float("inf"):
                count += 1

        return count

    def closeness_weight(self, graph, n, u):
        total_distance = 0

        for v in range(n):
            total_distance += graph.get_standard_distance(u, v)

        if total_distance == 0:
            return 0

        return 1 / total_distance

    def density_weight(self, graph, n, u):
        nearest_distances = []

        for a in range(n):
            best = float("inf")

            for b in range(n):
                if a != b:
                    best = min(best, graph.get_standard_distance(a, b))

            nearest_distances.append(best)

        r = sum(nearest_distances) / n

        count = 0
        for v in range(n):
            if graph.get_standard_distance(u, v) <= r:
                count += 1

        return count