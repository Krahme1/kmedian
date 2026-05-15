import random
from initializers.base_initializer import BaseInitializer


class UniformInitializer(BaseInitializer):
    def __init__(self, seed=None):
        super().__init__("uniform")
        self._seed = seed

    def initialize(self, graph, n, k):
        rng = random.Random(self._seed) if self._seed is not None else random
        return rng.sample(range(n), k)