from abc import ABC, abstractmethod


class BaseInitializer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def initialize(self, graph, n, k):
        pass