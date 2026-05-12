from solvers_alg.Solver import Solver


class KFSolver(Solver):
    def __init__(self):
        self.name = None
        self.solutionValue = 0
        self.selectedFacilities = []

    def getName(self):
        return self.name

    def solve(self):
        raise NotImplementedError

    def getSolutionValue(self):
        return self.solutionValue

    def getSelectedFacilities(self):
        return self.selectedFacilities
    
    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def setCosts(self, costs):
        self._costs = costs