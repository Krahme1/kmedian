from solvers_alg.Solver import Solver


class FLSolver(Solver):
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

    def setGraph(self, graph):
        self._graph = graph