# =============================================================
#  Resende & Werneck's Fast Interchange for k-median (FI in the paper)
#  Menu option : 13  ("Fast Interchange")
#  Class name  : InterchangeAlgorithmSolver
#  An efficient swap-based local-search method (comparison baseline).
# =============================================================
import random
import sys
import time

import torch

from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class InterchangeAlgorithmSolver(KMPSolver):
    def __init__(self, use_gpu):
        # Initialize Variables for Solver
        self._name = "Fast Interchange"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._check_counter = 0
        self._swap_counter = 0

        self._vertices = None
        
        #fast interchange tensors
        self._gain = None
        self._loss = None
        self._d1 = None
        self._d1_indices = None
        self._d2 = None
        self._d2_indices = None
        self._extra = None
        
        self.maxTime = None

    def initialize(self, problem:KMProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        # prepare the initial vertices
        vertices = [0 for _ in range(self._n)]
        # randomly pick k vertices
        for value in random.sample([i for i in range(0, self._n)], k=self._k):
            vertices[value] = 1
        self._vertices = torch.tensor(vertices)

        #fast interchange tensors
        self._gain = torch.zeros(size=(self._n,))
        self._loss = torch.zeros(size=(self._n,))
        closest_values, closest_indices = torch.topk(self._vertices * (1 - self._graph._normalized_distances), k=2, dim=1) 
        closest_values = 1 - closest_values
        self._d1 = closest_values[:,0]
        self._d1_indices = closest_indices[:,0]
        self._d2 = closest_values[:,1]
        self._d2_indices = closest_indices[:,1]
        self._extra = torch.zeros(size=(self._n, self._n))
        
        if self._n < 1000:
            self.maxTime = 5
        elif self._n > 1000 and self._n < 1500:
            self.maxTime = 1
        elif self._n > 1000 and self._n < 3000:
            self.maxTime = 2
        elif self._n > 1000 and self._n < 5000:
            self.maxTime = 3
        elif self._n > 1000 and self._n < 6000:
            self.maxTime = 20
        elif self._n > 1000 and self._n < 15000 and self._k < 1000:
            self.maxTime = 50
        elif self._n > 1000 and self._n < 15000 and self._k == 1000:
            self.maxTime = 75
        elif self._n > 1000 and self._n < 15000 and self._k == 2000:
            self.maxTime = 100
        elif self._n > 1000 and self._n < 15000 and self._k > 2000:
            self.maxTime = 200

    def getName(self):
        return self._name
    
    def getSelectedFacilities(self):
        return self._selectedFacilities
    
    def getSolutionValue(self):
        return self._solutionValue
    
    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def solve(self, runNum=None):
        self._start_time = time.time()
        self._affected_clients = [i for i in range(self._n) if self._vertices[i] == 0]
        self.resetStructures()
        while True:
            for client in self._affected_clients: self.updateStructures(client)
            if time.time() - self._start_time >= self.maxTime:
                break
            removed_facility, inserted_facility, profit = self.findBestNeighbor()
            if time.time() - self._start_time >= self.maxTime:
                break
            if profit <= 0: break
            self._affected_clients = []
            for client in range(self._n):
                if self._vertices[client] == 0:
                    if removed_facility == self._d1_indices[client] or removed_facility == self._d2_indices[client] or self._graph._normalized_distances[client,inserted_facility] < self._graph._normalized_distances[client,self._d2_indices[client]]:
                        self._affected_clients.append(client)
            for client in self._affected_clients: self.undoUpdateStructures(client)
            if time.time() - self._start_time >= self.maxTime:
                break
            self._vertices[inserted_facility] = 1
            self._vertices[removed_facility] = 0
            self.updateClosest()

        self._selectedFacilities = [i for i in range(self._n) if self._vertices[i] == 1]
        self._solutionValue = calculate_distance(self._graph, self._selectedFacilities, self._n)
        
        
    def resetStructures(self):
        self._gain = torch.zeros(size=(self._n,))
        self._loss = torch.zeros(size=(self._n,))
        self._extra = torch.zeros(size=(self._n, self._n))
        
    def updateStructures(self, client):
        facility_to_remove = self._d1_indices[client]
        self._loss[facility_to_remove] += self._d2[client] - self._d1[client]
        for facility_to_insert in range(self._n):
            if time.time() - self._start_time >= self.maxTime:
                break
            if self._vertices[facility_to_insert] == 0:
                if self._graph._normalized_distances[client, facility_to_insert] < self._d2[client]:
                    self._gain[facility_to_insert] += max(0, self._d1[client] - self._graph._normalized_distances[client,facility_to_insert])
                    self._extra[facility_to_insert,facility_to_remove] += self._d2[client] - max(self._graph._normalized_distances[client,facility_to_insert], self._d1[client])
        
    def undoUpdateStructures(self, client):
        facility_to_remove = self._d1_indices[client]
        self._loss[facility_to_remove] -= self._d2[client] - self._d1[client]
        for facility_to_insert in range(self._n):
            if time.time() - self._start_time >= self.maxTime:
                break
            if self._vertices[facility_to_insert] == 0:
                if self._graph._normalized_distances[client, facility_to_insert] < self._d2[client]:
                    self._gain[facility_to_insert] -= max(0, self._d1[client] - self._graph._normalized_distances[client,facility_to_insert])
                    self._extra[facility_to_insert,facility_to_remove] -= self._d2[client] - max(self._graph._normalized_distances[client,facility_to_insert], self._d1[client])
        
    def findBestNeighbor(self):
        currProfit = -1
        bestProfit = -1
        bestInsert = -1
        bestRemove = -1
        for currInsert in range(self._n):
            if time.time() - self._start_time >= self.maxTime:
                break
            if self._vertices[currInsert] == 0:
                for currRemove in range(self._n):
                    if self._vertices[currRemove] == 1:
                        currProfit = self._gain[currInsert] - self._loss[currRemove] + self._extra[currInsert,currRemove]
                        if currProfit > bestProfit:
                            bestProfit = currProfit
                            bestInsert = currInsert
                            bestRemove = currRemove
        return bestRemove, bestInsert, bestProfit  
        
    def updateClosest(self):
        closest_values, closest_indices = torch.topk(self._vertices * (1 - self._graph._normalized_distances), k=2, dim=1) 
        closest_values = 1 - closest_values
        self._d1 = closest_values[:,0]
        self._d1_indices = closest_indices[:,0]
        self._d2 = closest_values[:,1]
        self._d2_indices = closest_indices[:,1]