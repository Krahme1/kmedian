import random
import time

import torch

from problems.KMProblem import KMProblem
from solvers_alg.KMP.KMPSolver import KMPSolver


class DominguezAlgorithmSolver(KMPSolver):
    def __init__(self, use_gpu=False):
        # Initialize Variables for Solver
        self._name = "Dominguez"
        self._solutionValue = 0
        self._selectedFacilities = []

        self.verbose = False
        self._n = None
        self._k = None
        self._graph = None

        self._vertices = None
        
    def initialize(self, problem:KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()

        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        self._size = (self._n, self._k)
        
        self._distanceValues = self._graph._normalized_distances.detach().clone()
        self._facilityAssignments = torch.zeros(self._size)
        self._clientAssignments = torch.zeros(self._size)
        
        self._mathRowIndices = torch.arange(self._n)
        self._mathColIndices = torch.arange(self._k)
        
        self._vertices = [0] * self._n
        
        """
        if self.verbose is True:
            print("These are the initial (normalized) distance values")
            print(self.distanceValues)
            print()
        """
        for i in range(self._n):
            self._vertices[i] = 0

    def getName(self):
        return self._name
    
    def getSelectedFacilities(self):
        return self._selectedFacilities
    
    def getSolutionValue(self):
        return self._solutionValue
    
    def setN (self, n):
        self._n = n

    def setK (self, k):
        self._k = k

    def setGraph (self, graph):
        self._graph = graph

    def setMaxTime (self, max_time):
        self._maxTime = max_time
    
    def solve(self, runNum=None):
        best_facilities = None
        best_distance = None

        self.initializePerRunArrays()
        updateClients = True
        stabilized = False
        numIterations = 0
        while not stabilized:
            numIterations += 1
            energyBefore = self.calculateDistance()[0]
            #print("Start of the loop, EnergyBefore: ",energyBefore)
            if updateClients:
                self.updateClients()
            else:
                self.updateFacilities()
            updateClients = not updateClients
            energyAfter = self.calculateDistance()[0]
            #print("EnergyAfter: ",energyAfter)
            if updateClients and energyAfter >= energyBefore:
                stabilized = True
                #print("Stabilized:",stabilized)
                #print("Finished in ",numIterations," iterations")
        distance, selectedFacilities = self.calculateDistance()
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_facilities = selectedFacilities
        """
        if self.verbose is True:
            print("Returning values:",self.vertices)
        """

        self._selectedFacilities = best_facilities
        self._solutionValue = best_distance
            
    def initializePerRunArrays(self):
        self._facilityAssignments = torch.zeros(size=self._size)
        self._clientAssignments = torch.zeros(size=self._size)
        self._activeFacilityList = []
        
        # Randomly assign a facility to each cluster
        selectedFacilities = random.sample(range(self._n), self._k)
        for index, val in enumerate(selectedFacilities):
            self._facilityAssignments[val,index] = 1
            self._activeFacilityList.append(val)

        # Randomly assign clients to clusters
        for i in range(self._n):
            index = random.randint(0, self._k - 1)
            self._clientAssignments[i,index] = 1
            
        """
        if self.verbose is True:
            print("These are the initial facility assignments")
            print(self.facilityAssignments)
            print()
        """
        
        """
        if self.verbose is True:
            print("These are the initial client assignments")
            print(self.clientAssignments)
            print()
        """
            
    def updateClients(self):
        self._clientAssignments = -self._distanceValues @ self._facilityAssignments
        """
        if self.verbose is True:
            print("Update Clients. Client values were:")
            print(self.clientAssignments)
            print()
        """
        maxIndices = torch.argmax(self._clientAssignments, dim=1)
        """
        if self.verbose is True:
            print("Max Indices are: ",maxIndices)
            print()
        #"""
        self._clientAssignments = torch.zeros(size=self._size)
        self._clientAssignments[self._mathRowIndices,maxIndices] = 1
        """
        if self.verbose is True:
            print("Client values are now:")
            print(self.clientAssignments)
            print()
        #"""
    
    def updateFacilities(self):      
        self._facilityAssignments = -self._distanceValues.T @ self._clientAssignments
        """
        if self.verbose is True:
            print("Update Facilities. Facility values were:")
            print(self.facilityAssignments)
            print()
        #"""
        self._activeFacilityList = torch.argmax(self._facilityAssignments, dim=0).tolist()
        """
        if self.verbose is True:
            print("Active facilities are:",self.activeFacilityList)
            print()
        #"""
        self._facilityAssignments = torch.zeros(size=self._size)
        self._facilityAssignments[self._activeFacilityList,self._mathColIndices] = 1     
        """
        if self.verbose is True:
            print("Facility values are now:")
            print(self.facilityAssignments)
            print()
        """        
        
    def calculateDistance(self):
        client_clusters = torch.argmax(self._clientAssignments, dim=1)
        facility_indices = torch.argmax(self._facilityAssignments, dim=0)
        total_cost = 0.0
        for i in range(self._n):
            q = client_clusters[i].item()
            j = facility_indices[q].item()
            total_cost += self._distanceValues[i,j].item()
        return total_cost, facility_indices