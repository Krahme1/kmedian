# =============================================================
#  Pan & Zhu's local search for k-median
#  Menu option : 4  ("Zhu's algorithm MRA")
#  Class name  : ZhuAlgorithmSolver
#  A competing local-search method used as a comparison baseline.
# =============================================================
import random
import sys
import time

import torch

from problems.KMProblem import KMProblem
from solvers_alg.KMP.KMPSolver import KMPSolver


class ZhuAlgorithmSolver(KMPSolver):
    def __init__(self, max_time, solution=None):
        # Initialize Variables for Solver
        self._name = "Zhu Algorithm"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = None
        self._n = None
        self._k = None
        self._max_time = max_time
        self._check_counter = 0
        self._swap_counter = 0
        self._p = 3
        self._intersection = []

        self._solution = solution
        self._vertices = None
        
        self._maxTime = 0

    def initialize(self, problem:KMProblem):
        self._graph = problem.getGraph()
        self._n = problem.getN()
        self._k = problem.getK()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")
        
        # prepare the initial vertices
        if self._solution:
            vertices = []
            for i in range(self._n):
                if i in self._solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(self._n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, self._n)], k=self._k):
                vertices[value] = 1

        self._vertices = vertices

        if self._n > 6000:
            self._maxTime = 200
        elif self._n < 1000:
            self._maxTime = 5
        elif self._n > 1000 and self._k < 1000:
            self._maxTime = 10
        elif self._n > 1000 and self._k >= 1000:
            self._maxTime = 15
        else:
            self._maxTime = 15

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
    
        # print("Vertices: ", self.n, ", Facilities: ", self.k)
        
        # Zhu's algorithm runs a simple local search algorithm p times
        # and computes the intersection of facilities across the p solutions
        for iteration in range(self._p):
            
            # The p solutions need to start from random places
            #temp_vertices = self.vertices.copy()
            
            temp_vertices = [0 for _ in range(self._n)]
            
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, self._n)], k=self._k):
                temp_vertices[value] = 1
            
            # In each iteration, the simple local search algorithm is called
            facilities = self.simple(temp_vertices)
            #print("Iteration: ", iteration, ", Facilities: ", facilities, ", distance: ", self.calculate_distance(temp_vertices))            
            
            # After the first iteration, add the facilities to the intersection
            if iteration == 0:
            
                self._intersection = facilities
                
            # Subsequent iterations pop off facilities that didn't intersect
            else:
                
                temp_intersection = self._intersection.copy()
                
                for facility in temp_intersection:
                    
                    if facility not in facilities:
                    
                        self._intersection.remove(facility)
                        
        # If all p solutions had the same facilities, the rest of the algorithm does nothing
        if len(self._intersection) == self._k:
            
            for i in range(self._n):
            
                if i in self._intersection:
                
                    self._vertices[i] = 1
                    
                else:
                
                    self._vertices[i] = 0
                    
            return
            
        # If none of the p solutions have common facilities, the rest of the algorithm does nothing
        elif len(self._intersection) == 0:
        
            self._vertices = temp_vertices
            return            
                        
        # Calculate which cities are best served by the intersection of facilities
        temp_vertices = self._vertices.copy()
        cities = self.calculate_cities(temp_vertices)
            
        # Make an exclusion list of the intersected facilities and cities they serve
        exclusion_list = []
            
        # Add the intersected facilities to the exclusion list
        for i in range(len(self._intersection)):
            
            exclusion_list.append(self._intersection[i])
                
        # Add the served cities to the exclusion list
        for i in range(len(cities)):
            
            if cities[i] not in exclusion_list:
                
                exclusion_list.append(cities[i])
                
        # If every city is best served by the facilities in the intersection list, the rest of the algorithm does nothing
        if len(exclusion_list) == self._n:
        
            for i in range(self._n):
            
                if i in self._intersection:
                
                    self._vertices[i] = 1
                    
                else:
                
                    self._vertices[i] = 0
                    
            print("Exclusion list covers everything") 
            return             
                
        available_list = [i for i in range(self._n)]     
        
        # Make a list of vertices in the reduced problem
        for i in range(self._n):
        
            if i in exclusion_list:
            
                available_list.remove(i)
                
        
        temp_vertices = [0 for _ in range(self._n)]
        temp_k = self._k - len(self._intersection)
        
        # Randomly pick a reduced set of starting facilities
        for value in random.sample(available_list, k=temp_k):
        
            temp_vertices[value] = 1       
        
        # Run the simple local search algorithm on the reduced problem
        sub_facilities = self.simple(temp_vertices, exclusion_list)
        
        #print("Intersection: ", self.intersection)
        #print("Sub_facilities: ", sub_facilities)
        
        # Take the union of the intersected facilities and the facilities from the reduced problem
        for i in range(self._n):
            
            if i in self._intersection or (sub_facilities is not None and i in sub_facilities):
                
                self._vertices[i] = 1
                    
            else:
                
                self._vertices[i] = 0
            
        # Run the simple local search algorithm one more time
        facilities = self.simple(self._vertices)
        
        #print("FINAL: ", self.calculate_distance(self.vertices))   
        #print("Facilities: ", facilities)
        self._selectedFacilities = [i for i in range(self._n) if self._vertices[i] == 1]
        self._solutionValue = self.calculate_distance(self._vertices)
                 
    # The simple local search algorithm that chooses random swaps
    def simple(self, vertices, exclusion_list=None):
    
        start_time = time.time()
        best_distance = self.calculate_distance(vertices, exclusion_list)
        
        """
        if exclusion_list is None:
            
            available_clients = [i for i in range(self.n)]
            available_facilities = [i for i in range(self.n)]
                
        else:
            
            available_clients = [i for i in range(self.n) if i not in exclusion_list]
            available_facilities = [i for i in range(self.n) if i not in exclusion_list]
        """
           
        while True:
        
            if time.time() - start_time >= self._maxTime / 5:
                    
                break
        
            # Only attempt to swap clients that are in the reduced problem
            if exclusion_list is None:
            
                available_clients = [i for i in range(self._n) if vertices[i] == 0]
                
            else:
            
                available_clients = [i for i in range(self._n) if i not in exclusion_list and vertices[i] == 0]
                
            random.shuffle(available_clients)
            has_swapped = False
            
            #for client in (c for c in available_clients if vertices[c] == 0):
            for client in available_clients:
            
                # Only attempt to swap facilities that are in the reduced problem
                if exclusion_list is None:
                
                    available_facilities = [i for i in range(self._n) if vertices[i] == 1]
                    
                else:
                
                    available_facilities = [i for i in range(self._n) if i not in exclusion_list and vertices[i] == 1]
                    
                if len(available_facilities) == 0:
                
                    print("Somehow the local search has 0 available facilities")
                    return
                    
                random.shuffle(available_facilities)
                self._check_counter += 1
                facility = None
                            
                #for i in (f for f in available_facilities if vertices[f] == 1):
                for i in available_facilities:
                
                    if time.time() - start_time >= self._maxTime / 5:
                    
                        break
                    
                    temp_vertices = vertices.copy()
                    temp_vertices[i] = 0
                    temp_vertices[client] = 1
                    
                    # Calculate the distance using the exclusion list
                    new_distance = self.calculate_distance(temp_vertices, exclusion_list)
                        
                    if new_distance < (1 - (1 / self._n)) * best_distance:
                        
                        best_distance = new_distance
                        facility = i
                        break
                        
                if time.time() - start_time >= self._maxTime / 5:
                    
                    break
                            
                if facility is not None:
                    
                    vertices[client] = 1
                    vertices[facility] = 0
                    has_swapped = True
                    self._swap_counter += 1
                    break
                        
            if not has_swapped:
                
                break
                
        if exclusion_list is None:
        
            return [i for i in range(self._n) if vertices[i] == 1]
            
        else:
        
            return [i for i in range(self._n) if i not in exclusion_list and vertices[i] == 1]

    
    # Calculate the total distance using an exclusion list
    def calculate_distance(self, vertices, exclusion_list=None):
     
        # If there is an exclusion list
        if exclusion_list is not None:
            
            # All of the facilities in the exclusion list are turned off
            for i in range(len(exclusion_list)):
            
                vertices[exclusion_list[i]] = 0
     
        facility_tensor = torch.tensor(vertices)
        max_values, _ = torch.max(facility_tensor * (1 - self._graph._normalized_distances), dim=1)
        return torch.sum(1 - max_values)
        
        
    # Calculate the cities that are served by the intersected facilities
    def calculate_cities(self, vertices):
        '''
        Need to make sure I only exclude cities best served by the intersected facilities
        Do I simply set all vertices to 1? and see which have the best scores using the intersection list?
        '''
        # Make a list of vertices with only the intersected facilities
        for i in range(len(vertices)):
        
            #if i in self.intersection:
            
            vertices[i] = 1
                
            #else:
            
            #   vertices[i] = 0
    
        #facility_tensor = torch.tensor(vertices)
        #max_values, max_indices = torch.max(facility_tensor * (1 - self.graph._normalized_distances), dim=1)
        #print(facility_tensor)
        #print(max_values)
        #print(max_indices)
        #print(1 - self.graph._normalized_distances)
        
        cities = []
        
        for i in range(self._n):
        
            val = 0
            index = 0
        
            for j in range (self._n):
            
                if i != j:
                
                    if 1 - self._graph._normalized_distances[i][j] > val:
                    
                        val = 1 - self._graph._normalized_distances[i][j]
                        index = j
                        
            if index in self._intersection:
            
                cities.append(i)
                #print("Added", i)
                #self.intersection.add(i)
                
        return cities
        
        # Make a list of vertices served by the intersected facilities
        #for i in range(self.n):
        
        #    if max_indices[i] in self.intersection:
            
        #        cities.append(i)
        
        #return cities
        
    def remove_cities(self):
    
        #print out this tensor thing, if its a n x n matrix that is useful to know
        #identify the smallest distance (best city?) for each removed facility, these should be removed too
    
        # Loop through the table. Considering only the values in spots where a facility is located
        # If the biggest value is for a facility that has been removed, the city should be removed too
        
        cities = []
        
        for i in range (self._n):
        
            val = 0
            index = 0
        
            for j in range (self._n):
            
                if self._vertices[j] == 1:
                
                    if 1 - self._graph._normalized_distances[i][j] > val:
                    
                        val = 1 - self._graph._normalized_distances[i][j]
                        index = j
                        
            if index in self._intersection:
            
                cities.append(i)
                #self.intersection.add(i)
                
        return cities
                