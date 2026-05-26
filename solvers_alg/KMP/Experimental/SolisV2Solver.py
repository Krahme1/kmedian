import random

import torch

from problems.KMProblem import KMProblem
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.KMPSolver import KMPSolver


class HopfieldSecondParallelSolver(KMPSolver):
    def __init__(self, use_gpu, seed=None):
        self._name = "Hopfield Second Parallel Solver"
        self._solutionValue = None
        self._selectedFacilities = []

        self._n = None
        self._matrix_n = None
        self._k = None
        self._graph = None

        self._rng = random.Random(seed)

        self._num_rows = None
        self._num_cols = None
        self._size = None
        self._facility_update_value = 1.0
        self._client_update_value = 1.0

        self._client_inner_values = None       # PC (n,k)
        self._facility_inner_values = None     # PF (n,k)
        self._client_activation_values = None  # C (n,k) int
        self._facility_activation_values = None  # F (n,k) int

        self._distance_values = None  # D (n,n)

        self._active_facility_list = []  # length k: center index per cluster

        # CPU/GPU toggle
        self._use_gpu = use_gpu

        # If we select the GPU and cuda is not available, fail loudly.
        if use_gpu:
            self._device = 'cuda' if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = 'cpu'

        self._math_row_indices = None
        self._k_indices = None

        self._alpha = 0.5
        self._beta = 1.0
        self._gamma = 0.1
        # =================================================
        self._eta = 0.1
        self._epsilon = 1e-9
        # =================================================

    def set_seed(self, seed: int):
        self._rng = random.Random(seed)

    def initialize(self, problem: KMProblem):
        self._n = problem.getN()
        self._k = problem.getK()
        self._graph = problem.getGraph()
        if self._graph is None or self._n is None or self._k is None:
            raise ValueError("Graph, n, and k must be set before calling initialize().")

        # Initialize distance values
        if self._use_gpu:
            self._distance_values = self._graph._gpu_normalized_distances
        else:
            self._distance_values = self._graph._normalized_distances.clone().detach()

        if self._distance_values.ndim != 2:
            raise ValueError(
                f"Expected a 2D distance matrix, got shape {tuple(self._distance_values.shape)}."
            )

        num_clients, num_facility_candidates = self._distance_values.shape
        if num_clients != num_facility_candidates:
            raise ValueError(
                "HopfieldParallelSolver expects a square distance matrix for k-median. "
                f"Got shape {tuple(self._distance_values.shape)}."
            )
        if self._k > num_facility_candidates:
            raise ValueError(
                f"k={self._k} cannot exceed number of facility candidates ({num_facility_candidates})."
            )

        if self._n != num_clients:
            print(
                "Warning: problem n does not match distance matrix size. "
                f"Using matrix size n={num_clients} for tensor ops "
                f"(problem reported n={self._n})."
            )
        self._matrix_n = num_clients

        self._num_rows = self._matrix_n
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)

        self._math_row_indices = torch.arange(0, self._matrix_n, device=self._device)
        self._k_indices = torch.arange(0, self._k, device=self._device)

        self._facility_inner_values = torch.zeros(size=self._size, device=self._device)
        self._client_inner_values = torch.zeros(size=self._size, device=self._device)
        self._facility_activation_values = torch.zeros(
            size=self._size, dtype=torch.int, device=self._device
        )
        self._client_activation_values = torch.zeros(
            size=self._size, dtype=torch.int, device=self._device
        )

    def getName(self):
        return self._name

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def _compute_rho_values(self):
        """
        Compute rho_q for each cluster q.

        rho_q = beta * (1 / |C_q|) * sum_{i in C_q} d_{i, F_q}
        where C_q is the set of clients currently assigned to cluster q,
        and F_q is the currently active facility for cluster q.
        """
        assignments = torch.argmax(self._client_activation_values, dim=1)

        facilities = torch.tensor(
            self._active_facility_list,
            dtype=torch.long,
            device=self._device
        )

        assigned_facilities = facilities[assignments]
        assigned_distances = self._distance_values[
            self._math_row_indices,
            assigned_facilities
        ]

        sums = torch.zeros(
            self._k,
            dtype=self._distance_values.dtype,
            device=self._device
        )
        counts = torch.zeros(
            self._k,
            dtype=self._distance_values.dtype,
            device=self._device
        )

        sums.scatter_add_(0, assignments, assigned_distances)
        counts.scatter_add_(0, assignments, torch.ones_like(assigned_distances))

        rho = self._beta * sums / torch.clamp(counts, min=1)
        rho[counts == 0] = 0.0

        return rho

    def solve(self, runNum=None, starter_facilities=None):
        best_facilities = starter_facilities
        best_distance = (
            calculate_distance(self._graph, best_facilities, self._n)
            if starter_facilities else None
        )

        self._initialize_per_run_arrays(starter_facilities)

        iterations = 0
        stabilized = False

        # =================================================
        seen_states = set()
        # =================================================

        while not stabilized:
            prev_C = self._client_activation_values.clone()
            prev_F = self._facility_activation_values.clone()

            d_ci_q, D_minus_q = self._compute_dciq_and_Dminusq()

            # Client updates
            self._calculate_client_values(d_ci_q, D_minus_q)
            self._update_client()

            d_ci_q, D_minus_q = self._compute_dciq_and_Dminusq()

            # Facility updates
            self._calculate_facility_values(D_minus_q)
            self._update_facility()

            # Fixed-point condition
            stabilized = (
                torch.equal(prev_C, self._client_activation_values) and
                torch.equal(prev_F, self._facility_activation_values)
            )

            # =================================================
            state_key = (
                tuple(torch.argmax(self._client_activation_values, dim=1).tolist()),
                tuple(self._active_facility_list),
            )
            if state_key in seen_states:
                print("Repeated state detected; stopping iterative phase and starting cleanup.")
                stabilized = True
            else:
                seen_states.add(state_key)
            # =================================================

            iterations += 1

            tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()

            if best_distance is None or tmp_solution_value < best_distance:
                best_distance = tmp_solution_value
                best_facilities = list(tmp_selected_facilities)

        print(f"Converged in {iterations} iterations.")

        # =================================================
        self._final_cleanup_phase()
        # =================================================

        self._selectedFacilities, self._solutionValue = self._calculate_facilities_and_distance()

        if best_distance is not None and best_distance < self._solutionValue:
            self._selectedFacilities = list(best_facilities)
            self._solutionValue = best_distance

        print(f"Distance: {self._solutionValue}")

    def _initialize_per_run_arrays(self, starter_facilities):
        self._client_activation_values = torch.zeros(
            self._size, dtype=torch.int, device=self._device
        )
        self._facility_activation_values = torch.zeros(
            self._size, dtype=torch.int, device=self._device
        )

        self._active_facility_list = []

        # Use explicit warm start if provided
        if starter_facilities is not None and len(starter_facilities) > 0:
            if len(starter_facilities) != self._k:
                raise ValueError("starter_facilities must have length k.")
            if any(f < 0 or f >= self._matrix_n for f in starter_facilities):
                raise ValueError(
                    f"starter_facilities must be within [0, {self._matrix_n - 1}]."
                )
            if len(set(starter_facilities)) != len(starter_facilities):
                raise ValueError("starter_facilities must be distinct.")
            initial_facilities = list(starter_facilities)
        else:
            initial_facilities = self._rng.sample(
                [i for i in range(0, self._matrix_n)], k=self._k
            )

        index = 0
        for value in initial_facilities:
            self._facility_activation_values[value, index] = 1
            self._active_facility_list.append(value)
            index = index + 1

        d_ci_q, D_minus_q = self._compute_dciq_and_Dminusq()

        self._calculate_client_values(d_ci_q, D_minus_q)
        self._update_client()
        self._calculate_facility_values(D_minus_q)

        tmp_selected_facilities, tmp_solution_value = self._calculate_facilities_and_distance()
        print("Initial distance:", tmp_solution_value)

    def _compute_dciq_and_Dminusq(self):
        # d_ci_q: distance from each client i to the current facility
        # assigned to cluster q
        # Shape: (n, k)
        d_ci_q = self._distance_values[:, self._active_facility_list]

        # D_minus_q: for each client i and cluster q,
        # distance to the closest facility NOT in q.
        D_minus_q = torch.empty(
            (self._matrix_n, self._k),
            dtype=self._distance_values.dtype,
            device=self._device
        )

        for q in range(self._k):
            other_centers = [
                self._active_facility_list[r]
                for r in range(self._k)
                if r != q
            ]

            if len(other_centers) == 0:
                D_minus_q[:, q] = float("inf")
            else:
                D_minus_q[:, q] = torch.min(
                    self._distance_values[:, other_centers],
                    dim=1
                ).values

        return d_ci_q, D_minus_q

    def _calculate_client_values(self, d_ci_q, D_minus_q):
        # =================================================
        # Section 1.2 client potential:
        #
        # pciq = (1 - eta) * dciq + eta * D_minus_q
        #          if D_minus_q < (1 + alpha) * dciq
        #        dciq
        #          otherwise
        # =================================================
        threshold = (1.0 + self._alpha) * d_ci_q
        blended = (1.0 - self._eta) * d_ci_q + self._eta * D_minus_q

        self._client_inner_values = torch.where(
            D_minus_q < threshold,
            blended,
            d_ci_q
        )
        # =================================================

    def _update_client(self):
        # =================================================
        # Tie-aware client reassignment:
        # keep the current assignment if its potential is within epsilon
        # of the best potential; otherwise switch to the best cluster.
        # This guarantees exactly one active cluster per client.
        # =================================================
        new_client_activation_values = torch.zeros(
            size=self._size, dtype=torch.int, device=self._device
        )

        current_assignments = torch.argmax(self._client_activation_values, dim=1)
        best_assignments = torch.argmin(self._client_inner_values, dim=1)

        current_vals = self._client_inner_values[self._math_row_indices, current_assignments]
        best_vals = self._client_inner_values[self._math_row_indices, best_assignments]

        keep_current = current_vals <= best_vals + self._epsilon
        final_assignments = torch.where(keep_current, current_assignments, best_assignments)

        new_client_activation_values[self._math_row_indices, final_assignments] = 1
        self._client_activation_values = new_client_activation_values
        # =================================================

    def _calculate_facility_values(self, D_minus_q, chunk_size=1024):
        # =================================================
        # Section 1.2 facility potential:
        #
        # pfjq(t) = sum_{i : d_ij <= D^-_{qi}(t)} d_ij + delta_jq(t)
        #
        # delta_jq(t) =
        #   gamma * (rho_q(t) - D^-_{qj}(t))   if D^-_{qj}(t) < rho_q(t)
        #   0                                  otherwise
        #
        # where rho_q(t) is computed from only the clients currently assigned
        # to cluster q:
        #
        # rho_q(t) = beta * (1 / |C_q|) * sum_{i in C_q} d_{i, F_q}
        # =================================================
        PF = torch.empty(
            (self._matrix_n, self._k),
            dtype=self._distance_values.dtype,
            device=self._device
        )

        rho = self._compute_rho_values()

        for q in range(self._k):
            Dq = D_minus_q[:, q]
            base = torch.empty(
                self._matrix_n,
                dtype=self._distance_values.dtype,
                device=self._device
            )

            # base[j] = sum_i d_ij over clients satisfying d_ij <= D^-_{qi}(t)
            for start in range(0, self._matrix_n, chunk_size):
                end = min(start + chunk_size, self._matrix_n)
                block = self._distance_values[:, start:end]

                base[start:end] = torch.where(
                    block <= Dq[:, None],
                    block,
                    torch.zeros_like(block)
                ).sum(dim=0)

            # For candidate facility j, Dq[j] is D^-_{qj}(t).
            delta = self._gamma * torch.clamp(rho[q] - Dq, min=0.0)

            PF[:, q] = base + delta

        self._facility_inner_values = PF

    def _update_facility(self):
        # =================================================
        previous_facilities = list(self._active_facility_list)
        # =================================================
        self._facility_activation_values = torch.zeros(
            size=self._size, dtype=torch.int, device=self._device
        )
        chosen = []
        # =================================================
        used_facilities = set()
        # =================================================

        for q in range(self._k):
            costs = self._facility_inner_values[:, q].clone()

            # =================================================
            # Enforce distinct facilities across clusters.
            # =================================================
            if len(used_facilities) > 0:
                used_indices = torch.tensor(
                    sorted(used_facilities),
                    dtype=torch.long,
                    device=self._device
                )
                costs[used_indices] = float("inf")

            min_cost = torch.min(costs)
            current_j = previous_facilities[q]

            # Tie-aware facility reassignment:
            # keep the current facility if it is still feasible and tied for best.
            if current_j not in used_facilities and costs[current_j] <= min_cost + self._epsilon:
                j = current_j
            else:
                j = torch.argmin(costs).item()
            # =================================================

            chosen.append(j)
            used_facilities.add(j)
            self._facility_activation_values[j, q] = 1

        self._active_facility_list = chosen

    def _final_cleanup_phase(self):
        cleanup_iterations = 0
        # =================================================
        seen_cleanup_states = set()
        # =================================================
        while True:
            previous_facilities = list(self._active_facility_list)

            # Reassign each client to its closest currently selected facility.
            facility_tensor = torch.tensor(self._active_facility_list, device=self._device)
            client_to_facility_distances = self._distance_values[:, facility_tensor]
            closest_cluster = torch.argmin(client_to_facility_distances, dim=1)
            self._client_activation_values = torch.zeros(
                size=self._size, dtype=torch.int, device=self._device
            )
            self._client_activation_values[self._math_row_indices, closest_cluster] = 1

            # Update each facility to the 1-median of its assigned clients.
            self._facility_activation_values = torch.zeros(
                size=self._size, dtype=torch.int, device=self._device
            )
            new_facilities = []
            # =================================================
            used_facilities = set()
            # =================================================

            for q in range(self._k):
                assigned_clients = self._client_activation_values[:, q].to(self._distance_values.dtype)

                if torch.sum(assigned_clients).item() == 0:
                    # =================================================
                    # Keep previous facility if possible; otherwise choose
                    # the first available facility to preserve distinctness.
                    # =================================================
                    if previous_facilities[q] not in used_facilities:
                        j = previous_facilities[q]
                    else:
                        available = [idx for idx in range(self._matrix_n) if idx not in used_facilities]
                        j = available[0]
                    # =================================================
                else:
                    costs = (self._distance_values.t() @ assigned_clients).clone()

                    # =================================================
                    # Enforce distinct facilities in cleanup too.
                    # =================================================
                    if len(used_facilities) > 0:
                        used_indices = torch.tensor(
                            sorted(used_facilities),
                            dtype=torch.long,
                            device=self._device
                        )
                        costs[used_indices] = float("inf")

                    min_cost = torch.min(costs)
                    if previous_facilities[q] not in used_facilities and costs[previous_facilities[q]] <= min_cost + self._epsilon:
                        j = previous_facilities[q]
                    else:
                        j = torch.argmin(costs).item()
                    # =================================================

                new_facilities.append(j)
                used_facilities.add(j)
                self._facility_activation_values[j, q] = 1

            self._active_facility_list = new_facilities
            cleanup_iterations += 1

            # =================================================
            cleanup_state = tuple(new_facilities)
            if cleanup_state in seen_cleanup_states:
                print("Repeated cleanup state detected; stopping cleanup.")
                break
            seen_cleanup_states.add(cleanup_state)
            # =================================================

            if new_facilities == previous_facilities:
                break

        print(f"Cleanup converged in {cleanup_iterations} iterations.")

    def _calculate_facilities_and_distance(self):
        selected_facilities = list(self._active_facility_list)
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        return selected_facilities, selected_distance
