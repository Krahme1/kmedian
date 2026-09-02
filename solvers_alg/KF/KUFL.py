"""Resonance-network solver for metric k-uncapacitated facility location.

This module is a Section 2.4 adaptation of the original ARN.py implementation
for k-median.  It implements the three local-search moves represented by the
facility neurons in Section 2.4:

    real facility -> real node   : SWAP, equation (8)
    real facility -> dummy node  : CLOSE, equation (9)
    dummy node    -> real node   : OPEN, equation following (9)

The implementation keeps exactly k cluster slots and augments the n real
facility candidates with k-1 dummy facility candidates.  Dummy facilities are
placeholders only: they are never clients and no client is assigned to a
cluster whose facility is dummy.

The code uses normalized distances inside the neural
potentials but reports the objective with the graph's original distances.
Therefore this implementation rescales raw facility opening costs by the same
min-max distance scale before inserting them into equations (8)-(9).  This
keeps a positive neural activation equivalent to an improvement in the
original distance-plus-opening-cost objective.  If a data set already stores
facility costs in normalized-distance units, set
``facility_costs_in_normalized_units=True``.
"""

import random
from typing import Iterable, Optional

import torch

from problems.KFProblem import KFProblem
from solvers_alg.KF.KFSolver import KFSolver
from solvers_alg.solvers.brute_solver import calculate_distance_with_facility_cost


FACILITY = 1
CLIENT = 0


class HopfieldOriginalSolver(KFSolver):
    """Section 2.4 ARN solver for the k-uncapacitated facility-location problem.

    The class name is KFSolver so
    that code that instantiates ``HopfieldOriginalSolver`` needs only to import
    this module from the KF solver package instead of the KMP package.
    """

    def __init__(
        self,
        use_gpu: bool,
        activation_tolerance: float = 1e-9,
        facility_costs_in_normalized_units: bool = False,
    ):
        self._use_gpu = use_gpu
        self._device = None
        self._activation_tolerance = float(activation_tolerance)
        self._facility_costs_in_normalized_units = bool(
            facility_costs_in_normalized_units
        )

        self._problem = None
        self._graph = None
        self._n = None
        self._k = None
        self._name = "Hopfield (Section 2.4 k-UFL)"
        self._num_rows = None
        self._num_cols = None
        self._size = None
        self._math_row_indices = None
        self._k_indices = None
        self._full_distance_values = None
        self._candidatefacility_inner_values = None

        self._num_dummy_facilities = None
        self._dummy_start_index = None
        self._num_facility_candidates = None

        self._costs = None
        self._cost_tensor = None
        self._potential_cost_tensor = None
        self._distance_normalization_scale = 1.0

        self._distance_values = None
        self._client_inner_values = None
        self._facility_inner_values = None
        self._client_activation_values = None
        self._facility_activation_values = None
        self._facilities = None
        self._active_facility_list = None

        self._selectedFacilities = []
        self._solutionValue = None
        self._iterations = 0
        self.verbose = False

    def initialize(self, problem: KFProblem):
        """Initialize tensors that depend only on the KF instance."""
        self._problem = problem
        self._n = int(problem.getN())
        self._k = int(problem.getK())
        self._graph = problem.getGraph()
        self._costs = problem.getCosts()

        if self._k < 1:
            raise ValueError("k must be at least 1")
        if self._k > self._n:
            raise ValueError(
                "Section 2.4 initializes k distinct real facilities, so k cannot exceed n"
            )

        if self._use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # D' = 1 - normalized distance, as in Section 2.1.
        self._distance_values = self._create_distance_array().to(
            device=self._device
        )

        self._cost_tensor = self._create_cost_tensor().to(
            device=self._device, dtype=self._distance_values.dtype
        )
        self._distance_normalization_scale = self._get_distance_normalization_scale()
        if self._facility_costs_in_normalized_units:
            self._potential_cost_tensor = self._cost_tensor.clone()
        else:
            self._potential_cost_tensor = (
                self._cost_tensor / self._distance_normalization_scale
            )

        # Section 2.4 adds k-1 dummy facility candidates.  They occupy indices
        # n, ..., n+k-2.  They are not added to the client distance matrix.
        self._num_dummy_facilities = max(0, self._k - 1)
        self._dummy_start_index = self._n
        self._num_facility_candidates = self._n + self._num_dummy_facilities
        self._num_rows = self._num_facility_candidates
        self._num_cols = self._k
        self._size = (self._num_rows, self._num_cols)
        self._math_row_indices = torch.arange(self._n, device=self._device)
        self._k_indices = torch.arange(self._k, device=self._device)
        self._full_distance_values = self._distance_values

        self._client_inner_values = torch.full(
            (self._n, self._k),
            float("-inf"),
            dtype=self._distance_values.dtype,
            device=self._device,
        )

        # Here _facility_inner_values stores theta(f_{j,h}) for every candidate
        # j and cluster h.  Invalid moves are represented by -infinity.
        self._facility_inner_values = torch.full(
            (self._num_facility_candidates, self._k),
            float("-inf"),
            dtype=self._distance_values.dtype,
            device=self._device,
        )

        self._client_activation_values = torch.zeros(
            (self._n, self._k), dtype=torch.int64, device=self._device
        )
        self._facility_activation_values = torch.zeros(
            (self._num_facility_candidates, self._k),
            dtype=torch.int64,
            device=self._device,
        )
        self._facilities = torch.zeros(
            (1, self._num_facility_candidates),
            dtype=torch.int64,
            device=self._device,
        )
        self._active_facility_list = []

        self._selectedFacilities = []
        self._solutionValue = None
        self._iterations = 0

    def getName(self):
        return self._name

    def getSelectedFacilities(self):
        return self._selectedFacilities

    def getSolutionValue(self):
        return self._solutionValue

    def getIterations(self):
        return self._iterations

    def setN(self, n):
        self._n = n

    def setK(self, k):
        self._k = k

    def setGraph(self, graph):
        self._graph = graph

    def solve(
        self,
        runNum=None,
        starter_facilities: Optional[Iterable[int]] = None,
        verbose: Optional[bool] = None,
    ):
        """Run the Section 2.4 neural-network/local-search algorithm.

        The network starts with k distinct *real* facilities, as specified in
        Section 2.4.  Each iteration evaluates every feasible open, close, and
        swap operation and performs the one with maximum positive activation
        potential.  The network stabilizes when no positive potential remains.
        """
        # ``runNum`` is retained for drop-in compatibility with the supplied
        # ARN.py and the existing experiment driver; the Section 2.4 update
        # rule itself does not use it.
        _ = runNum
        if self._problem is None:
            raise ValueError("Call initialize(problem) before solve().")

        if verbose is not None:
            self.verbose = bool(verbose)
        self._initialize_per_run_arrays(starter_facilities)

        if self.verbose:
            _, initial_value = self._calculate_facilities_and_distance()
            print("Initial solution value:", initial_value)
            print("Initial real facilities:", self._real_active_facilities())

        self._iterations = 0

        while True:
            # Equations (8), (9), and the dummy->real opening potential.
            self._calculate_facility_values()
            best_value, best_candidate, best_cluster = self._best_facility_update()

            # Equations (10)-(11): if M <= 0, all facility neuron states stay
            # unchanged and the network is stable.
            if best_candidate is None or best_value <= self._activation_tolerance:
                break

            old_candidate = self._active_facility_list[best_cluster]
            self._update_facility(best_candidate, best_cluster)

            # Equations (2) and (4), with dummy clusters excluded from client
            # assignment exactly as specified in Section 2.4.
            self._calculate_client_values()
            self._update_client()

            self._iterations += 1

            if self.verbose:
                move = self._move_name(old_candidate, best_candidate)
                real_facilities, objective = self._calculate_facilities_and_distance()
                print(
                    f"Iteration {self._iterations}: {move} in cluster {best_cluster}; "
                    f"theta={best_value:.12g}; objective={objective}; "
                    f"real facilities={real_facilities}"
                )

        self._selectedFacilities, self._solutionValue = (
            self._calculate_facilities_and_distance()
        )

        if self.verbose:
            print("Stable after", self._iterations, "facility updates")
            print("Selected real facilities:", self._selectedFacilities)
            print("Final solution value:", self._solutionValue)

        return self._selectedFacilities

    def _initialize_per_run_arrays(self, starter_facilities=None):
        """Initialize exactly k real facilities and corresponding neuron states."""
        self._client_inner_values.fill_(float("-inf"))
        self._facility_inner_values.fill_(float("-inf"))
        self._client_activation_values.zero_()
        self._facility_activation_values.zero_()
        self._facilities.zero_()
        self._active_facility_list = []

        if starter_facilities is None:
            initial_set = random.sample(range(self._n), k=self._k)
        else:
            initial_set = [int(x) for x in starter_facilities]
            if len(initial_set) != self._k:
                raise ValueError(
                    f"starter_facilities must contain exactly k={self._k} real facilities"
                )
            if len(set(initial_set)) != self._k:
                raise ValueError("starter_facilities must be distinct")
            if any(j < 0 or j >= self._n for j in initial_set):
                raise ValueError(
                    "Section 2.4 requires all initial facilities to be real nodes in 0..n-1"
                )

        for h, facility in enumerate(initial_set):
            self._facility_activation_values[facility, h] = 1
            self._facilities[0, facility] = 1
            self._active_facility_list.append(facility)

        self._calculate_client_values()
        self._update_client()

    def _create_distance_array(self):
        """Return D' = 1 - normalized distances for the n real nodes."""
        if self._use_gpu and hasattr(self._graph, "_gpu_normalized_distances"):
            normalized_source = self._graph._gpu_normalized_distances
        else:
            normalized_source = self._graph._normalized_distances

        if isinstance(normalized_source, torch.Tensor):
            normalized = normalized_source.clone().detach().to(self._device)
        else:
            normalized = torch.as_tensor(
                normalized_source, dtype=torch.float32, device=self._device
            )

        if not normalized.is_floating_point():
            normalized = normalized.float()

        if normalized.shape != (self._n, self._n):
            raise ValueError(
                "graph._normalized_distances must be an n by n matrix for the real nodes"
            )
        return 1.0 - normalized

    def _create_cost_tensor(self):
        """Convert problem facility costs alpha_j to a length-n tensor."""
        if isinstance(self._costs, dict):
            values = [self._costs[i] for i in range(self._n)]
        else:
            values = list(self._costs)

        if len(values) != self._n:
            raise ValueError(
                f"Expected {self._n} facility costs, but received {len(values)}"
            )

        return torch.as_tensor(values, dtype=torch.float64)

    def _get_distance_normalization_scale(self):
        """Return the raw-distance units represented by one normalized unit.

        Section 2.1 applies min-max normalization before forming D'.  If
        d_norm = (d_raw - d_min) / scale, then differences of D' are raw
        distance improvements divided by ``scale``.  Raw opening costs must
        therefore be divided by the same scale before appearing in Eqs. (8)
        and (9).
        """
        raw = torch.as_tensor(self._graph._distances, dtype=torch.float64).detach().cpu()
        normalized = torch.as_tensor(
            self._graph._normalized_distances, dtype=torch.float64
        ).detach().cpu()

        raw_range = float((torch.max(raw) - torch.min(raw)).item())
        normalized_range = float(
            (torch.max(normalized) - torch.min(normalized)).item()
        )

        if raw_range <= 0.0 or normalized_range <= 0.0:
            return 1.0

        scale = raw_range / normalized_range
        if scale <= 0.0 or not torch.isfinite(torch.tensor(scale)):
            return 1.0
        return float(scale)

    def _calculate_client_values(self):
        """Compute client potentials theta(c_{i,h}) from Eq. (2).

        For a real facility F_h, theta(c_{i,h}) = D'[i,F_h].  For a dummy
        facility, the entire cluster column is -infinity so no client can be
        assigned to that cluster.
        """
        self._client_inner_values.fill_(float("-inf"))

        for h, facility in enumerate(self._active_facility_list):
            if self._is_real(facility):
                self._client_inner_values[:, h] = self._distance_values[:, facility]

    def _update_client(self):
        """Apply Eq. (4), breaking ties consistently via torch.argmax."""
        self._client_activation_values.zero_()
        max_indices = torch.argmax(self._client_inner_values, dim=1)
        rows = torch.arange(self._n, device=self._device)
        self._client_activation_values[rows, max_indices] = 1

    def _calculate_facility_values(self):
        """Compute all Section 2.4 facility activation potentials.

        The resulting matrix has one row per real-or-dummy candidate and one
        column per cluster.  Entries that do not represent a feasible open,
        close, or swap operation remain -infinity.
        """
        theta = self._facility_inner_values
        theta.fill_(float("-inf"))

        active_real_mask = torch.zeros(
            self._n, dtype=torch.bool, device=self._device
        )
        for facility in self._active_facility_list:
            if self._is_real(facility):
                active_real_mask[facility] = True
        inactive_real = torch.nonzero(~active_real_mask, as_tuple=False).flatten()

        active_dummy_mask = torch.zeros(
            self._num_dummy_facilities,
            dtype=torch.bool,
            device=self._device,
        )
        for facility in self._active_facility_list:
            if self._is_dummy(facility):
                active_dummy_mask[facility - self._dummy_start_index] = True
        inactive_dummy_offsets = torch.nonzero(
            ~active_dummy_mask, as_tuple=False
        ).flatten()

        for h, current in enumerate(self._active_facility_list):
            if self._is_real(current):
                # m_{i,h}: closest REAL facility other than F_h.
                backup = self._closest_real_similarity(exclude_cluster=h)

                # Loss in reward from closing/removing F_h while retaining the
                # backup facilities: second sum in Eqs. (8) and (9).
                removal_loss = torch.relu(
                    self._distance_values[:, current] - backup
                ).sum()

                # Eq. (8): real -> real SWAP.
                if inactive_real.numel() > 0:
                    insertion_gain = torch.relu(
                        self._distance_values[:, inactive_real]
                        - backup.unsqueeze(1)
                    ).sum(dim=0)
                    theta[inactive_real, h] = (
                        insertion_gain
                        - removal_loss
                        + self._potential_cost_tensor[current]
                        - self._potential_cost_tensor[inactive_real]
                    )

                # Eq. (9): real -> dummy CLOSE.  Every unused dummy represents
                # the same close move; all receive the same activation value.
                if inactive_dummy_offsets.numel() > 0:
                    inactive_dummies = (
                        inactive_dummy_offsets + self._dummy_start_index
                    )
                    theta[inactive_dummies, h] = (
                        self._potential_cost_tensor[current] - removal_loss
                    )

            else:
                # Current F_h is dummy.  m_i is the similarity to the closest
                # currently selected REAL facility.  Section 2.4 defines m_i=0
                # if there is no real facility; k-1 dummies prevent that state,
                # but the helper implements the definition explicitly anyway.
                current_best = self._closest_real_similarity(exclude_cluster=None)

                # Dummy -> real OPEN operation.
                if inactive_real.numel() > 0:
                    insertion_gain = torch.relu(
                        self._distance_values[:, inactive_real]
                        - current_best.unsqueeze(1)
                    ).sum(dim=0)
                    theta[inactive_real, h] = (
                        insertion_gain - self._potential_cost_tensor[inactive_real]
                    )

                # Dummy -> dummy is intentionally undefined/invalid.

    def _closest_real_similarity(self, exclude_cluster=None):
        """Return the vector m_i (or m_{i,h}) used in Section 2.4."""
        real_facilities = []
        for h, facility in enumerate(self._active_facility_list):
            if exclude_cluster is not None and h == exclude_cluster:
                continue
            if self._is_real(facility):
                real_facilities.append(facility)

        if not real_facilities:
            return torch.zeros(
                self._n,
                dtype=self._distance_values.dtype,
                device=self._device,
            )

        return torch.max(
            self._distance_values[:, real_facilities], dim=1
        ).values

    def _best_facility_update(self):
        """Return M and a deterministic (L, mu) attaining it."""
        flat = self._facility_inner_values.reshape(-1)
        if flat.numel() == 0:
            return float("-inf"), None, None

        max_value_tensor, flat_index_tensor = torch.max(flat, dim=0)
        max_value = float(max_value_tensor.item())

        if not torch.isfinite(max_value_tensor):
            return max_value, None, None

        flat_index = int(flat_index_tensor.item())
        # Row-major flattening of [candidate, cluster].
        candidate = flat_index // self._k
        cluster = flat_index % self._k
        return max_value, candidate, cluster

    def _update_facility(self, new_candidate: int, cluster: int):
        """Apply the facility neuron state update in Eqs. (10)-(11)."""
        old_candidate = self._active_facility_list[cluster]

        if old_candidate == new_candidate:
            return

        self._facility_activation_values[old_candidate, cluster] = 0
        self._facilities[0, old_candidate] = 0

        self._facility_activation_values[new_candidate, cluster] = 1
        self._facilities[0, new_candidate] = 1
        self._active_facility_list[cluster] = new_candidate

    def _calculate_facilities_and_distance(self):
        """Return real facilities and the original KF objective value."""
        selected_facilities = self._real_active_facilities()

        if not selected_facilities:
            # This cannot occur with k cluster slots and only k-1 distinct
            # dummies, but keep the failure explicit if the state is corrupted.
            raise RuntimeError("Invalid state: no real facility is selected")

        selected_value = calculate_distance_with_facility_cost(
            self._graph,
            selected_facilities,
            self._costs,
            self._n,
        )
        return selected_facilities, selected_value

    def _real_active_facilities(self):
        return [
            int(facility)
            for facility in self._active_facility_list
            if self._is_real(facility)
        ]

    def _is_real(self, candidate: int):
        return 0 <= int(candidate) < self._n

    def _is_dummy(self, candidate: int):
        return self._dummy_start_index <= int(candidate) < self._num_facility_candidates

    def _move_name(self, old_candidate: int, new_candidate: int):
        if self._is_real(old_candidate) and self._is_real(new_candidate):
            return "swap"
        if self._is_real(old_candidate) and self._is_dummy(new_candidate):
            return "close"
        if self._is_dummy(old_candidate) and self._is_real(new_candidate):
            return "open"
        return "invalid"
