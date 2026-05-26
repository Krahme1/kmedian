"""
Initialization study for the original 2nk Hopfield k-median solver.
"""

import argparse
import contextlib
import io
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from reader.KMPJSONCoordinateReader import KMPJSONCoordinateReader
from reader.KMPJSONDistanceReader import KMPJSONDistanceReader
from solvers_alg.solvers.brute_solver import calculate_distance
from solvers_alg.KMP.Main.ARN import HopfieldOriginalSolver

from initializers.uniform_initializer import UniformInitializer
from initializers.dq_sampling_initializer import DQSamplingInitializer
from initializers.weighted_d_initializer import WeightedDInitializer
from initializers.pam_build_initializer import PAMBuildInitializer


PATHS_FILE = os.path.join("resources", "KMP", "paths.txt")


def read_dataset_paths():
    dataset_paths = {}

    with open(PATHS_FILE, "r") as file:
        for line in file:
            line = line.strip()

            if line == "" or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            dataset_paths[key.strip()] = value.strip()

    return dataset_paths


def get_dataset_path(dataset_key):
    dataset_key = str(dataset_key)
    dataset_paths = read_dataset_paths()

    if dataset_key not in dataset_paths:
        raise ValueError(
            f"Dataset key {dataset_key} was not found in {PATHS_FILE}. "
            f"Available keys: {sorted(dataset_paths.keys())}"
        )

    return os.path.join("resources", "KMP", dataset_paths[dataset_key])


def get_reader_for_dataset(dataset_key):
    dataset_key = str(dataset_key)

    # Based on the dataset folders in resources/KMP/paths.txt.
    # Coordinate-style JSON datasets.
    coordinate_reader_keys = ["1", "2", "3", "5"]

    # Distance-matrix JSON datasets.
    distance_reader_keys = ["4", "6", "7", "8", "9", "10", "11"]

    if dataset_key in coordinate_reader_keys:
        return KMPJSONCoordinateReader()

    if dataset_key in distance_reader_keys:
        return KMPJSONDistanceReader()

    raise ValueError(f"No reader rule for dataset key {dataset_key}")


def load_kmedian_problems(dataset_key, use_gpu):
    dataset_path = get_dataset_path(dataset_key)
    reader = get_reader_for_dataset(dataset_key)
    problems = []

    for root, _, files in os.walk(dataset_path):
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(root, filename)

            try:
                if reader.canRead(file_path):
                    problem = reader.parse(file_path, use_gpu)
                    dataset_group = infer_dataset_group(dataset_key, file_path, problem.getName())
                    problems.append((dataset_group, file_path, filename, problem))
                else:
                    print(f"Skipped unreadable file: {file_path}")

            except Exception as error:
                print(f"Failed to read {file_path}: {type(error).__name__}: {error}")

    return problems


def infer_dataset_group(dataset_key, file_path, problem_name):
    dataset_key = str(dataset_key)
    text = f"{file_path} {problem_name}".lower()

    if dataset_key == "1":
        return "random_geometric"
    if dataset_key == "2":
        return "random_huge"
    if dataset_key == "3":
        return "usca312"
    if dataset_key == "4":
        return "pmed"
    if dataset_key == "5":
        return "tsplib"
    if dataset_key == "6":
        return "special"
    if dataset_key == "7":
        return "barabasi"
    if dataset_key == "8":
        return "erdos_renyi"
    if dataset_key == "9":
        if "grid" in text:
            return "grid"
        return "path_grid"
    if dataset_key == "10":
        if "balanced_tree" in text:
            return "balanced_tree"
        if "random_tree" in text:
            return "random_tree"
        return "trees"

    if dataset_key == "11":
        return "sbm"

    return "unknown"


def get_distance_matrix(problem):
    graph = problem.getGraph()

    if hasattr(graph, "_distances"):
        D = graph._distances
    elif hasattr(graph, "_normalized_distances"):
        D = graph._normalized_distances
    else:
        n = problem.getN()
        D = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                D[i][j] = graph.get_standard_distance(i, j)

        return D

    if isinstance(D, torch.Tensor):
        return D.detach().cpu().numpy().astype(float)

    return np.asarray(D, dtype=float)


def get_distance(graph, u, v):
    return graph.get_standard_distance(u, v)


def make_initializers(seed=None):
    return [
        UniformInitializer(seed=seed),
        DQSamplingInitializer(0),
        DQSamplingInitializer(0.5),
        DQSamplingInitializer(1),
        DQSamplingInitializer(1.5),
        DQSamplingInitializer(2),
        DQSamplingInitializer(10),
        WeightedDInitializer("degree"),
        WeightedDInitializer("closeness"),
        WeightedDInitializer("density"),
        PAMBuildInitializer(),
    ]


def get_initial_facilities(initializer, problem):
    return initializer.initialize(
        problem.getGraph(),
        problem.getN(),
        problem.getK(),
    )


def assign_clients_to_facilities(D, facilities):
    clusters = {facility: [] for facility in facilities}

    for client in range(D.shape[0]):
        best_facility = min(facilities, key=lambda facility: D[client][facility])
        clusters[best_facility].append(client)

    return clusters


def compute_average_nearest_neighbor_radius(D):
    nearest_distances = []
    n = D.shape[0]

    for u in range(n):
        best = float("inf")

        for v in range(n):
            if u == v:
                continue

            if D[u][v] < best:
                best = D[u][v]

        if best < float("inf"):
            nearest_distances.append(best)

    if len(nearest_distances) == 0:
        return 0.0

    return float(np.mean(nearest_distances))


def estimate_degree_from_distances(D, u):
    distances = []

    for v in range(D.shape[0]):
        if u == v:
            continue

        d = D[u][v]

        if d > 0:
            distances.append(d)

    if len(distances) == 0:
        return 0

    min_positive = min(distances)
    return sum(1 for d in distances if abs(d - min_positive) < 1e-9)


def compute_initial_features(D, facilities, optimal=None):
    n = D.shape[0]
    k = len(facilities)

    nearest_distances = []

    for client in range(n):
        nearest_distances.append(
            min(D[client][facility] for facility in facilities)
        )

    nearest_distances = np.array(nearest_distances, dtype=float)
    initial_cost = float(np.sum(nearest_distances))

    pairwise_distances = []

    for i in range(k):
        for j in range(i + 1, k):
            pairwise_distances.append(D[facilities[i]][facilities[j]])

    pairwise_distances = np.array(pairwise_distances, dtype=float)

    clusters = assign_clients_to_facilities(D, facilities)
    cluster_sizes = [len(clusters[facility]) for facility in facilities]

    proportions = [size / n for size in cluster_sizes if size > 0]
    entropy = -sum(p * math.log(p) for p in proportions)

    r = compute_average_nearest_neighbor_radius(D)

    degrees = [estimate_degree_from_distances(D, facility) for facility in facilities]

    closeness_values = []
    density_values = []

    for facility in facilities:
        total_distance = float(np.sum(D[facility]))

        if total_distance > 0:
            closeness_values.append(1 / total_distance)
        else:
            closeness_values.append(0)

        density_values.append(
            sum(1 for u in range(n) if D[facility][u] <= r)
        )

    if optimal is None or optimal == 0:
        initial_ratio = ""
    else:
        initial_ratio = float(initial_cost / optimal)

    return {
        "initial_cost_F_S0": initial_cost,
        "initial_ratio_F_S0_OPT": initial_ratio,
        "f1_min_center_separation": float(np.min(pairwise_distances)) if len(pairwise_distances) > 0 else 0.0,
        "f2_avg_center_separation": float(np.mean(pairwise_distances)) if len(pairwise_distances) > 0 else 0.0,
        "f3_coverage_radius": float(np.max(nearest_distances)) if len(nearest_distances) > 0 else 0.0,
        "f4_cluster_size_variance": float(np.var(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0,
        "f5_cluster_entropy": float(entropy),
        "f6_avg_degree": float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
        "f7_avg_closeness": float(np.mean(closeness_values)) if len(closeness_values) > 0 else 0.0,
        "f8_avg_local_density": float(np.mean(density_values)) if len(density_values) > 0 else 0.0,
    }


def run_solver_with_initial_facilities(problem, initial_facilities, use_gpu):
    solver = HopfieldOriginalSolver(use_gpu)
    solver.initialize(problem)

    start_time = time.time()

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        solver.solve(starter_facilities=initial_facilities)

    runtime = time.time() - start_time
    output_text = output.getvalue()

    match = re.search(r"Converged in (\d+) iterations", output_text)
    iterations = int(match.group(1)) if match else ""

    return solver.getSelectedFacilities(), solver.getSolutionValue(), iterations, runtime


def run_one_experiment(dataset_group, file_path, filename, problem, initializer, use_gpu, run_id, seed=None):
    n = problem.getN()
    k = problem.getK()
    graph = problem.getGraph()
    optimal = problem.getOptimal()
    D = get_distance_matrix(problem)

    initial_facilities = get_initial_facilities(initializer, problem)
    initial_features = compute_initial_features(D, initial_facilities, optimal)

    final_facilities, final_cost, iterations, runtime = run_solver_with_initial_facilities(
        problem,
        initial_facilities,
        use_gpu,
    )

    if final_cost is None:
        final_cost = calculate_distance(graph, final_facilities, n)

    if optimal is None or optimal == 0:
        final_ratio = ""
    else:
        final_ratio = float(final_cost / optimal)

    initial_cost = initial_features["initial_cost_F_S0"]
    improvement = float(initial_cost - final_cost)

    if initial_cost == 0:
        improvement_percent = ""
    else:
        improvement_percent = float(improvement / initial_cost)

    initial_set = set(initial_facilities)
    final_set = set(final_facilities)
    kept_facilities = len(initial_set.intersection(final_set))
    replaced_facilities = k - kept_facilities

    row = {
        "dataset": dataset_group,
        "instance": problem.getName(),
        "file_name": filename,
        "file_path": file_path,
        "n": n,
        "k": k,
        "optimal_OPT": optimal,
        "initializer": initializer.name,
        "run_id": run_id,
        "seed": seed if seed is not None else "",
        "initial_facilities_S0": initial_facilities,
        "final_facilities_SH": final_facilities,
        "final_cost_F_SH": float(final_cost),
        "final_ratio_F_SH_OPT": final_ratio,
        "improvement_F_S0_minus_F_SH": improvement,
        "improvement_F_S0_minus_F_SH_over_F_S0": improvement_percent,
        "kept_facilities_size_intersection": kept_facilities,
        "replaced_facilities": replaced_facilities,
        "iterations": iterations,
        "runtime": float(runtime),
    }

    row.update(initial_features)
    return row


FIELDNAMES = [
    "dataset",
    "instance",
    "file_name",
    "file_path",
    "n",
    "k",
    "optimal_OPT",
    "initializer",
    "run_id",
    "seed",
    "initial_facilities_S0",
    "final_facilities_SH",
    "initial_cost_F_S0",
    "final_cost_F_SH",
    "initial_ratio_F_S0_OPT",
    "final_ratio_F_SH_OPT",
    "improvement_F_S0_minus_F_SH",
    "improvement_F_S0_minus_F_SH_over_F_S0",
    "kept_facilities_size_intersection",
    "replaced_facilities",
    "f1_min_center_separation",
    "f2_avg_center_separation",
    "f3_coverage_radius",
    "f4_cluster_size_variance",
    "f5_cluster_entropy",
    "f6_avg_degree",
    "f7_avg_closeness",
    "f8_avg_local_density",
    "iterations",
    "runtime",
]


def safe_sheet_name(name):
    invalid = ["\\", "/", "*", "[", "]", ":", "?"]

    for char in invalid:
        name = name.replace(char, "_")

    return name[:31]


def prepare_rows_for_excel(rows):
    prepared_rows = []

    for row in rows:
        row = row.copy()
        row["initial_facilities_S0"] = " ".join(map(str, row["initial_facilities_S0"]))
        row["final_facilities_SH"] = " ".join(map(str, row["final_facilities_SH"]))
        prepared_rows.append(row)

    return prepared_rows


def save_results(rows, output_file):
    if len(rows) == 0:
        print("No results were produced.")
        return

    rows = prepare_rows_for_excel(rows)
    df = pd.DataFrame(rows)

    for field in FIELDNAMES:
        if field not in df.columns:
            df[field] = ""

    df = df[FIELDNAMES]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        for (dataset, initializer), group in df.groupby(["dataset", "initializer"]):
            sheet_name = safe_sheet_name(f"{dataset}_{initializer}")
            group.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nSaved results to {output_file}")


def get_number_of_runs(initializer, runs):
    # PAM BUILD is deterministic, so one run is enough unless the user explicitly wants repeated duplicates.
    if initializer.name == "pam_build":
        return 1

    return runs


def run_experiments(dataset_key, output_file, use_gpu, runs, limit, base_seed):
    problems = load_kmedian_problems(str(dataset_key), use_gpu)

    if limit is not None:
        problems = problems[:limit]

    rows = []

    print(f"Loaded {len(problems)} problems for dataset key {dataset_key}.")

    for problem_index, (dataset_group, file_path, filename, problem) in enumerate(problems, start=1):
        print("\n==============================")
        print(
            f"Problem {problem_index}/{len(problems)}: "
            f"{problem.getName()}  n={problem.getN()}  k={problem.getK()}  dataset={dataset_group}"
        )
        print("==============================")

        for initializer_template in make_initializers(seed=base_seed):
            number_of_runs = get_number_of_runs(initializer_template, runs)

            for run_id in range(number_of_runs):
                seed = None if base_seed is None else base_seed + run_id

                # Re-create the initializers each run so seeded uniform runs can vary by run_id.
                initializer = next(
                    item for item in make_initializers(seed=seed)
                    if item.name == initializer_template.name
                )

                try:
                    print(
                        f"Running {initializer.name}, run {run_id + 1}/{number_of_runs}"
                    )

                    row = run_one_experiment(
                        dataset_group=dataset_group,
                        file_path=file_path,
                        filename=filename,
                        problem=problem,
                        initializer=initializer,
                        use_gpu=use_gpu,
                        run_id=run_id,
                        seed=seed,
                    )

                    rows.append(row)

                    ratio_text = row["final_ratio_F_SH_OPT"]
                    if ratio_text == "":
                        ratio_text = "NA"
                    else:
                        ratio_text = f"{ratio_text:.3f}"

                    print(
                        f"Done: initial={row['initial_cost_F_S0']:.3f}, "
                        f"final={row['final_cost_F_SH']:.3f}, "
                        f"ratio={ratio_text}, "
                        f"iterations={row['iterations']}, "
                        f"time={row['runtime']:.3f}s"
                    )

                except Exception as error:
                    print(
                        f"⚠️ Skipped {problem.getName()} with {initializer.name}: "
                        f"{type(error).__name__}: {error}"
                    )

    save_results(rows, output_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_key", default="4")
    parser.add_argument("--output", default="initialization_study_results.xlsx")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)

    args = parser.parse_args()

    run_experiments(
        dataset_key=args.dataset_key,
        output_file=args.output,
        use_gpu=args.use_gpu,
        runs=args.runs,
        limit=args.limit,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()