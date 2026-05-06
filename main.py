import os
import time

from ExperimentManager import ExperimentManager
from solvers_alg.ZhangKFSolver import ZhangKFSolver
from reader.CKMPJSONCoordinateReader import CKMPJSONCoordinateReader
from reader.CKMPJSONDistanceReader import CKMPJSONDistanceReader
from reader.KCPJSONCoordinateReader import KCPJSONCoordinateReader
from reader.KCPJSONDistanceReader import KCPJSONDistanceReader
from reader.KFJSONCoordinateReader import KFJSONCoordinateReader
from reader.KFJSONDistanceReader import KFJSONDistanceReader
from reader.KMPJSONCoordinateReader import KMPJSONCoordinateReader
from reader.KMPJSONDistanceReader import KMPJSONDistanceReader
from solvers_alg.AryaMultiSolver import AryaMultiSolver
from solvers_alg.CohenAddadMultiSolver import CohenAddadMultiSolver
from solvers_alg.CohenAddadSolver import CohenAddadSolver
from solvers_alg.DominguezAlgorithmSolver import DominguezAlgorithmSolver
from solvers_alg.DropWorstFacilityKCenterSolver import \
    DropWorstFacilityKCenterSolver
from solvers_alg.FarthestClientReassignmentKCenterSolver import \
    FarthestClientReassignmentKCenterSolver
from solvers_alg.FarthestFirstKCenterSolver import FarthestFirstKCenterSolver
from solvers_alg.GreedyAddRemoveKCenterSolver import \
    GreedyAddRemoveKCenterSolver
from solvers_alg.HaralampievAlgorithmSolver import HaralampievAlgorithmSolver
from solvers_alg.HopfieldAlgorithmSolver import HopfieldAlgorithmSolver
from solvers_alg.HopfieldBestHalfMultiAlgorithmSolver import \
    HopfieldBestHalfMultiSolver
from solvers_alg.HopfieldBestHalfSecondClosestAlgorithmSolver import \
    HopfieldBestHalfSecondClosestAlgorithmSolver
from solvers_alg.HopfieldBestHalfSingleSolver import \
    HopfieldBestHalfSingleSolver
from solvers_alg.HopfieldExhaustiveAlgorithmSolver import \
    HopfieldExhaustiveAlgorithmSolver
from solvers_alg.HopfieldOriginal2nkCKMPSolver import \
    HopfieldOriginal2nkCKMPSolver
from solvers_alg.HopfieldOriginal2nkSolver import HopfieldOriginalSolver
from solvers_alg.HopfieldOriginal2nkSolverKCenter import \
    HopfieldOriginal2nkSolverKCenter
from solvers_alg.HopfieldParallelCKMSolver import HopfieldParallelCKMSolver
from solvers_alg.HopfieldParallelKCPSolver import HopfieldParallelKCPSolver
from solvers_alg.HopfieldParallelKFSolver import HopfieldParallelKFSolver
from solvers_alg.HopfieldParallelSolver import HopfieldParallelSolver
from solvers_alg.HopfieldSecondParallelSolver import \
    HopfieldSecondParallelSolver
from solvers_alg.HopfieldThirdParallelSolver import HopfieldThirdParallelSolver
from solvers_alg.InterchangeAlgorithmSolver import InterchangeAlgorithmSolver
from solvers_alg.LocalRecenterKCenterSolver import LocalRecenterKCenterSolver
from solvers_alg.LocalSearchSolver import LocalSearchSolver
from solvers_alg.LocalSearchSolverKCenter import LocalSearchSolverKCenter
from solvers_alg.PAMSolver import PAMSolver
from solvers_alg.RandomizedSwapKCenterSolver import RandomizedSwapKCenterSolver
from solvers_alg.SameiSolisObaKFSolver import SameiSolisObaKFSolver
from solvers_alg.ZhuAlgorithmSolver import ZhuAlgorithmSolver
from utils.user_input import get_input_arguments

# -------------------------------
# 🔹 Utility functions
# -------------------------------

def get_data_path(key, problem_family):
    """Return the full path to a dataset folder based on its key, using paths.txt mappings."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if problem_family == "1":
        resources_dir = os.path.join(base_dir, 'resources/KMP')
    elif problem_family == "2":
        resources_dir = os.path.join(base_dir, 'resources/KCP')
    elif problem_family == "3":
        resources_dir = os.path.join(base_dir, 'resources/CKMP')
    elif problem_family == "4":
        resources_dir = os.path.join(base_dir, 'resources/KF')
    else:
        raise ValueError(f"Invalid problem_family: {problem_family}")
    paths_file = os.path.join(resources_dir, 'paths.txt')

    # Read mappings from paths.txt
    paths = {}
    with open(paths_file, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                paths[k.strip()] = v.strip()

    # Get the relative path for the given key
    relative_path = paths.get(str(key))
    if not relative_path:
        raise ValueError(f"No dataset path found for key {key}")

    return os.path.join(resources_dir, relative_path)

def load_problems(dataset_path, dataset_key, use_gpu, problem_family):
    """
    Loads problem files for a given dataset key.
    - Nested directories for case 1,2,3 → use KMPJSONCoordinateReader
    - Flat files for case 4,5,6 → use KMPJSONDistanceReader
    """
    if problem_family == "1":
        # K-Median Problem
        if dataset_key in ["1", "2", "3"]:
            # Nested coordinate datasets
            problems = {}

            for subdir in sorted(os.listdir(dataset_path)):
                subdir_path = os.path.join(dataset_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                problems[subdir] = []
                for filename in os.listdir(subdir_path):
                    if not filename.endswith(".json"):
                        continue
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        reader = KMPJSONCoordinateReader()
                        if reader.canRead(file_path):
                            problem = reader.parse(file_path, use_gpu)
                            problems[subdir].append(problem)
                        else:
                            print(f"⚠️ Skipped {file_path}: cannot read")
                    except Exception as e:
                        print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["4", "6"]:
            # Flat distance datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = KMPJSONDistanceReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["5"]:
            # Flat coordinate datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = KMPJSONCoordinateReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        
    elif problem_family == "2":
        # K-Center Problem
        if dataset_key in ["1", "2", "3"]:
            # Nested coordinate datasets
            problems = {}

            for subdir in sorted(os.listdir(dataset_path)):
                subdir_path = os.path.join(dataset_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                problems[subdir] = []
                for filename in os.listdir(subdir_path):
                    if not filename.endswith(".json"):
                        continue
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        reader = KCPJSONCoordinateReader()
                        if reader.canRead(file_path):
                            problem = reader.parse(file_path, use_gpu)
                            problems[subdir].append(problem)
                        else:
                            print(f"⚠️ Skipped {file_path}: cannot read")
                    except Exception as e:
                        print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["4", "6"]:
            # Flat distance datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = KCPJSONDistanceReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["5"]:
            # Flat coordinate datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = KCPJSONCoordinateReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        
    elif problem_family == "3":
        # K-Center Problem
        if dataset_key in ["1", "2", "3"]:
            # Nested coordinate datasets
            problems = {}

            for subdir in sorted(os.listdir(dataset_path)):
                subdir_path = os.path.join(dataset_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                problems[subdir] = []
                for filename in os.listdir(subdir_path):
                    if not filename.endswith(".json"):
                        continue
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        reader = CKMPJSONCoordinateReader()
                        if reader.canRead(file_path):
                            problem = reader.parse(file_path, use_gpu)
                            problems[subdir].append(problem)
                        else:
                            print(f"⚠️ Skipped {file_path}: cannot read")
                    except Exception as e:
                        print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["4", "6"]:
            # Flat distance datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = CKMPJSONDistanceReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["5"]:
            # Flat coordinate datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = CKMPJSONCoordinateReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        
    elif problem_family == "4":
        # K-Center Problem
        if dataset_key in ["1", "2", "3"]:
            # Nested coordinate datasets
            problems = {}

            for subdir in sorted(os.listdir(dataset_path)):
                subdir_path = os.path.join(dataset_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                problems[subdir] = []
                for filename in os.listdir(subdir_path):
                    if not filename.endswith(".json"):
                        continue
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        reader = KFJSONCoordinateReader()
                        if reader.canRead(file_path):
                            problem = reader.parse(file_path, use_gpu)
                            problems[subdir].append(problem)
                        else:
                            print(f"⚠️ Skipped {file_path}: cannot read")
                    except Exception as e:
                        print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["4", "6"]:
            # Flat distance datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = KFJSONDistanceReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems
        elif dataset_key in ["5"]:
            # Flat coordinate datasets
            problems = []

            for filename in sorted(os.listdir(dataset_path)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(dataset_path, filename)
                try:
                    reader = CKMPJSONCoordinateReader()
                    if reader.canRead(file_path):
                        problem = reader.parse(file_path, use_gpu)
                        problems.append(problem)
                    else:
                        print(f"⚠️ Skipped {file_path}: cannot read")
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
            return problems

# -------------------------------
# 🔹 Main script
# -------------------------------


if __name__ == '__main__':

    # Get input arguments
    args = get_input_arguments()
    solver = None

    match args['problem_family']:
        case "1":
            match args['algorithm']:
                case "1":
                    solver = HopfieldAlgorithmSolver(use_gpu=args["use_gpu"])

                case "2":
                    solver = HaralampievAlgorithmSolver(temperature=args["parameters"]["temperature"], epoch_length=args["parameters"]["epoch"], decay_interval=args["parameters"]["decay"])

                case "3":
                    solver = LocalSearchSolver(max_time=args["parameters"]["max_time"])

                case "4":
                    solver = ZhuAlgorithmSolver(max_time=args["parameters"]["max_time"])

                case "5":
                    solver = AryaMultiSolver(max_time=args["parameters"]["max_time"])

                case "6":
                    solver = CohenAddadSolver(max_time=args["parameters"]["max_time"])

                case "7":
                    solver = CohenAddadMultiSolver(max_time=args["parameters"]["max_time"])

                case "8":
                    solver = HopfieldOriginalSolver(use_gpu=args["use_gpu"])

                case "9":
                    solver = HopfieldBestHalfSingleSolver(use_gpu=args["use_gpu"])

                case "10":
                    solver = HopfieldBestHalfMultiSolver(use_gpu=args["use_gpu"])

                case "11":
                    solver = HopfieldBestHalfSecondClosestAlgorithmSolver(use_gpu=args["use_gpu"])

                case "12":
                    solver = HopfieldExhaustiveAlgorithmSolver(use_gpu=args["use_gpu"])

                case "13":
                    solver = InterchangeAlgorithmSolver(use_gpu=args["use_gpu"])

                case "14":
                    solver = DominguezAlgorithmSolver(use_gpu=args["use_gpu"])

                case "15":
                    solver = HopfieldParallelSolver(use_gpu=args["use_gpu"], seed=13)

                case "16":
                    solver = PAMSolver(use_gpu=args["use_gpu"])

                case "17":
                    solver = HopfieldSecondParallelSolver(use_gpu=args["use_gpu"], seed=99)

                case "18":
                    solver = HopfieldThirdParallelSolver(use_gpu=args["use_gpu"], seed=99)

        case "2":
            match args['algorithm']:
                case "1":
                    solver = HopfieldOriginal2nkSolverKCenter(use_gpu=args["use_gpu"])

                case "2":
                    solver = LocalSearchSolverKCenter(run_time=args["parameters"]["runs"])

                case "3":
                    solver = HopfieldParallelKCPSolver(use_gpu=args["use_gpu"])

                case "4":
                    solver = FarthestFirstKCenterSolver(run_time=args["parameters"]["runs"])

                case "5":
                    solver = FarthestClientReassignmentKCenterSolver()

                case "6":
                    solver = GreedyAddRemoveKCenterSolver()

                case "7":
                    solver = DropWorstFacilityKCenterSolver()

                case "8":
                    solver = LocalRecenterKCenterSolver()

                case "9":
                    solver = RandomizedSwapKCenterSolver()

        case "3":
            match args['algorithm']:
                case "1":
                    solver = HopfieldOriginal2nkCKMPSolver(use_gpu=args["use_gpu"])
                case "2":
                    solver = HopfieldParallelCKMSolver(use_gpu=args["use_gpu"])

        case "4":
            match args['algorithm']:
                case "1":
                    solver = HopfieldParallelKFSolver(use_gpu=args["use_gpu"])

                case "2":
                    solver = SameiSolisObaKFSolver(swap_size=args["parameters"]["swap_size"])

                case "3":
                    solver = ZhangKFSolver(swap_size=args["parameters"]["swap_size"], epsilon_prime=args["parameters"]["epsilon_prime"])

    problem_family = args["problem_family"]

    dataset_key = args["dataset"]
    dataset_path = get_data_path(dataset_key, problem_family)

    match dataset_key:
        case "1": print(f"\nTesting algorithm {solver.getName()} on dataset Random-Small")

        case "2": print(f"\nTesting algorithm {solver.getName()} on dataset Random-Large")

        case "3": print(f"\nTesting algorithm {solver.getName()} on dataset USCA312")

        case "4": print(f"\nTesting algorithm {solver.getName()} on dataset P-Median")

        case "5": print(f"\nTesting algorithm {solver.getName()} on dataset TSPLib")

        case "6": print(f"\nTesting algorithm {solver.getName()} on dataset Special")

    problems = load_problems(dataset_path, dataset_key, args["use_gpu"], problem_family)

    runs = args["parameters"].get("runs", 1)
    manager = ExperimentManager(problems, solver, problem_family, runs)
    manager.run(dataset_key)