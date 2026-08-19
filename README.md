# K-Median Algorithms

This project implements and compares different algorithms for the **k-median problem**:
given a set of points, choose *k* of them as "facilities" so that every other point is as
close as possible to its nearest facility. It's the code behind the paper *"A Resonance
Neural Network for the K-Median Problem"* (Bloch-Hansen, Rossiter, Solis-Oba).

## How to run it

You need **Python 3.10 or newer**. Set up once:

```bash
conda create -n kmedian python=3.11 -y
conda activate kmedian
conda install pytorch cpuonly -c pytorch -y
pip install "numpy<2" networkx pandas pulp
```

Then run:

```bash
python main.py
```

It asks a few questions, in order:
1. Problem type (1-4) - pick **1** for k-median
2. Algorithm (1-18) - see the table below
3. Number of runs - press **Enter** for 1
4. Dataset (1-11)
5. GPU - pick **CPU**

When it finishes, it saves the results to a `results_*.csv` file.

## Which algorithm is which

The menu number, the code's class name, and the paper's name don't match each other,
so here's the key (k-median, problem type 1):

| Menu # | File | Paper name |
|---|---|---|
| **8** | ARN.py | **ARN - the fast method** |
| **12** | MARN.py | **MARN - the accurate method** |
| 1 | CodySolver.py | Cody's Hopfield network |
| 2 | HaralampievAlgorithmSolver.py | Haralampiev's network |
| 3 | LocalSearchSolver.py | Local search |
| 4 | ZhuAlgorithmSolver.py | Pan & Zhu |
| 5 | AryaMultiSolver.py | Arya local search |
| 6 / 7 | CohenAddad(Multi)Solver.py | Cohen-Addad |
| 13 | InterchangeAlgorithmSolver.py | Fast interchange |
| 14 | DominguezAlgorithmSolver.py | Dominguez network |

The two main algorithms from the paper are **ARN (option 8)** and **MARN (option 12)**.

## Datasets

Pick a dataset by number:

1 Random-Small . 2 Random-Large . 3 USCA312 . 4 P-Median (OR-Library) . 5 TSPLib .
6 Special . 7 Barabasi . 8 Erdos-Renyi . 9 Path-Grid . 10 Trees . 11 SBM

The number of facilities (*k*) is set **inside each dataset file**. To test different
values of *k*, use a dataset that has many instances (like P-Median), or create new
ones with the tools in `graph_generators/`.

## Folder guide

- `main.py` - start here; picks the problem, algorithm, and dataset
- `ExperimentManager.py` - runs the algorithm and saves the results
- `solvers_alg/` - the algorithms (the k-median ones are in `KMP/Main/`)
- `problems/` - the problem definitions
- `initializers/` - different ways to choose the starting facilities
- `graph_generators/` - tools to create test graphs
- `reader/` - loads the datasets
- `datasets/` - the test data
- `utils/` - helper code

## Notes

- NumPy must be version 1 (`numpy<2`) for this PyTorch build.
- Results are currently saved to the main folder; they could be moved into a
  `results/` folder later to stay organized.
