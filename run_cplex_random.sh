#!/bin/bash
#SBATCH --account=def-rsolisob_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=cplex-random-%j.log

module load python/3.11
source ~/kmedian-env/bin/activate
cd ~/scratch/kmedian
python run_cplex_random.py
