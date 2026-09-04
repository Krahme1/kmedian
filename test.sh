#!/bin/bash
#SBATCH --account=def-rsolisob_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --time=00:02:00
#SBATCH --mem=1G
#SBATCH --output=test-%j.log

echo "Hello from $(hostname)"
module load python/3.11
python -c "print('cluster job works:', 2+2)"
