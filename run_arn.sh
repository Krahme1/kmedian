#!/bin/bash
#SBATCH --account=def-rsolisob_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=arn-%j.log

module load python/3.11
source ~/kmedian-env/bin/activate
cd ~/scratch/kmedian
printf "1\n8\n\n4\nn\n" | python main.py
