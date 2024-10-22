#!/bin/bash
#SBATCH --job-name=time_benchmarking_no_tcs
#SBATCH --output=time_benchmarking_no_tcs.out
#SBATCH --error=time_benchmarking_no_tcs.out
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="rtx_6000:1"


export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /coc/flash9/bdevnani3/lmms-eval
source .env/bin/activate
cd /nethome/bdevnani3/flash/lmms-eval/LLaVA-NeXT/llava/eval
export PYTHONIOENCODING=utf-8

python time_benchmarking.py