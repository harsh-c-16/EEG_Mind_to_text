#!/bin/bash
#SBATCH -J eegnet-sweep
#SBATCH -p btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o /csehome/b24cm1027/PRML/outputs/eegnet_sweep/output.txt
#SBATCH -e /csehome/b24cm1027/PRML/outputs/eegnet_sweep/error.txt

set -euo pipefail

mkdir -p /csehome/b24cm1027/PRML/outputs/eegnet_sweep

cd /csehome/b24cm1027/PRML
echo "Starting EEGNet sweep at $(date)"

conda run -n prml-p300 env PYTHONPATH=src python scripts/tune_eegnet_sweep.py \
  --data-dir /scratch/b24cm1027/P300 \
  --output-dir /csehome/b24cm1027/PRML/outputs/eegnet_sweep \
  --max-combos 12 \
  --epochs 20,30 \
  --batch-sizes 32,64 \
  --lrs 0.001,0.0005 \
  --dropouts 0.25,0.4 \
  --f1s 8,16 \
  --ds 2 \
  --f2s 16,32 \
  --kernel-lengths 32,64 \
  --patience 7

echo "Finished EEGNet sweep at $(date)"
