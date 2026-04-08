#!/bin/bash
#SBATCH -J eegnet-sweep2
#SBATCH -p btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o /csehome/b24cm1027/PRML/outputs/eegnet_sweep_stage2/output.txt
#SBATCH -e /csehome/b24cm1027/PRML/outputs/eegnet_sweep_stage2/error.txt

set -euo pipefail

mkdir -p /csehome/b24cm1027/PRML/outputs/eegnet_sweep_stage2

cd /csehome/b24cm1027/PRML
echo "Starting EEGNet stage2 sweep at $(date)"

conda run -n prml-p300 env PYTHONPATH=src python scripts/tune_eegnet_sweep.py \
  --data-dir /scratch/b24cm1027/P300 \
  --output-dir /csehome/b24cm1027/PRML/outputs/eegnet_sweep_stage2 \
  --max-combos 20 \
  --epochs 25,35 \
  --batch-sizes 32 \
  --lrs 0.001,0.0007,0.0005 \
  --dropouts 0.2,0.25,0.3 \
  --f1s 16,24 \
  --ds 2 \
  --f2s 16,24,32 \
  --kernel-lengths 24,32,40 \
  --patience 8

echo "Finished EEGNet stage2 sweep at $(date)"
