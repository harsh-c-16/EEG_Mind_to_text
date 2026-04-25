#!/bin/bash
#SBATCH -J eegnet-sweep2
#SBATCH -p btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o ${OUTPUT_DIR}/output.txt
#SBATCH -e ${OUTPUT_DIR}/error.txt

set -euo pipefail

: "${DATA_DIR:?DATA_DIR environment variable is not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR environment variable is not set}"
: "${PROJECT_DIR:=${HOME}/PRML}"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
echo "Starting EEGNet stage2 sweep at $(date)"

conda run -n prml-p300 env PYTHONPATH=src python scripts/tune_eegnet_sweep.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
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
