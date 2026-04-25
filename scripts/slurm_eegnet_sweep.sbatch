#!/bin/bash
#SBATCH -J eegnet-sweep
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
echo "Starting EEGNet sweep at $(date)"

conda run -n prml-p300 env PYTHONPATH=src python scripts/tune_eegnet_sweep.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
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
