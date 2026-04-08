# EEG Based BCI for Mind-to-Text Communication

This project provides an end-to-end classical machine learning baseline for the P300 speller dataset.

## What it does

- Loads MATLAB, NumPy, or pickle-based EEG exports
- Preprocesses EEG with notch and band-pass filtering
- Extracts flash-locked epochs from `data.flash` and `data.trial`
- Builds PSD and CSP features
- Trains a flash-level classifier for hit/no-hit detection
- Decodes letters and words from trial-level flash predictions

## Repository layout

- `src/eeg_bci/` contains the reusable package
- `eeg-bci` is the command-line entry point

## Installation

```bash
pip install -e .
```

## Training

```bash
eeg-bci train --data /scratch/b24cm1027/P300/subject1.mat --model-dir outputs/model
```

## Evaluation

```bash
eeg-bci evaluate --data /scratch/b24cm1027/P300/subject1.mat --model-dir outputs/model
```

## Prediction

```bash
eeg-bci predict --data /scratch/b24cm1027/P300/subject1.mat --model-dir outputs/model
```

## Notes

- The dataset description says the first 3 words are used for training and the last 4 for testing.
- The code follows that convention by default using the first 15 trials for training and the remaining 20 for testing.
- If your dataset file stores fields differently, update the loader in `src/eeg_bci/dataset.py`.
