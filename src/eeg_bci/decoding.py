from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import default_speller_matrix


@dataclass(slots=True)
class DecodedTrial:
    trial_id: int
    letter: str
    row: int
    col: int
    row_score: float
    col_score: float


def decode_trial_letters(
    trial_ids: np.ndarray,
    stimulations: np.ndarray,
    probabilities: np.ndarray,
    speller_matrix: np.ndarray | None = None,
) -> list[DecodedTrial]:
    matrix = default_speller_matrix() if speller_matrix is None else speller_matrix
    trial_ids = np.asarray(trial_ids, dtype=int)
    stimulations = np.asarray(stimulations, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    decoded: list[DecodedTrial] = []
    for trial_id in np.unique(trial_ids):
        trial_mask = trial_ids == trial_id
        if not np.any(trial_mask):
            continue
        row_scores = defaultdict(list)
        col_scores = defaultdict(list)
        for stim, probability in zip(stimulations[trial_mask], probabilities[trial_mask]):
            if 1 <= stim <= 6:
                row_scores[int(stim) - 1].append(float(probability))
            elif 7 <= stim <= 12:
                col_scores[int(stim) - 7].append(float(probability))

        if not row_scores or not col_scores:
            continue

        row_index, row_score = _best_score(row_scores)
        col_index, col_score = _best_score(col_scores)
        letter = str(matrix[row_index, col_index])
        decoded.append(
            DecodedTrial(
                trial_id=int(trial_id),
                letter=letter,
                row=row_index,
                col=col_index,
                row_score=row_score,
                col_score=col_score,
            )
        )
    return decoded


def group_letters_into_words(letters: list[str], letters_per_word: int = 5) -> list[str]:
    words = []
    for index in range(0, len(letters), letters_per_word):
        chunk = letters[index : index + letters_per_word]
        if len(chunk) == letters_per_word:
            words.append("".join(chunk))
    return words


def decoded_trials_to_words(decoded: list[DecodedTrial], letters_per_word: int = 5) -> list[str]:
    letters = [trial.letter for trial in decoded]
    return group_letters_into_words(letters, letters_per_word=letters_per_word)


def _best_score(score_map: dict[int, list[float]]) -> tuple[int, float]:
    best_index = -1
    best_score = float("-inf")
    for index, values in score_map.items():
        score = float(np.mean(values))
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score
