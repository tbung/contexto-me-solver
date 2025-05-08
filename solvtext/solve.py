from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from solvtext.data import load_words


@dataclass(order=True)
class Guess:
    rank: int
    word: str = field(compare=False)
    word_idx: int = field(compare=False)


class Solver:
    def __init__(
        self,
        data_dir: Path,
        debug_target: str | None = None,
        distance_offset: float = 0,
    ):
        self.words: list[str]
        self.vectors: npt.NDArray[np.float32]

        self.words, self.vectors = load_words(data_dir)

        self.candidate_mask: npt.NDArray[np.integer] = np.ones(
            (len(self.words),), dtype=np.integer
        )
        self.already_guessed_mask: npt.NDArray[np.bool_] = np.zeros(
            (len(self.words),), dtype=np.bool_
        )

        self.guesses: list[Guess] = []

        self.debug_target: str | None = debug_target

        self.distance_offset: float = distance_offset

    @property
    def num_guesses(self) -> int:
        return len(self.guesses)

    def _vector_distances(
        self, vector: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return 1 - (
            (vector @ self.vectors.T)
            / np.linalg.norm(vector)
            / np.linalg.norm(self.vectors, axis=1)
        )

    @lru_cache
    def distances(self, word_idx: int) -> npt.NDArray[np.float32]:
        return self._vector_distances(self.vectors[word_idx])

    def update_candidates(self) -> None:
        score = np.zeros_like(self.candidate_mask)
        for guess, other in zip(self.guesses, self.guesses[1:]):
            score += (
                self.distances(guess.word_idx)
                <= self.distances(other.word_idx) + self.distance_offset
            )

            if self.debug_target and score[
                self.words.index(self.debug_target)
            ] < np.amax(score):
                print(
                    f"{guess.word} ({guess.rank}) - {self.debug_target}: {self.distances(guess.word_idx)[self.words.index(self.debug_target)]}"
                )
                print(
                    f"{other.word} ({other.rank}) - {self.debug_target}: {self.distances(other.word_idx)[self.words.index(self.debug_target)]}"
                )

        score[self.already_guessed_mask] = 0

        self.candidate_mask = np.logical_and(
            (1 - self.already_guessed_mask), score == np.amax(score)
        ).astype(int)

    def add_guess(self, word_idx: int, rank: int):
        guess = Guess(rank, self.words[word_idx], word_idx)

        self.guesses.append(guess)
        self.guesses.sort()

        self.update_candidates()

    def make_guess(self) -> int:
        if len(self.guesses) < 2:
            candidate_idx = np.where(self.candidate_mask)[0]
            vector = self.vectors.mean(axis=0)
            word_idx = candidate_idx[
                np.random.choice(
                    self._vector_distances(vector)[candidate_idx].argsort()[:10]
                )
            ]

        elif self.guesses[0].rank <= 300:
            candidate_idx = np.where(self.candidate_mask)[0]
            vector = self.vectors[
                [guess.word_idx for guess in self.guesses if guess.rank <= 300]
            ].mean(axis=0)
            word_idx = candidate_idx[
                self._vector_distances(vector)[candidate_idx].argmin()
            ]

        elif self.guesses[0].rank <= 1500:
            candidate_idx = np.where(self.candidate_mask)[0]
            word_idx = candidate_idx[
                np.random.choice(
                    self.distances(self.guesses[0].word_idx)[candidate_idx].argsort()[
                        : max(self.candidate_mask.sum() // 2, 1)
                    ]
                )
            ]
        elif self.guesses[0].rank <= 3000:
            candidate_idx = np.where(self.candidate_mask)[0]
            word_idx = candidate_idx[
                np.random.choice(
                    self.distances(self.guesses[0].word_idx)[candidate_idx].argsort()[
                        self.candidate_mask.sum() // 2 :
                    ]
                )
            ]
        else:
            candidate_idx = np.where(self.candidate_mask)[0]
            word_idx = candidate_idx[
                np.random.choice(
                    self.distances(self.guesses[0].word_idx)[candidate_idx].argsort()[
                        self.candidate_mask.sum() * 3 // 4 :
                    ]
                )
            ]

        self.already_guessed_mask[word_idx] = 1

        return cast(int, word_idx)

    @property
    def best_rank(self) -> int:
        if len(self.guesses) == 0:
            return np.iinfo(np.int32).max

        return self.guesses[0].rank

    def has_candidates(self) -> bool:
        return self.candidate_mask.sum() > 0

    def reset(self) -> None:
        self.candidate_mask = np.ones((len(self.words),), dtype=np.integer)
        self.already_guessed_mask = np.zeros((len(self.words),), dtype=np.bool_)

        self.guesses = []
