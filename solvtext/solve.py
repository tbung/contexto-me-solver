from dataclasses import dataclass, field
from typing import Literal, overload
from solvtext.data import load_words
from functools import lru_cache
from rich.prompt import IntPrompt
import numpy as np
import numpy.typing as npt


@dataclass(order=True)
class Guess:
    rank: int
    word: str = field(compare=False)
    word_idx: int = field(compare=False)


class Solver:
    def __init__(self):
        self.words: list[str]
        self.vectors: npt.NDArray[np.float32]

        self.words, self.vectors = load_words()

        self.candidate_mask: npt.NDArray[np.integer] = np.ones(
            (len(self.words),), dtype=np.integer
        )

        self.guesses: list[Guess] = []

    @lru_cache
    def distances(self, word_idx: int) -> npt.NDArray[np.float32]:
        return 1 - (
            (self.vectors[word_idx] @ self.vectors.T)
            / np.linalg.norm(self.vectors[word_idx])
            / np.linalg.norm(self.vectors, axis=1)
        )

    def add_guess(self, word_idx: int, rank: int):
        guess = Guess(rank, self.words[word_idx], word_idx)

        self.guesses.append(guess)
        self.guesses.sort()

        guess = self.guesses[0]

        for other in self.guesses[1:]:
            self.candidate_mask &= self.distances(guess.word_idx) <= self.distances(
                other.word_idx
            )

    @overload
    def make_guess(self, interactive: Literal[False]) -> int: ...

    @overload
    def make_guess(self, interactive: Literal[True]) -> None: ...

    @overload
    def make_guess(self) -> None: ...

    def make_guess(self, interactive: bool = True) -> int | None:
        print(self.candidate_mask.sum())

        word_idx = np.random.choice(np.where(self.candidate_mask)[0])
        word = self.words[word_idx]

        self.candidate_mask[word_idx] = 0

        if not interactive:
            return word_idx

        rank = IntPrompt.ask(f"Guess: {word}\nEnter rank (or -1 if word is unknown)")
        if rank == -1:
            return

        self.add_guess(word_idx, rank)


def simulate():
    solver = Solver()
    target_idx = np.random.choice(np.arange(len(solver.words)))

    distances = solver.distances(target_idx).copy()
    rank = distances.argsort().argsort()

    num_guesses = 0

    print(solver.words[target_idx])

    while True:
        guess_idx = solver.make_guess(interactive=False)
        solver.add_guess(guess_idx, rank[guess_idx])

        num_guesses += 1

        if rank[guess_idx] == 0:
            break

    return num_guesses


def main():
    num_guesses = []
    for _ in range(100):
        num_guesses.append(simulate())

    print(np.mean(num_guesses))

    # solver = Solver()
    # num_guesses = 0
    # try:
    #     while True:
    #         solver.make_guess()
    #         num_guesses += 1
    # except KeyboardInterrupt:
    #     print(f"{num_guesses}")
