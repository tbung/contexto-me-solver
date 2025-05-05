from dataclasses import dataclass, field
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
            / np.linalg.norm(self.vectors[0])
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

    def make_guess(self) -> int | None:
        print(self.candidate_mask.sum())

        word_idx = np.random.choice(np.where(self.candidate_mask)[0])
        word = self.words[word_idx]

        self.candidate_mask[word_idx] = 0

        rank = IntPrompt.ask(f"Guess: {word}\nEnter rank (or -1 if word is unknown)")
        if rank == -1:
            return

        self.add_guess(word_idx, rank)


def main():
    solver = Solver()
    num_guesses = 0
    try:
        while True:
            solver.make_guess()
            num_guesses += 1
    except KeyboardInterrupt:
        print(f"{num_guesses}")
