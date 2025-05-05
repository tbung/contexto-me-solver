from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field
from typing import Literal, overload

import rich
from solvtext.data import load_words
from functools import lru_cache
from rich.prompt import IntPrompt, Prompt, PromptBase, InvalidResponse
import numpy as np
import numpy.typing as npt


class GuessPrompt(PromptBase[int | str]):
    response_type = int | str
    choices: list[str] = ["n", "c"]
    illegal_choice_message = (
        "[prompt.invalid]Please enter a valid integer number or one of ['n', 'c']"
    )

    def process_response(self, value: str) -> int | str:
        value = value.strip()
        try:
            return_value: int = int(value)
            return return_value
        except ValueError:
            pass

        if not self.check_choice(value):
            raise InvalidResponse(self.illegal_choice_message)

        return value


@dataclass(order=True)
class Guess:
    rank: int
    word: str = field(compare=False)
    word_idx: int = field(compare=False)


class Solver:
    def __init__(self, debug_target: str | None = None, distance_offset: float = 0):
        self.words: list[str]
        self.vectors: npt.NDArray[np.float32]

        self.words, self.vectors = load_words()

        self.candidate_mask: npt.NDArray[np.integer] = np.ones(
            (len(self.words),), dtype=np.integer
        )

        self.guesses: list[Guess] = []

        self.debug_target: str | None = debug_target

        self.distance_offset: float = distance_offset

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
            self.candidate_mask &= (
                self.distances(guess.word_idx)
                <= self.distances(other.word_idx) + self.distance_offset
            )

            if (
                self.debug_target
                and self.candidate_mask[self.words.index(self.debug_target)] == 0
            ):
                print(
                    f"{guess.word} ({guess.rank}) - {self.debug_target}: {self.distances(guess.word_idx)[self.words.index(self.debug_target)]}"
                )
                print(
                    f"{other.word} ({other.rank}) - {self.debug_target}: {self.distances(other.word_idx)[self.words.index(self.debug_target)]}"
                )

    @overload
    def make_guess(self, interactive: Literal[False]) -> int: ...

    @overload
    def make_guess(self, interactive: Literal[True]) -> None: ...

    @overload
    def make_guess(self) -> None: ...

    def make_guess(self, interactive: bool = True) -> int | None:
        if interactive:
            rich.print(
                f"Remaining search space: {self.candidate_mask.sum()}/{self.candidate_mask.shape[0]}"
            )

        word_idx = np.random.choice(np.where(self.candidate_mask)[0])
        word = self.words[word_idx]

        self.candidate_mask[word_idx] = 0

        if not interactive:
            return word_idx

        rich.print(f"Guess: [bold]{word}[/bold]")
        rank = GuessPrompt.ask(
            "Enter rank or 'n' if unknown or 'c' if contexto.me picks a different word",
            show_choices=False,
        )
        if rank == "n":
            return

        if rank == "c":
            new_word = Prompt.ask("Enter new word", default=word)
            if new_word in self.words:
                word_idx = self.words.index(new_word)
                word = new_word
                self.candidate_mask[word_idx] = 0
            else:
                print(f"Unknown word, using '{word}'")

            rank = IntPrompt.ask(f"Enter rank for {word}")

        assert type(rank) is int

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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate a number of games (non-interactive)",
    )
    parser.add_argument(
        "-n",
        "--num-simulate",
        action="store",
        nargs="?",
        default=100,
        type=int,
        help="Number of games to simulate",
    )
    parser.add_argument(
        "--debug-target", action="store", nargs="?", default=None, type=str
    )
    parser.add_argument(
        "-d",
        "--distance-offset",
        action="store",
        nargs="?",
        default=0.05,
        type=float,
        help="Minimum distance to decision surface. Higher values means more guesses, but a lower chance of completely missing the target.",
    )

    args = parser.parse_args()

    if args.simulate:
        num_guesses = []
        for _ in range(args.num_simulate):
            num_guesses.append(simulate())

        print(np.mean(num_guesses))
        return

    solver = Solver(
        debug_target=args.debug_target, distance_offset=args.distance_offset
    )
    num_guesses = 0
    try:
        while True:
            solver.make_guess()
            num_guesses += 1
    except (KeyboardInterrupt, ValueError):
        rich.print(
            f"\nNumber of guesses (valid/all): {len(solver.guesses)}/{num_guesses}"
        )
