import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import rich
from rich.prompt import IntPrompt, InvalidResponse, Prompt, PromptBase

from solvtext.api import get_rank_from_api
from solvtext.solve import Solver


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


def run_interactive(data_dir: Path, distance_offset: float, debug_target: str | None):
    num_guesses_with_invalid = 0

    solver = Solver(
        data_dir, debug_target=debug_target, distance_offset=distance_offset
    )

    while solver.has_candidates() and solver.best_rank > 1:
        num_guesses_with_invalid += 1

        rich.print(
            f"Remaining search space: {solver.candidate_mask.sum()}/{solver.candidate_mask.shape[0]}"
        )

        word_idx = solver.make_guess()
        word = solver.words[word_idx]

        rich.print(f"Guess: [bold]{word}[/bold]")
        rank = GuessPrompt.ask(
            "Enter rank or 'n' if unknown or 'c' if contexto.me picks a different word",
            show_choices=False,
        )

        if rank == "n":
            solver.update_candidates()
            continue

        if rank == "c":
            new_word = Prompt.ask("Enter new word", default=word)
            if new_word in solver.words:
                word_idx = solver.words.index(new_word)
                word = new_word
                solver.already_guessed_mask[word_idx] = 1
            else:
                print(f"Unknown word, using '{word}'")

            rank = IntPrompt.ask(f"Enter rank for {word}")

        assert type(rank) is int

        solver.add_guess(word_idx, rank)

    print(
        f"Number of guesses (valid/all): {solver.num_guesses}/{num_guesses_with_invalid}"
    )

    if solver.best_rank > 1:
        print("No more candidates to try")


def run_automatic(
    data_dir: Path, game: int, distance_offset: float, debug_target: str | None
):
    num_guesses_with_invalid = 0

    solver = Solver(
        data_dir, distance_offset=distance_offset, debug_target=debug_target
    )

    while solver.has_candidates() and solver.best_rank > 0:
        num_guesses_with_invalid += 1

        word_idx: int = solver.make_guess()

        print(f"Guess: {solver.words[word_idx]}")
        time.sleep(0.2)

        result = get_rank_from_api(game, solver.words[word_idx])

        if result is None:
            solver.update_candidates()
            continue

        if result.word != result.lemma and result.lemma in solver.words:
            word_idx = solver.words.index(result.lemma)
            print(f"New Word: {result.lemma}")

        print(f"Rank: {result.distance}")

        solver.add_guess(word_idx, result.distance)

    print(
        f"Number of guesses (valid/all): {solver.num_guesses}/{num_guesses_with_invalid}"
    )

    if solver.best_rank > 0:
        print("No more candidates to try")


def _run_simulation(solver: Solver) -> int:
    target_idx: int = np.random.choice(np.arange(len(solver.words)))

    distances = solver.distances(target_idx).copy()
    rank = distances.argsort().argsort()

    print(solver.words[target_idx])

    while solver.has_candidates() and solver.best_rank > 0:
        guess_idx = solver.make_guess()
        solver.add_guess(guess_idx, rank[guess_idx])

    return solver.num_guesses


def run_simulations(data_dir: Path, n_runs: int):
    solver = Solver(data_dir)

    num_guesses: list[int] = []

    for _ in range(n_runs):
        num_guesses.append(_run_simulation(solver))
        solver.reset()

    print(f"Average number of guesses: {np.mean(num_guesses)}")


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--data-dir",
        action="store",
        type=Path,
        default=Path("./data"),
        help="Path to directory where word list and embeddings will be stored",
    )
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
        default=0,
        type=float,
        help="Minimum distance to decision surface. Higher values means more guesses, but a lower chance of completely missing the target.",
    )
    parser.add_argument(
        "--automate-game",
        action="store",
        nargs="?",
        default=None,
        type=int,
        help="Automatically run game GAME via the API",
    )

    args = parser.parse_args()

    if args.simulate:
        run_simulations(args.data_dir, args.num_simulate)
    elif args.automate_game is not None:
        run_automatic(
            args.data_dir, args.automate_game, args.distance_offset, args.debug_target
        )
    else:
        run_interactive(args.data_dir, args.distance_offset, args.debug_target)
