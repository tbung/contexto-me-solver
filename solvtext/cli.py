import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import rich
from rich.prompt import IntPrompt, InvalidResponse, Prompt, PromptBase

from solvtext.api import get_rank_from_api
from solvtext.solve import Solver


class GuessPrompt(PromptBase[int | str]):
    """
    Prompt for a guess rank. Accepts integers or 'n'/'c' for unknown or contexto.me choice.
    """

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
    """
    Run a solver in interactive mode, allowing the user to input guesses and ranks.

    Parameters
    ----------
    data_dir : Path
        Path to the directory where the word list and embeddings are stored.
    distance_offset : float
        Minimum distance to decision surface. Higher values mean more guesses, but a lower chance of completely missing the target.
    debug_target : str | None
        If provided, the solver will be run with this target word for additional debug output.
    """
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
    data_dir: Path,
    game: int,
    distance_offset: float,
    max_guesses: int | None,
    debug_target: str | None = None,
) -> int:
    """
    Run a solver in automatic mode, retrieving the rank from the contexto.me API.

    Parameters
    ----------
    data_dir : Path
        Path to the directory where the word list and embeddings are stored.
    game : int
        The game number to run.
    distance_offset : float
        Minimum distance to decision surface. Higher values mean more guesses, but a lower chance of completely missing the target.
    max_guesses : int | None
        Maximum number of guesses to make. If None, no limit is set.
    debug_target : str | None
        If provided, the solver will be run with this target word for additional debug output.

    Returns
    -------
    int
        The number of guesses made.
    """

    num_guesses_with_invalid = 0

    solver = Solver(
        data_dir, distance_offset=distance_offset, debug_target=debug_target
    )

    while (
        solver.has_candidates()
        and solver.best_rank > 0
        and (max_guesses is None or solver.num_guesses < max_guesses)
    ):
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

    if solver.best_rank > 0 or (
        max_guesses is not None and solver.num_guesses > max_guesses
    ):
        print("No more candidates to try")
        return -1

    return solver.num_guesses


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
    """
    Run a number of simulations with random targets.

    Parameters
    ----------
    data_dir : Path
        Path to the directory where the word list and embeddings are stored.
    n_runs : int
        Number of simulations to run.
    """

    solver = Solver(data_dir)

    num_guesses: list[int] = []

    for _ in range(n_runs):
        num_guesses.append(_run_simulation(solver))
        solver.reset()

    print(f"Average number of guesses: {np.mean(num_guesses)}")


def main():
    """
    Entry point to the command line interface for the solver.
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="A contexto.me solver.",
    )

    parser.add_argument(
        "--data-dir",
        action="store",
        type=Path,
        required=True,
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
        default=0.2,
        type=float,
        help="Minimum distance to decision surface. Higher values means more guesses, but a lower chance of completely missing the target.",
    )
    parser.add_argument(
        "--automate-game",
        action="store",
        nargs="?",
        default=None,
        type=int,
        metavar="GAME",
        help="Automatically run game GAME via the API",
    )
    parser.add_argument(
        "--max-guesses",
        action="store",
        nargs="?",
        default=None,
        type=int,
        metavar="TRIES",
        help="Maximum number of guesses when automatically running a game via the API",
    )
    parser.add_argument(
        "--automate-random-games",
        action="store",
        nargs="?",
        default=None,
        type=int,
        metavar="NUM_GAMES",
        help="Automatically run NUM_GAMES games via the API",
    )

    args = parser.parse_args()

    if args.simulate:
        run_simulations(args.data_dir, args.num_simulate)
    elif args.automate_game is not None:
        run_automatic(
            args.data_dir,
            args.automate_game,
            args.distance_offset,
            args.max_guesses,
            args.debug_target,
        )
    elif args.automate_random_games is not None:
        num_guesses: list[int] = []
        args.data_dir.mkdir(exist_ok=True)
        with (args.data_dir / "results.csv").open("a+") as f:
            for game in cast(
                npt.NDArray[np.integer],
                np.random.choice(963, args.automate_random_games),
            ):
                try:
                    n = run_automatic(
                        args.data_dir,
                        game,
                        args.distance_offset,
                        args.max_guesses,
                        args.debug_target,
                    )
                except Exception as e:
                    print(e)
                    n = -1
                num_guesses.append(n)

                f.write(f"{game},{n}\n")
                f.flush()

        n_arr = np.array(num_guesses)
        print(f"Median number of guesses: {np.median(n_arr[n_arr != -1])}")
        print(
            f"Median number of guesses with failures: {np.median(np.where(n_arr == -1, 100, n_arr))}"
        )
    else:
        run_interactive(args.data_dir, args.distance_offset, args.debug_target)
