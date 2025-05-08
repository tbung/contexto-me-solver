from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import rich
from rich.prompt import IntPrompt, Prompt, PromptBase, InvalidResponse

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


def run_interactive(distance_offset: float, debug_target: str | None):
    num_guesses_with_invalid = 0

    solver = Solver(debug_target=debug_target, distance_offset=distance_offset)

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


def run_automatic(game: int, distance_offset: float, debug_target: str | None):
    pass


def _run_simulation(solver: Solver) -> int:
    target_idx: int = np.random.choice(np.arange(len(solver.words)))

    distances = solver.distances(target_idx).copy()
    rank = distances.argsort().argsort()

    print(solver.words[target_idx])

    while solver.has_candidates() and solver.best_rank > 0:
        guess_idx = solver.make_guess()
        solver.add_guess(guess_idx, rank[guess_idx])

    return solver.num_guesses


def run_simulations(n_runs: int):
    solver = Solver()

    num_guesses: list[int] = []

    for _ in range(n_runs):
        num_guesses.append(_run_simulation(solver))
        solver.reset()

    print(f"Average number of guesses: {np.mean(num_guesses)}")


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
    parser.add_argument(
        "--automate-game", action="store", nargs="?", default=None, type=int
    )

    args = parser.parse_args()

    if args.simulate:
        run_simulations(args.num_simulate)
    elif args.automate_game is not None:
        run_automatic(args.automate_game, args.distance_offset, args.debug_target)
    else:
        run_interactive(args.distance_offset, args.debug_target)
