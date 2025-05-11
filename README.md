# Contexto.me Solver

An automatic solver for the [contexto.me](https://contexto.me/en/) word search game.

## Installation

This project requires Python version 3.11 or newer.

```bash
pip install git+https://github.com/tbung/contexto-me-solver.git
```

## Usage

To run a game, simply do

```bash
solvtext --data-dir ./data
```

where `./data` is the directory where the word list and word embeddings will be downloaded to.
The program will then present you with a guess and ask you for the rank. If contexto.me does not know the word, type `n` and a new guess will be made. If contexto.me lemmatizes the word to a different word, you can type `c` and then input the new word and rank for better accuracy.

### Advanced Usage

`solvtext` additionally accepts the following flags:

- `--simulate`: Simulate a number of games (non-interactive)
- `-n NUM`, `--num-simulate NUM`: Number of games to simulate (default: 100).
- `-d OFFSET`, `--distance-offset OFFSET`: Minimum distance to decision surface. Higher values means more guesses, but a lower chance of completely missing the target (default: 0.2).
- `--automate-game GAME`: Automatically run game GAME via the contexto.me API.
- `--max-guesses NUM`: Maximum number of guesses when automatically running a game via the API (default: None). 
- `--automate-random-games NUM_GAMES`: Automatically run NUM_GAMES random games via the API.

## How It Works

This solver generates guesses from a word list (also called search space in the following). It starts out with two random guesses and will then reduce the search space based on the rank of these guesses. Since every pair of ranked words implies the rule, that the target word needs to be closer to the (numerically) lower ranked word than the higher ranked word, we can cut up the search space with a decision surface for every ranked pair. The next guess can then be drawn from the part of the search space that is still valid (fulfills all rules).

Since our word list and embedding vectors are not the same as the ones used by contexto.me, there is a chance that the target word does not fulfill at least one of these rules. To mitigate this, we soften up the decision in two ways: first, instead of considering the decision boundary exactly where words become closer to the lower ranked word, we offset the decision boundary by a fixed amount towards the higher ranked word (controlled via the `--distance-offset` argument). Second, instead of considering every word that does not fulfill even one rule invalid, we count how many rules each word fulfills and consider all words with the highest amount of rules fulfilled as valid candidates for the next guess.

### Implementation Details

- Contexto.me credits [Stanford's NLP Group](https://nlp.stanford.edu/projects/glove/) for the word embeddings, so we use their Common Crawl embeddings.
- We use the 20k most common english words list compiled by [Josh Kaufman](https://github.com/first20hours/google-10000-english) as our search space.
- Contexto.me lemmatizes all guesses, so our word list is also lemmatized, using the [NLTK](https://www.nltk.org/) WordNet Lemmatizer.
- We only consider consecutive ranked word pairs for our decision instead of computing every pairwise decision surface.
- Instead of drawing uniformly from all valid candidates, the initial two guesses are made from the 10 words closest to the mean of all words. All other guesses are made according to the following heuristics:
    - If the current best guess is ranked higher than 3000, pick the next guess from the 25% most distant candidates to the current best guess.
    - If the current best guess is ranked between 1500 and 3000, pick the next guess from the 50% most distant candidates to the current best guess.
    - If the current best guess is ranked between 300 and 1500, pick the next guess from the 50% least distant candidates to the current best guess.
    - If the current best guess is ranked lower than or equal to 300, pick the closest candidate to the mean of the closest 5 guesses ranked lower than or equal to 300.

### Performance

The median number of guesses over 100 games is 38.5.

This was done automatically and every game with over 100 guesses was considered invalid, as in practice in an interactive setting the solver should be restarted way before that. If all invalid games (including some that are invalid because the contexto.me API did not respond) are counted with 100 guesses, the median goes up to 41.

## Things to Try

The following things could be implemented to improve the performance of the solver:

- Do a proper ablation study over the heuristics, different word lists and different word embeddings mentioned in [Implementation Details](README.md#implementation-details)
- Improve the word list continuously with every game, removing redundant and unknown words.
