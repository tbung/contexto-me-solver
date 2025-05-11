from pydantic import BaseModel, ValidationError
import requests


class ApiResponse(BaseModel):
    """
    Expected respons from the contexto.me API.

    Attributes
    ----------
    distance : int
        Zero-based rank of the last guess.
    word : str
        Guessed word.
    lemma: str
        Lemma of the guessed word.
    """

    distance: int
    word: str
    lemma: str


def get_rank_from_api(game: int, word: str) -> ApiResponse | None:
    """
    Get the rank and possible lemma of a guess from the contexto.me API.

    Parameters
    ----------
    game : int
        Game for which to make the guess.
    word : str
        Guessed word.

    Returns
    -------
    ApiResponse or None
        Returns the response data or None if the word is not known.

    Raises
    ------
    JSONDecodeError or ValidationError
        If the API responds with a status_code that is not 200 or 404, or if the response data is not expected,
        an exception is raised.
    """

    r = requests.get(f"https://api.contexto.me/machado/en/game/{game}/{word}")

    # Status code 404 indicates the word is unknown
    if r.status_code == 404:
        return None

    try:
        response = ApiResponse.model_validate(r.json())
    except (requests.JSONDecodeError, ValidationError):
        print(f"Response(code={r.status_code}, content={r.text})")
        raise

    return response
