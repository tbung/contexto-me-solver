from pydantic import BaseModel
import requests


class ApiResponse(BaseModel):
    distance: int
    word: str
    lemma: str


def get_rank_from_api(game: int, word: str) -> ApiResponse | None:
    r = requests.get(f"https://api.contexto.me/machado/en/game/{game}/{word}")

    # Status code 404 indicates the word is unknown
    if r.status_code == 404:
        return None

    response = ApiResponse.model_validate(r.json())

    return response
