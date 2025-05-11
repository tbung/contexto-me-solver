from pathlib import Path
from zipfile import ZipFile

import nltk
import numpy as np
import numpy.typing as npt
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from rich.progress import track


def _download(file_path: Path, url: str):
    if file_path.is_file():
        return

    try:
        with file_path.open("wb") as f:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in track(
                    r.iter_content(chunk_size=8192),
                    total=int(r.headers.get("content-length", 0)) / 8192,
                ):
                    f.write(chunk)
    except:
        file_path.unlink()
        raise


def _pt2wn(pos: str) -> str:
    pos = pos.lower()

    if pos.startswith("jj"):
        tag = ADJ
    elif pos == "md":
        tag = VERB
    elif pos.startswith("rb"):
        tag = ADV
    elif pos.startswith("vb"):
        tag = VERB
    elif pos == "wrb":
        tag = ADV
    else:
        tag = NOUN

    return tag


def load_words(data_dir: Path) -> tuple[list[str], npt.NDArray[np.float32]]:
    data_dir.mkdir(exist_ok=True)

    embedding_file = "glove.840B.300d"

    if (data_dir / "words.txt").is_file() and (data_dir / "vectors.npz").is_file():
        with (data_dir / "words.txt").open() as f:
            filtered_words = [word.strip() for word in f.readlines()]

        vectors = np.load(data_dir / "vectors.npz")

        return filtered_words, vectors["arr_0"]

    glove: dict[str, npt.NDArray[np.float32]] = {}

    print("Loading glove embeddings...")
    if (data_dir / f"{embedding_file}.npz").is_file():
        glove = np.load(data_dir / f"{embedding_file}.npz", allow_pickle=True)[
            "arr_0"
        ].item()
    else:
        _download(
            data_dir / f"{embedding_file}.zip",
            "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        )
        with ZipFile(data_dir / f"{embedding_file}.zip") as f:
            f.extractall(data_dir)
        with (data_dir / f"{embedding_file}.txt").open() as f:
            for line in f:
                entries = line.split(" ")
                glove[entries[0]] = np.array(entries[1:], dtype=np.float32)

        np.savez_compressed(
            data_dir / f"{embedding_file}.npz", glove, allow_pickle=True
        )

    print("Encoding word list...")

    _download(
        data_dir / "20k.txt",
        "https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt",
    )
    with (data_dir / "20k.txt").open() as f:
        words = [word.strip() for word in f.readlines()]

    # not necessarily correct, but contexto.me seems to try to lemmatize all inputs
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger_eng")
    wnl = WordNetLemmatizer()
    words_tagged: list[tuple[str, str]] = pos_tag(words)
    words = [wnl.lemmatize(word, pos=_pt2wn(tag)) for word, tag in words_tagged]

    # filtered_words = [word for word in words if word in glove]
    filtered_words: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word not in seen and word in glove:
            filtered_words.append(word)
            seen.add(word)
    vectors = np.vstack([glove[word] for word in filtered_words])

    with (data_dir / "words.txt").open("wt") as f:
        f.write("\n".join(filtered_words))

    np.savez_compressed(data_dir / "vectors.npz", vectors)

    return filtered_words, vectors
