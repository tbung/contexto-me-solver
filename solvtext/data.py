from pathlib import Path
import numpy as np
import numpy.typing as npt
import nltk
from nltk.stem import WordNetLemmatizer


def load_words() -> tuple[list[str], npt.NDArray[np.float32]]:
    data_path = Path("/home/tillb/Projects/context-me-solver/data")

    if (data_path / "words.txt").is_file() and (data_path / "vectors.npz").is_file():
        with (data_path / "words.txt").open() as f:
            filtered_words = [word.strip() for word in f.readlines()]

        vectors = np.load(data_path / "vectors.npz")

        return filtered_words, vectors["arr_0"]

    glove: dict[str, npt.NDArray[np.float32]] = {}

    print("Loading glove embeddings...")
    if (data_path / "glove.840B.300d.npz").is_file():
        glove = np.load(data_path / "glove.840B.300d.npz", allow_pickle=True)[
            "arr_0"
        ].item()
    else:
        with (data_path / "glove.840B.300d.txt").open() as f:
            for line in f:
                entries = line.split(" ")
                glove[entries[0]] = np.array(entries[1:], dtype=np.float32)

        np.savez_compressed(data_path / "glove.840B.300d.npz", glove, allow_pickle=True)

    print("Encoding word list...")
    wnl = WordNetLemmatizer()

    with (data_path / "20k.txt").open() as f:
        words = {word.strip() for word in f.readlines()}

    # not necessarily correct from a nlp perspective, but contexto.me seems to treat all infinitives this way,
    # e.g. shipping and ship are assigned the same rank
    words = {
        wnl.lemmatize(word, pos="v" if word.endswith("ing") else "n") for word in words
    }

    filtered_words = [word for word in words if word in glove]
    vectors = np.vstack([glove[word] for word in filtered_words])

    with (data_path / "words.txt").open("wt") as f:
        f.write("\n".join(filtered_words))

    np.savez_compressed(data_path / "vectors.npz", vectors)

    return filtered_words, vectors
