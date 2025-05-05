from pathlib import Path
import numpy as np
import numpy.typing as npt


def load_words() -> tuple[list[str], npt.NDArray[np.float32]]:
    data_path = Path("/home/tillb/Projects/context-me-solver/data")

    if (data_path / "filtered_words.txt").is_file() and (data_path / "vectors.npz").is_file():
        with (data_path / "filtered_words.txt").open() as f:
            filtered_words = [word.strip() for word in f.readlines()]

        vectors = np.load(data_path / "vectors.npz")

        return filtered_words, vectors["arr_0"]

    glove: dict[str, npt.ArrayLike] = {}

    with (data_path / "glove.840B.300d.txt").open() as f:
        for line in f:
            entries = line.split(" ")
            glove[entries[0]] = np.array(entries[1:], dtype=np.float32)

    with (data_path / "20k.txt").open() as f:
        words = [word.strip() for word in f.readlines()]

    filtered_words = [word for word in words if word in glove]
    vectors = np.vstack([glove[word] for word in filtered_words])

    with (data_path / "filtered_words.txt").open() as f:
        f.write("\n".join(filtered_words))

    np.savez_compressed(data_path / "vectors.npz", vectors)

    return filtered_words, vectors
