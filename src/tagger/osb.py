import xxhash


def orthogonal_sparse_bigrams(text: str, window_size: int = 5) -> list[str]:
    """
    Generate orthogonal sparse bigrams from a string.

    Args:
        text: Input string to process
        window_size: Size of the sliding window (default: 5)

    Returns:
        List of orthogonal sparse bigrams
    """

    words = text.split()
    bigrams = []

    for i in range(len(words)):
        for j in range(1, min(window_size, len(words) - i)):
            bigram = f"{words[i]} {'*' * (j-1)} {words[i+j]}"
            bigrams.append(bigram)

    return bigrams


def bigram_indices(text: str, seed: str = "osb", window_size: int = 5) -> list[int]:
    """
    Generate indices for orthogonal sparse bigrams from a string.

    Args:
        text: Input string to process
        seed_phrase: Seed phrase to use for hashing
        window_size: Size of the sliding window (default: 5)

    Returns:
        List of indices for orthogonal sparse bigrams
    """

    bigrams = orthogonal_sparse_bigrams(text, window_size)
    indices = []
    seed = xxhash.xxh32(seed).intdigest()

    for bigram in bigrams:
        indices.append(xxhash.xxh32(bigram, seed).intdigest())

    return indices


def wrap_indices(indices: list[int], vector_size: int) -> list[int]:
    return [idx % vector_size for idx in indices]
