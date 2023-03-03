from tqdm import tqdm


def tally_vocab(text: str, verbose=False) -> dict[str, int]:
    """Get unique words and their frequency count"""

    vocab = dict()
    words = text.split()

    iter = tqdm(words) if verbose else words
    for w in iter:
        vocab.setdefault(w, 0)
        vocab[w] += 1

    vocab_sorted = dict(sorted(vocab.items(), key=lambda it: it[1], reverse=True))

    return vocab_sorted
