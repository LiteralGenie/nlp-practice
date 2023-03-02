import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

from torch import nn
from torch.utils.data import DataLoader2
from torchtext import datasets
from tqdm import tqdm

from config import paths

Lines = list[str]
VocabTally = dict[str, int]


class Nnlm(nn.Module):
    params_feature = 1000
    params_hidden = 500

    def __init__(self, vocab_size: int, lookback_count: int):
        super().__init__()

        self.vocab_size = vocab_size

        # number of previous words to peek at +1
        self.n = lookback_count + 1

        self.C = nn.Embedding(self.vocab_size, self.params_feature)

        # hidden layer weights
        self.H = nn.Linear((self.n - 1) * self.params_feature, self.params_hidden)

        # output layer weights
        self.U = nn.Linear(self.params_feature)

    def forward(self, xb):
        # convert words to embeddings
        out = self.C(xb)

        # concat embeddings
        out = out.view(-1, (self.n - 1) * self.params_feature)

        # feed to hidden layer
        out = self.H(out)
        out = nn.Tanh(out)

        # feed to output layer
        out = self.U(out)
        out = nn.Softmax(out)

        return out


def get_clean_data(fp: Path, dataloader: DataLoader2) -> list:
    """Remove unwanted characters"""

    def main():
        if fp.exists():
            with open(fp) as file:
                return json.load(file)

        pbar = tqdm(dataloader)
        lines = []
        for l in pbar:
            cleaned = clean(" ".join(l))
            if cleaned:
                lines.append(cleaned)

        with open(fp, "w+") as file:
            json.dump(lines, file, indent=2)

        return lines

    def clean(text: str) -> str:
        result = text.strip()

        # Convert line-breaks to spaces
        result = result.replace("\n", " ")

        # Replace periods with <END>
        result = re.sub(r"\. ", " zENDz ", result)

        # Remove multi-spaces
        result = re.sub(r" +", " ", result)

        # Remove multi-quotes
        result = re.sub(r"'+", "'", result)

        # Remove other characters
        result = re.sub("[^a-z0-9' ]", "", result, flags=re.IGNORECASE)

        return result

    return main()


def get_vocab(fp: Path, text: str) -> VocabTally:
    """Get unique words and their frequency count"""

    if fp.exists():
        with open(fp) as file:
            return json.load(file)

    vocab = dict()
    words = text.split()

    for w in tqdm(words):
        w = w.lower()
        vocab.setdefault(w, 0)
        vocab[w] += 1

    vocab_sorted = dict(sorted(vocab.items(), key=lambda it: it[1], reverse=True))

    with open(fp, "w+") as file:
        json.dump(vocab_sorted, file, indent=2)

    return vocab_sorted


def constrain(
    lines: Lines, vocab: VocabTally, freq_thresh: int, length_thresh: int
) -> Tuple[Lines, VocabTally]:
    """
    Reduce dataset size by...
        - replacing words that rarely appear with <UNK>
        - omitting short sentences
    """

    UNK = "<UNK>"

    lines = lines.copy()
    vocab = vocab.copy()
    vocab[UNK] = 0

    whitelist = set(k for k, v in vocab.items() if v >= freq_thresh)

    for i, l in enumerate(tqdm(lines)):
        result = l
        words = l.split()

        if len(words) < length_thresh:
            continue

        for w in words:
            if w not in whitelist:
                result = result.replace(w, "<UNK>")
                vocab[UNK] += 1

        lines[i] = result

    return (lines, vocab)


if __name__ == "__main__":
    print("Loading dataset")
    pipe = datasets.EnWik9(str(paths.DATA_DIR))
    loader = DataLoader2(pipe, shuffle=True, drop_last=True)

    print("Loading vocab")
    lines = get_clean_data(paths.DATASET_DIR / "enwiki_data.jsonc", loader)
    vocab = get_vocab(paths.DATASET_DIR / "enwiki_vocab.jsonc", " ".join(lines))

    print("Constraining data")
    sequence_length = 5
    lines, vocab = constrain(lines, vocab, 100, sequence_length)

    print("Initializing dataset")
    vocab_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_vocab = {i: w for i, w in enumerate(vocab)}

    print(
        f"Training with {len(lines):,} sentences and {len(vocab):,} words and {sequence_length=}"
    )
    model = Nnlm(len(vocab), sequence_length)
