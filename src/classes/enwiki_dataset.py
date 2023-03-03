import json
import re
from pathlib import Path
from random import random
from typing import Tuple

import torch
from config import paths
from torch.utils.data import DataLoader2, Dataset
from torchtext import datasets
from tqdm import tqdm
from utils.data_utils import tally_vocab

Lines = list[str]
VocabTally = dict[str, int]


# @DEPRECATED
# its dirty
class EnwikiDataset(Dataset):
    """Cleaned version of http://mattmahoney.net/dc/textdata.html

    Occupies ~3 GB on disk
    """

    lines: list[list[str]]
    vocab: VocabTally

    vocab_to_index: dict[str, int]
    index_to_vocab: dict[int, str]

    def __init__(self, lines: Lines, vocab: VocabTally, sequence_length: int):
        super().__init__()

        self.lines = [l.split() for l in lines]
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.vocab_to_index = {w: i for i, w in enumerate(vocab)}
        self.index_to_vocab = {i: w for i, w in enumerate(vocab)}

    @classmethod
    def load(cls, sequence_length: int, freq_thresh: int = 100) -> "EnwikiDataset":
        lines, vocab = cls._constrain(freq_thresh, sequence_length)

        ds = EnwikiDataset(lines, vocab, sequence_length)
        print(
            f"Loaded Enwiki dataset with with {len(lines):,} sentences and {len(vocab):,} words and {sequence_length=}"
        )

        return ds

    def __getitem__(self, index):
        words = self.lines[index]

        maxStart = len(words) - self.sequence_length
        start = round(random() * maxStart)
        end = start + self.sequence_length
        sample = words[start : end - 1]
        label = words[end - 1]

        sample_idx = torch.LongTensor([self.vocab_to_index[w] for w in sample])
        label_idx = self.vocab_to_index[label]

        return sample_idx, label_idx

    def __len__(self):
        return len(self.lines)

    @classmethod
    def _constrain(
        cls, freq_thresh: int, length_thresh: int
    ) -> Tuple[Lines, VocabTally]:
        """
        Reduce dataset size by...
            - replacing words that rarely appear with <UNK>
            - omitting short sentences

        This saves an additional copy of the data (~0.9 GB) and vocab (~0.1 GB) to disk
        """

        def constrain(
            lines: Lines, vocab: VocabTally, freq_thresh: int, length_thresh: int
        ):
            UNK = "<UNK>"

            lines_filtered = []
            vocab_filtered = dict()
            vocab_filtered[UNK] = 0

            whitelist = set(k for k, v in vocab.items() if v >= freq_thresh)

            print("Constraining vocab...")
            for l in tqdm(lines):
                words = l.split()

                if len(words) < length_thresh:
                    continue

                for i, w in enumerate(words):
                    if w not in whitelist:
                        words[i] = "<UNK>"
                        vocab_filtered[UNK] += 1
                    else:
                        vocab_filtered.setdefault(w, 0)
                        vocab_filtered[w] += 1

                lines_filtered.append(" ".join(words))

            return (lines_filtered, vocab_filtered)

        def constrain_and_save(
            lines: Lines,
            vocab: VocabTally,
            freq_thresh: int,
            length_thresh: int,
            fp_lines: Path,
            fp_vocab: Path,
        ) -> Tuple[Lines, VocabTally]:
            lines, vocab = constrain(lines, vocab, freq_thresh, length_thresh)

            line_data = dict(data=lines, meta=dict(length_thresh=length_thresh))
            with open(fp_lines, "w+") as file:
                json.dump(line_data, file, indent=2)

            vocab_data = dict(data=vocab, meta=dict(freq_thresh=freq_thresh))
            with open(fp_vocab, "w+") as file:
                json.dump(vocab_data, file, indent=2)

            return lines, vocab

        def create():
            lines, vocab = cls._preprocess()
            return constrain_and_save(
                lines, vocab, freq_thresh, length_thresh, fp_lines, fp_vocab
            )

        fp_lines = paths.DATASET_DIR / "enwiki_lines_constrained.json"
        fp_vocab = paths.DATASET_DIR / "enwiki_vocab_constrained.json"

        if not fp_lines.exists() or not fp_vocab.exists():
            return create()
        else:
            # Load from file and validate
            with open(fp_lines) as file:
                lines_data = json.load(file)
                if lines_data["meta"]["length_thresh"] != length_thresh:
                    return create()
            with open(fp_vocab) as file:
                vocab_data = json.load(file)
                if vocab_data["meta"]["freq_thresh"] != freq_thresh:
                    return create()

        return lines_data["data"], vocab_data["data"]

    @classmethod
    def _preprocess(cls) -> Tuple[Lines, VocabTally]:
        """
        This function will...
            - download the dataset (~1.2 GB)
            - clean up unwanted characters and store a copy (~0.9 GB)
            - build a frequency count for each word (~0.1 GB)
        """

        fp_lines = paths.DATASET_DIR / "enwiki_lines.json"
        if fp_lines.exists():
            with open(fp_lines) as file:
                lines = json.load(file)
        else:
            print("Downloading dataset...")
            pipe = datasets.EnWik9(str(paths.DATA_DIR))
            loader = DataLoader2(pipe)

            print("Cleaning data...")
            lines = cls._clean_data(fp_lines, loader)

        fp_vocab = paths.DATASET_DIR / "enwiki_vocab.json"
        if fp_vocab.exists():
            with open(fp_vocab) as file:
                vocab = json.load(file)
        else:
            print("Counting vocab...")
            vocab = tally_vocab(" ".join(lines), verbose=True)
            with open(fp_vocab) as file:
                json.dump(vocab, file, indent=2)

        return (lines, vocab)

    @classmethod
    def _clean_data(cls, fp: Path, dataloader) -> Lines:
        """Join sentences and remove unwanted characters"""

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
            result = text

            # Convert line-breaks to spaces
            result = result.replace("\n", " ")

            # Remove abnormal characters
            result = re.sub("[^a-z0-9' ]", "", result, flags=re.IGNORECASE)

            # Replace periods with <END>
            result = re.sub(r"\. ", " <END> ", result)

            # Remove multi-spaces
            result = re.sub(r" +", " ", result)

            # Remove multi-quotes
            result = re.sub(r"'+", "'", result)

            return result.strip().lower()

        return main()
