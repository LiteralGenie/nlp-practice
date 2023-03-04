import json
import re
from functools import lru_cache
from random import random
from typing import Tuple

import torch
from config import paths
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import tally_vocab

Lines = list[str]
VocabTally = dict[str, int]

UNK = "<UNK>"
END = "<END>"


class _DataGenerator:
    fp_base = paths.DATASET_DIR / "diplomacy"

    def __init__(self) -> None:
        self.fp_base.mkdir(exist_ok=True)

    @lru_cache(1)
    def get_raw(self) -> list[dict]:
        fp_raw = self.fp_base / "diplomacy.jsonl"

        if not fp_raw.exists():
            raise Exception(
                f"Cannot find {fp_raw}. Please download the dataset from https://sites.google.com/view/qanta/projects/diplomacy"
            )

        print("Loading dataset...")
        data = fp_raw.read_text()
        result = [json.loads(l) for l in data.splitlines()]
        return result

    @lru_cache(1)
    def get_raw_lines(self) -> Lines:
        raw_data = self.get_raw()

        result = []
        for data in raw_data:
            result.append(". ".join(data["messages"]))

        return result

    @lru_cache(1)
    def get_cleaned_lines(self) -> Lines:
        fp_cleaned = self.fp_base / "cleaned.json"
        if fp_cleaned.exists():
            with open(fp_cleaned) as file:
                return json.load(file)

        result = []
        lines = self.get_raw_lines()

        print("Cleaning data...")
        for l in tqdm(lines):
            cleaned = l

            # Convert line-breaks to spaces
            cleaned = cleaned.replace("\n", " ")

            # Remove abnormal characters
            cleaned = re.sub("[^a-z0-9' ]", "", cleaned, flags=re.IGNORECASE)

            # Remove multi-spaces
            cleaned = re.sub(r" +", " ", cleaned)

            # Remove multi-quotes
            cleaned = re.sub(r"'+", "'", cleaned)

            cleaned = cleaned.strip().lower()
            result.append(cleaned)

        with open(fp_cleaned, "w+") as file:
            json.dump(result, file, indent=2)

        return result

    @lru_cache(1)
    def get_vocab_tally(self) -> VocabTally:
        fp_vocab = self.fp_base / "vocab.json"
        if fp_vocab.exists():
            with open(fp_vocab) as file:
                return json.load(file)

        lines = self.get_cleaned_lines()

        print("Counting vocab...")
        result = tally_vocab("\n".join(lines), verbose=True)

        with open(fp_vocab, "w+") as file:
            json.dump(result, file, indent=2)

        return result

    @lru_cache(1)
    def get_filtered_lines_vocab(
        self, freq_thresh: int, length_thresh: int
    ) -> Tuple[Lines, VocabTally]:
        fp_filtered = self.fp_base / "filtered.json"
        if fp_filtered.exists():
            with open(fp_filtered) as file:
                result = json.load(file)
                cached_freq_thresh = result["meta"].get("freq_thresh")
                cached_length_thresh = result["meta"].get("length_thresh")
                if (
                    cached_freq_thresh == freq_thresh
                    and cached_length_thresh == length_thresh
                ):
                    return result["lines"], result["vocab"]

        result_lines = []
        result_vocab = dict()
        result_vocab[UNK] = 0

        lines = self.get_cleaned_lines()
        vocab = self.get_vocab_tally()

        whitelist = set(k for k, v in vocab.items() if v >= freq_thresh)

        print("Filtering...")
        for l in tqdm(lines):
            words = l.split()

            if len(words) < length_thresh:
                continue

            for i, w in enumerate(words):
                if w not in whitelist:
                    words[i] = UNK
                    result_vocab[UNK] += 1
                else:
                    result_vocab.setdefault(w, 0)
                    result_vocab[w] += 1

            result_lines.append(" ".join(words))

        with open(fp_filtered, "w+") as file:
            result = dict(
                lines=result_lines,
                vocab=result_vocab,
                meta=dict(freq_thresh=freq_thresh, length_thresh=length_thresh),
            )
            json.dump(result, file, indent=2)

        return (result_lines, result_vocab)


class DiplomacyDataset(Dataset):
    """https://sites.google.com/view/qanta/projects/diplomacy"""

    name = "Diplomacy"
    UNK = UNK
    END = END

    def __init__(self, lines: Lines, vocab: VocabTally, sequence_length: int):
        super().__init__()

        self.vocab = vocab
        self.sequence_length = sequence_length
        self.vocab_to_index = {w: i for i, w in enumerate(vocab)}
        self.index_to_vocab = {i: w for i, w in enumerate(vocab)}

        self.ngrams = []
        for l in lines:
            words = l.split()
            startMax = len(words) - sequence_length
            for i in range(0, startMax + 1):
                self.ngrams.append(words[i : i + sequence_length])

    @classmethod
    def load(cls, sequence_length: int, freq_thresh: int = 100) -> "DiplomacyDataset":
        lines, vocab = _DataGenerator().get_filtered_lines_vocab(
            freq_thresh, sequence_length
        )

        ds = cls(lines, vocab, sequence_length)
        print(
            f"Loaded Diplomacy dataset with with {len(lines):,} sentences and {len(vocab):,} words and {sequence_length=}"
        )

        return ds

    def __getitem__(self, index):
        words = self.ngrams[index]

        sample = words[:-1]
        label = words[-1]

        sample_idx = torch.LongTensor([self.vocab_to_index[w] for w in sample])
        label_idx = self.vocab_to_index[label]

        return sample_idx, label_idx

    def __len__(self):
        return len(self.ngrams)
