from random import randint

import torch
from torch.utils.data import Dataset

Lines = list[str]
VocabTally = dict[str, int]


class SimpleDataset(Dataset):
    """
    Dataset that a model should eventually score 100% on

    Well, not really. Currently returns a random sequence that may be duplicated between training / test.
    Probably only gets you to 100% for large sequence lengths.
    """

    name = "HelloWorld"

    def __init__(
        self, sequence_length: int, vocab_size: int = 10, length: int | None = None
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.length = length or (vocab_size**sequence_length)
        self.vocab = [0 for i in range(vocab_size)]
        self.vocab_to_index = {str(i): i for i in range(vocab_size)}
        self.index_to_vocab = {i: str(i) for i in range(vocab_size)}

        print(
            f"Initialized SimpleDataset with {self.sequence_length=:,} {self.vocab_size=:,} {self.length=:,}"
        )

    @classmethod
    def load(cls, sequence_length: int, *args, **kwargs) -> "SimpleDataset":
        return cls(sequence_length)

    def __getitem__(self, index):
        sample = [
            randint(0, self.vocab_size - 1) for _ in range(self.sequence_length - 1)
        ]
        label = sum(sample) % self.vocab_size

        return torch.tensor(sample), label

    def __len__(self):
        return self.length
