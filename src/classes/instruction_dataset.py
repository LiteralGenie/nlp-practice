import json
import random
from functools import lru_cache
from typing import ClassVar, TypedDict

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from config import paths

Lines = list[str]


def format_instruction(instruction: str, input: str) -> str:
    instruction = instruction.strip()
    input = input.strip()

    if input:
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        return (
            f"""
{prompt}

### Instruction:
{instruction}

### Input:
{input}

### Response:
""".strip()
            + "\n"
        )
    else:
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        return (
            f"""
{prompt}

### Instruction:
{instruction}

### Response:
""".strip()
            + "\n"
        )


class _RawData(TypedDict):
    instruction: str
    input: str
    output: str


class _TokenizedData(TypedDict):
    head: list[int]
    tail: list[int]


class _Generator:
    fp_base = paths.DATASET_DIR / "stanford_alpaca"

    def __init__(self):
        self.fp_base.mkdir(exist_ok=True)

    @lru_cache(1)
    def get_raw(self) -> list[_RawData]:
        with open(self.fp_base / "raw.json") as file:
            return json.load(file)


class InstructionDataset(Dataset):
    """https://github.com/tatsu-lab/stanford_alpaca"""

    name: ClassVar[str] = "Instructions"

    data: list[_TokenizedData]
    sequence_length: int
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        data: list[_TokenizedData],
        sequence_length: int,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()

        self.data = data
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls, tokenizer: PreTrainedTokenizer, sequence_length: int
    ) -> "InstructionDataset":
        raw_data = _Generator.get_raw()
        tokenized_data: list[_TokenizedData] = []
        invalid = []
        for i, d in enumerate(raw_data):
            head = format_instruction(d["instruction"], d["input"])
            head_tkns = tokenizer.encode(head)
            tail_tkns = tokenizer.encode(d["output"])

            if len(head_tkns) + len(tail_tkns) > sequence_length:
                invalid.append(dict(idx=i, data=d))

            tokenized_data.append(_TokenizedData(head=head_tkns, tail=tail_tkns))

        if invalid:
            msgs = [f"{x['idx']} - {x['data']['instruction']}"[:75] for x in invalid]
            if len(msgs) > 9:
                msgs = msgs[:9] + ["..."]
            msgs = "\n\t".join(msgs)
            logger.warning(
                f"Skipping {len(invalid)} sequences longer than {sequence_length} tokens\n\t{msgs}"
            )

        ds = cls(tokenized_data, sequence_length, tokenizer)
        logger.info(
            f"Loaded {cls.name} dataset with {len(ds):,} total samples and {sequence_length=:,}"
        )

        return ds

    def __getitem__(self, index: int):
        sample = self.data[index]

        tail_end = random.sample(range(len(sample["tail"])), k=1)[0]

        input = torch.LongTensor(sample["head"] + sample["tail"][:tail_end])
        label = torch.LongTensor(sample["tail"][tail_end])

        return input, label

    def __len__(self):
        return len(self.data)
