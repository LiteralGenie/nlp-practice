import json
import random
from functools import lru_cache
from typing import ClassVar, TypedDict

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from classes.cache import PickleCache
from config import paths

Lines = list[str]


def format_prompt(instruction: str, input: str) -> str:
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
        cache = PickleCache(paths.CACHE_DIR / "instructions" / "tokenized_data.pkl")

        # Tokenize
        tokenized_data = cache.load()
        if tokenized_data is None:
            raw_data = _Generator().get_raw()
            tokenized_data: list[_TokenizedData] = []
            for i, d in enumerate(tqdm(raw_data, desc="Tokenizing...")):
                head = format_prompt(d["instruction"], d["input"])
                head_tkns = tokenizer.encode(head)
                tail_tkns = tokenizer.encode(d["output"])

                tokenized_data.append(_TokenizedData(head=head_tkns, tail=tail_tkns))
            cache.dump(tokenized_data)

        # Filter by sequence length
        truncated = []
        invalid = []
        for i, d in enumerate(tokenized_data):
            overflow = len(d["head"]) + len(["tail"]) - sequence_length
            if overflow > 0:
                if len(d["tail"]) > overflow:
                    tokenized_data[i]["tail"] = d["tail"][:-overflow]
                    truncated.append(d)
                else:
                    invalid.append(d)

        # Log filtered
        if truncated:
            logger.info(
                f"Skipping {len(truncated)} un-truncate-able sequences longer than {sequence_length} tokens"
            )
        if invalid:
            msgs = [f"{x['idx']} - {x['data']['instruction']}"[:75] for x in invalid]
            if len(msgs) > 9:
                msgs = msgs[:9] + ["..."]
            msgs = "\n\t".join(msgs)
            logger.warning(
                f"Skipping {len(invalid)} un-truncate-able sequences longer than {sequence_length} tokens\n\t{msgs}"
            )
        tokenized_data = [x for x in tokenized_data if x not in truncated]

        # Return
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
