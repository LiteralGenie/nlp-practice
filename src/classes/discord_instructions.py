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

from .discord_dataset import (
    _ChannelMessage,
    _DiscordChannel,
    _DiscordMessage,
    _DiscordUser,
    _Scraper,
)
from .instruction_dataset import format_prompt


class _GroupedMessage(TypedDict):
    messages: list[_DiscordMessage]
    user: _DiscordUser


class _ChannelData(TypedDict):
    channel: _DiscordChannel
    messages: list[_GroupedMessage]


class _Generator:
    fp_base = paths.DATASET_DIR / "discord"

    def __init__(self):
        self.fp_base.mkdir(exist_ok=True)

    @lru_cache(1)
    def get_messages(self) -> list[_ChannelData]:
        """Returns one huge string per channel"""

        result: list[_ChannelData] = []

        logger.info("Loading cached messages...")
        for history in tqdm(_Scraper().get_data()):
            # Filter bot commands and output
            messages_filtered: list[_ChannelMessage] = []
            for m in history["messages"]:
                if m["user"]["is_bot"]:
                    continue
                if m["message"]["content"].strip().startswith("!"):
                    continue
                messages_filtered.append(m)

            # Group consecutive messages from same user
            messages_grouped_by_user: list[_GroupedMessage] = []

            buffer = [messages_filtered[0]["message"]]
            current_user = messages_filtered[0]["user"]
            for msg in messages_filtered[1:]:
                new_user = msg["user"]
                if new_user["id"] != current_user["id"]:
                    messages_grouped_by_user.append(
                        _GroupedMessage(messages=buffer, user=current_user)
                    )

                    buffer = [msg["message"]]
                    current_user = new_user
                else:
                    buffer.append(msg["message"])
            if buffer:
                messages_grouped_by_user.append(
                    _GroupedMessage(messages=buffer, user=current_user)
                )

            result.append(
                _ChannelData(
                    channel=history["channel"], messages=messages_grouped_by_user
                )
            )

        return result


class DiscordInstructions(Dataset):
    name: ClassVar[str] = "DiscordInstructions"
    instruction: ClassVar[str] = "Continue the conversation."

    data: list[_ChannelData]
    sequence_length: int
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        data: list[_ChannelData],
        sequence_length: int,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()

        self.data = data
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls, tokenizer: PreTrainedTokenizer, sequence_length: int, from_cache=False
    ) -> "DiscordInstructions":
        cache = PickleCache(paths.CACHE_DIR / "discord" / "partitioned_channels.pkl")
        cache_meta = dict(sequence_length=sequence_length)

        data = None
        if from_cache:
            data = cache.load(meta=cache_meta)
        if data is None:
            raw_data = _Generator().get_messages()
            data: list[_ChannelData] = []
            for d in raw_data:
                data.extend(cls._partition(d, tokenizer, sequence_length))
            cache.dump(data, meta=cache_meta)

        ds = cls(data, sequence_length, tokenizer)
        logger.info(
            f"Loaded {cls.name} dataset with {len(ds):,} total samples and {sequence_length=:,}"
        )
        return ds

    @classmethod
    def _partition(
        cls, data: _ChannelData, tokenizer: PreTrainedTokenizer, sequence_length: int
    ) -> list[_ChannelData]:
        """Divide data.messages into _ChannelData's that (almost) never exceed sequence_length tokens

        This assumes grouping that messages at indicies [j] is always shorter than [j, j+1]
        but that may not be true due to _format_user_instruction works
        but it shouldn't matter unless sequence_length is teeny
        """

        channel = data["channel"]
        messages = data["messages"]

        result: list[_ChannelData] = []
        buffer = _ChannelData(channel=channel, messages=[messages[0]])
        pbar = tqdm(messages[1:100], desc=f"Partitioning #{channel['name']}...")
        for msg in pbar:
            tentative_msgs = buffer["messages"] + [msg]
            tentative_buffer = _ChannelData(channel=channel, messages=tentative_msgs)

            instruction = cls._format_instruction(tentative_buffer, randomize=False)
            input = cls._format_input(tentative_buffer)
            length = len(tokenizer.encode(format_prompt(instruction, input)))
            if length > sequence_length:
                result.append(buffer)
                buffer = _ChannelData(channel=channel, messages=[msg])
            else:
                buffer = tentative_buffer

        result.append(buffer)

        return result

    @classmethod
    def _format_input(cls, data: _ChannelData) -> str:
        input: list[str] = []
        for grp in data["messages"]:
            message_concat = (
                grp["user"]["name"].lower() + ": " + grp["messages"][0]["content"]
            )
            for msg in grp["messages"][1:]:
                message_concat += "\n" + msg["content"]

            input.append(message_concat)

        input_str = "\n".join(input)

        return input_str

    @classmethod
    def _format_instruction(cls, data: _ChannelData, randomize=True) -> str:
        users = list(set(m["user"]["name"] for m in data["messages"]))
        random.shuffle(users)

        append_channel_instruction = not randomize or random.random() < 0.5
        append_user_instruction = not randomize or random.random() < 0.5

        instruction = cls.instruction
        if append_channel_instruction:
            instruction += " " + cls._format_channel_instruction(
                data["channel"]["name"]
            )
        if append_user_instruction:
            instruction += " " + cls._format_user_instruction(users)

        return instruction

    @classmethod
    def _format_channel_instruction(cls, channel: str):
        channel = channel.lower()
        return f"The conversation takes place in #{channel}."

    @classmethod
    def _format_user_instruction(cls, users: list[str]) -> str:
        assert len(users)
        last = users[-1].lower()
        head = [u.lower() for u in users[:-1]]

        if head:
            others = ", ".join(head)
            return f"The conversation is between {others} and {last}."
        else:
            return f"The conversation is between {last} and himself / herself."

    @lru_cache(None)
    def _encode_cached(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def __getitem__(self, index: int):
        sample = self.data[index]

        # Select random subset of messages
        start = random.randrange(0, len(sample["messages"]))
        end = random.randrange(0, len(sample["messages"]))
        (start, end) = (start, end) if start <= end else (end, start)
        messages = sample["messages"][start : end + 1]

        # Tokenize
        channel_data = _ChannelData(channel=sample["channel"], messages=messages)
        instr = self._format_instruction(channel_data)
        input = self._format_input(channel_data)

        instr_tokens = self.tokenizer.encode(instr)
        input_tokens = self.tokenizer.encode(input)
        tokens = instr_tokens + input_tokens
        if len(tokens) > self.sequence_length:
            logger.warning(
                f"Sequence is longer than {self.sequence_length} ({len(tokens)})"
            )
            tokens = tokens[: self.sequence_length]

        # Truncate final line by random amount
        final_msg = self._format_input(
            _ChannelData(channel=sample["channel"], messages=[messages[-1]])
        )
        final_msg_tokens = self.tokenizer.encode(final_msg)
        min_end = len(tokens) - len(final_msg_tokens)
        end = random.randrange(min_end, len(tokens))
        tokens_truncated = tokens[:end]

        # Return
        input = torch.LongTensor(tokens_truncated[:-1])
        label = torch.LongTensor(tokens_truncated[-1])
        return input, label

    def __len__(self):
        return len(self.data)
