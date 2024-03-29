import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, TypedDict, cast

import torch
from discord import Intents, Message, TextChannel, Thread
from discord.ext import commands
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer

from classes.cache import PickleCache
from config import paths
from utils.data_utils import tally_vocab
from utils.misc import load_toml

logger.add(
    paths.LOG_DIR / "discord_dataset.log",
    filter=lambda d: d["extra"].get("cat") == "discord",
)

file_log = logger.bind(cat="discord")


class _ChannelMessage(TypedDict):
    message: "_DiscordMessage"
    user: "_DiscordUser"


class _ChannelHistory(TypedDict):
    messages: list[_ChannelMessage]
    channel: "_DiscordChannel"


class _Database:
    """
    Store Discord messages
    """

    db_fp = paths.DATASET_DIR / "discord" / "db.sqlite"

    def __init__(self):
        self._init_tables()

    def _init_tables(self):
        with self.connect() as DB:
            DB.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER     NOT NULL,
                    id_user         INTEGER     NOT NULL,
                    id_channel      INTEGER     NOT NULL,
                    id_guild        INTEGER,

                    content         TEXT        NOT NULL,
                    date_created    REAL        NOT NULL,
                    date_edited     REAL        NOT NULL,
                    date_scraped    REAL        NOT NULL,

                    PRIMARY KEY (id),
                    FOREIGN KEY (id_user) REFERENCES users,
                    FOREIGN KEY (id_channel) REFERENCES channels
                ) STRICT;
                """
            )

            DB.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id              INTEGER     NOT NULL,

                    name            TEXT        NOT NULL,
                    is_bot          INTEGER     NOT NULL,

                    PRIMARY KEY (id)
                ) STRICT;
                """
            )

            DB.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    id              INTEGER     NOT NULL,

                    date_created    REAL        NOT NULL,
                    name            TEXT        NOT NULL,

                    PRIMARY KEY (id)
                ) STRICT;
                """
            )

    def connect(self):
        self.db_fp.parent.mkdir(exist_ok=True)

        db = sqlite3.connect(self.db_fp)
        db.row_factory = sqlite3.Row

        return db

    def get_channel_history(self, id: int) -> _ChannelHistory:
        with self.connect() as DB:
            raw_messages: list[_DiscordMessage] = DB.execute(
                f"""
                SELECT * FROM messages
                WHERE id_channel = ?
                ORDER BY date_created asc
                """,
                (id,),
            ).fetchall()
            raw_messages = [dict(x) for x in raw_messages]

            users = DB.execute(
                f"""
                SELECT * FROM users
                """
            ).fetchall()
            users = {u["id"]: dict(u) for u in users}

            channel: _DiscordChannel = DB.execute(
                f"""
                SELECT * FROM channels
                WHERE id = ? 
                """,
                (id,),
            ).fetchone()
            channel = dict(channel)

        messages: list[_ChannelMessage] = []
        for m in raw_messages:
            messages.append(_ChannelMessage(message=m, user=users[m["id_user"]]))

        result = _ChannelHistory(messages=messages, channel=channel)
        return result

    def insert(
        self,
        message: Optional["_DiscordMessage"] = None,
        channel: Optional["_DiscordChannel"] = None,
        user: Optional["_DiscordUser"] = None,
    ):
        with self.connect() as DB:
            if user:
                DB.execute(
                    """
                    INSERT OR REPLACE INTO users
                    ( id,  is_bot,  name) VALUES
                    (:id, :is_bot, :name)
                    """,
                    user,
                )

            if channel:
                DB.execute(
                    """
                    INSERT OR REPLACE INTO channels
                    ( id,  date_created,  name) VALUES
                    (:id, :date_created, :name)
                    """,
                    channel,
                )

            if message:
                DB.execute(
                    """
                    INSERT INTO messages 
                    ( id,  id_user,  id_channel,  id_guild,  content,  date_created,  date_edited,  date_scraped) VALUES
                    (:id, :id_user, :id_channel, :id_guild, :content, :date_created, :date_edited, :date_scraped)
                    """,
                    message,
                )
        pass


class _ChRequest(TypedDict):
    channel_id: int
    after: float | None
    before: float | None


class _DiscordMessage(TypedDict):
    id: int
    id_user: int
    id_channel: int
    id_guild: int | None

    content: str
    date_created: float
    date_edited: float
    date_scraped: float


class _DiscordUser(TypedDict):
    id: int

    is_bot: bool
    name: str


class _DiscordChannel(TypedDict):
    id: int

    date_created: float
    name: str


class _Bot(commands.Bot):
    """
    Search Discord for messages
    """

    _tgt_channels: list[_ChRequest]
    _messages: list[_DiscordMessage]
    _users: dict[int, _DiscordUser]
    _channels: dict[int, _DiscordChannel]

    def __init__(self, *args, **kwargs):
        self._tgt_channels = []
        self._messages = []
        self._users = dict()
        self._channels = dict()

        intents = Intents.default()
        intents.message_content = True

        super().__init__(*args, command_prefix="__never__", intents=intents, **kwargs)

    async def on_ready(self):
        await self._scrape()
        await self.close()

    @classmethod
    def scrape(cls, channels: list[_ChRequest]):
        SECRETS = load_toml(paths.SECRETS_FILE)

        bot = cls()
        bot._tgt_channels = channels
        bot.run(SECRETS["discord_key"])  # type: ignore

        return (bot._messages, bot._users, bot._channels)

    async def _scrape(self):
        for tgt in self._tgt_channels:
            # Prep search params
            ch = await self.fetch_channel(tgt["channel_id"])
            before = tgt["before"]
            after = tgt["after"]

            if before is None and after is None:
                before = datetime.now(tz=timezone.utc)
            else:
                before = (
                    datetime.fromtimestamp(before, tz=timezone.utc) if before else None
                )
                after = (
                    datetime.fromtimestamp(after, tz=timezone.utc) if after else None
                )

            if not isinstance(ch, (TextChannel, Thread)):
                file_log.error(
                    f"Channel {getattr(ch, 'name', '???')} ({ch.id}) should be TextChannel and not {ch.__class__}"
                )
                return

            # Save channel
            if ch.id not in self._channels:
                self._channels[ch.id] = _DiscordChannel(
                    id=ch.id,
                    date_created=ch.created_at.timestamp() if ch.created_at else 0,
                    name=ch.name,
                )

            # Prep for saving messages
            file_log.info(f"Scraping channel {ch.name} ({ch.id})")
            found: list[_DiscordMessage] = []

            history_iters = []
            if before:
                # old messages
                history_iters.append(ch.history(before=before, limit=None))
            if after:
                # new messages
                history_iters.append(ch.history(after=after, limit=None))

            # Loop over messages
            for iter in history_iters:
                pbar = tqdm_asyncio(iter)
                async for msg in pbar:
                    msg = cast(Message, msg)
                    pbar.set_description(f"{msg.created_at.isoformat()}")

                    # Save message
                    found.append(
                        _DiscordMessage(
                            id=msg.id,
                            id_user=msg.author.id,
                            id_channel=msg.channel.id,
                            id_guild=msg.guild.id if msg.guild else None,
                            content=msg.content,
                            date_created=msg.created_at.timestamp(),
                            date_edited=msg.edited_at.timestamp()
                            if msg.edited_at
                            else msg.created_at.timestamp(),
                            date_scraped=datetime.now(tz=timezone.utc).timestamp(),
                        )
                    )

                    # Save user
                    if msg.author.id not in self._users:
                        self._users[msg.author.id] = _DiscordUser(
                            id=msg.author.id,
                            is_bot=msg.author.bot,
                            name=msg.author.name,
                        )

            self._messages.extend(found)
            file_log.info(f"Found {len(found)} new messages")


class _Scraper:
    """
    Asks _Bot for messages and saves in _Database
    Also exposes the data to _Generator
    """

    def fetch_messages(self):
        """
        Scan for new messages and save to _Database
        """

        CONFIG = load_toml(paths.CONFIG_FILE)

        # Scrape
        requests = []
        for id in CONFIG["discord"]["scraped_channels"]:  # type: ignore
            before = None
            after = None

            with _Database().connect() as DB:
                min_date = DB.execute(
                    """
                    SELECT date_created FROM messages
                    WHERE id_channel = ?
                    ORDER BY date_created ASC
                    LIMIT 1
                    """,
                    (id,),
                ).fetchone()
                before = min_date["date_created"] if min_date else None

                max_date = DB.execute(
                    """
                    SELECT date_created FROM messages
                    WHERE id_channel = ?
                    ORDER BY date_created DESC
                    LIMIT 1
                    """,
                    (id,),
                ).fetchone()
                after = max_date["date_created"] if max_date else None

            req = _ChRequest(channel_id=id, before=before, after=after)
            requests.append(req)

        # Save
        for req in requests:
            messages, users, channels = _Bot.scrape([req])

            for ch in channels.values():
                _Database().insert(channel=ch)
            for user in users.values():
                _Database().insert(user=user)
            for msg in messages:
                _Database().insert(message=msg)

    def get_data(self) -> list[_ChannelHistory]:
        """
        Get cached messages in _Database
        """

        CONFIG = load_toml(paths.CONFIG_FILE)
        channels: list[int] = CONFIG["discord"]["scraped_channels"]  # type: ignore

        result = [_Database().get_channel_history(ch) for ch in channels]
        return result


Lines = list[str]
VocabTally = dict[str, int]

UNK = "<UNK>"
END = "<END>"
NEWLINE = "<NEWLINE>"


class _Generator:
    fp_base = paths.DATASET_DIR / "discord"

    def __init__(self):
        self.fp_base.mkdir(exist_ok=True)

    @lru_cache(1)
    def get_seqs_by_channel(self) -> Lines:
        """Returns one huge string per channel"""

        result: Lines = []

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
            messages_grouped_by_user: list[list[_ChannelMessage]] = []

            buffer = [messages_filtered[0]]
            current_user = buffer[0]["user"]

            for msg in messages_filtered[1:]:
                new_user = msg["user"]
                if new_user["id"] != current_user["id"]:
                    messages_grouped_by_user.append((buffer))

                    buffer = [msg]
                    current_user = new_user
                else:
                    buffer.append(msg)
            if buffer:
                messages_grouped_by_user.append((buffer))

            # Stringify
            parts: list[str] = []
            for messages in messages_grouped_by_user:
                pt = []
                user = messages[0]["user"]

                for m in messages:
                    pt.append(m["message"]["content"].strip())

                pt_text = f"{user['name'].lower()}: "
                pt_text += "\n".join(s for s in pt)
                parts.append(pt_text)

            parts_text = "\n".join(parts)
            result.append(parts_text)

        return result


class _Ngram(TypedDict):
    seq_idx: int
    start_idx: int
    end_idx: int


# @todo reusable n-gram dataset
class DiscordDataset(Dataset):
    name = "Discord"
    UNK = UNK
    END = END
    NEWLINE = NEWLINE

    ngrams: list[_Ngram]

    def __init__(
        self,
        sequences: list[list[int]],
        tokenizer: PreTrainedTokenizer,
        sequence_length: int,
    ):
        super().__init__()

        self.sequences = sequences
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.ngrams = []
        for seq_idx, seq in enumerate(sequences):
            if len(seq) < sequence_length:
                logger.warning(
                    f"Skipping sequence {seq_idx} with less than {sequence_length} tokens"
                )
                continue

            for token_idx, token in enumerate(seq[:-sequence_length]):
                self.ngrams.append(
                    _Ngram(
                        seq_idx=seq_idx,
                        start_idx=token_idx,
                        end_idx=token_idx + sequence_length,
                    )
                )

    @classmethod
    def load(
        cls, tokenizer: PreTrainedTokenizer, sequence_length: int, from_cache=False
    ) -> "DiscordDataset":
        cache = PickleCache(paths.CACHE_DIR / "discord" / "token_seqs.pkl")
        cache_meta = dict(sequence_length=sequence_length)

        token_seqs = None
        if from_cache:
            token_seqs = cache.load(meta=cache_meta)
        if token_seqs is None:
            lines = _Generator().get_seqs_by_channel()

            logger.info("Tokenizing messages...")
            token_seqs = [tokenizer.encode(l) for l in tqdm(lines)]
            cache.dump(token_seqs, meta=cache_meta)

        ds = cls(token_seqs, tokenizer, sequence_length)
        logger.info(
            f"Loaded {cls.name} dataset with {sum(len(x) for x in token_seqs):,} total tokens and {len(ds):,} total samples and {sequence_length=:,}"
        )

        return ds

    def __getitem__(self, index):
        data = self.ngrams[index]

        seq = self.sequences[data["seq_idx"]]
        start = data["start_idx"]
        end = data["end_idx"]

        sample = torch.LongTensor(seq[start:end])
        label = torch.LongTensor([seq[end]])

        return sample, label

    def __len__(self):
        return len(self.ngrams)


if __name__ == "__main__":
    # @TODO: generate dataset and cache after fetching messages
    _Scraper().fetch_messages()
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)
    # DiscordDataset.load(from_cache=False, sequence_length=1024, tokenizer=tokenizer)
