import json
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Tuple, TypedDict, cast

import torch
from discord import Intents, Message, TextChannel, Thread
from discord.ext import commands
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from config import paths
from utils.data_utils import tally_vocab
from utils.misc import load_toml

logger = logger.bind(cat="discord")
logger.add(
    paths.LOG_DIR / "discord_dataset.log",
    filter=lambda d: bool(d["extra"].get("discord")),
)


class _ChannelMessage(TypedDict):
    _message: "_DiscordMessage"
    _user: "_DiscordUser"


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
                    is_bot          INTEGER     NOT NULL,
                    user_name       TEXT        NOT NULL,

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

                    name            TEXT        NOT NULL,

                    PRIMARY KEY (id)
                ) STRICT;
                """
            )

    @classmethod
    def connect(cls):
        cls.db_fp.mkdir(exist_ok=True)

        db = sqlite3.connect(cls.db_fp)
        db.row_factory = sqlite3.Row

        return db

    def get_channel_history(self, id: int) -> _ChannelHistory:
        with self.connect() as DB:
            raw_messages: list[_DiscordMessage] = DB.execute(
                f"""
                SELECT * FROM messages,
                WHERE id_channel = ?
                """,
                (id,),
            ).fetchall()

            users = DB.execute(
                f"""
                SELECT * FROM users
                """
            ).fetchall()
            users = {u["id"]: u for u in users}

            channel: _DiscordChannel = DB.execute(
                f"""
                SELECT * FROM channels
                wHERE id = ? 
                """,
                (id,),
            ).fetchone()

        messages = []
        for m in raw_messages:
            messages.append(dict(message=m, user=users[m["id_user"]]))

        result = _ChannelHistory(messages=messages, channel=channel)
        return result


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

        return bot._messages

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
                logger.error(
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
            logger.info(f"Scraping channel {ch.name} ({ch.id})")
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
            logger.info(f"Found {len(found)} new messages")


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

        requests = []
        for id in CONFIG["discord"]["scraped_channels"]:  # type: ignore
            before = None
            after = None

            with _Database.connect() as DB:
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

        result = []
        for req in requests:
            msgs = _Bot.scrape([req])
            with _Database().connect() as DB:
                for m in msgs:
                    DB.execute(
                        """
                        INSERT INTO messages 
                        ( id,  id_user,  id_channel,  id_guild,  content,  date_created,  date_edited,  date_scraped,  is_bot,  user_name) VALUES
                        (:id, :id_user, :id_channel, :id_guild, :content, :date_created, :date_edited, :date_scraped, :is_bot, :user_name)
                        """,
                        m,
                    )
            result.extend(msgs)

        return result

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
    def get_lines(self) -> Lines:
        data = _Scraper().get_data()

        # Filter non-conversational messages
        data_filtered: list[_DiscordMessage] = []
        for m in data:
            if m["is_bot"]:
                continue
            if m["content"].strip().startswith("!"):
                continue
            data_filtered.append(m)

        # Group by channel
        data_channels: dict[int, list[_DiscordMessage]] = dict()
        for m in data_filtered:
            data_channels.setdefault(m["id_channel"], [])
            data_channels[m["id_channel"]].append(m)

        # Group consecutive messages
        data_users: dict[int, list[list[_DiscordMessage]]] = dict()
        for ch, msgs in data_channels.items():
            data_users[ch] = []
            msgs.sort(key=lambda d: d["date_created"])
            msgs = list(msgs)

            buffer = [msgs.pop(0)]
            current_user = buffer[0]["id_user"]
            while msgs:
                m = msgs.pop(0)
                u = m["id_user"]
                if u != current_user:
                    data_users[ch].append(buffer)

                    buffer = [m]
                    current_user = u
                else:
                    buffer.append(m)
            if buffer:
                data_users[ch].append(buffer)

        # Prefix message groups with user
        result: list[str] = []
        # @TODO

        return list(data_users.values())

    @lru_cache(1)
    def get_cleaned_lines(self) -> Lines:
        fp_cleaned = self.fp_base / "cleaned.json"
        if fp_cleaned.exists():
            with open(fp_cleaned) as file:
                return json.load(file)

        result = []
        lines = self.get_lines()

        print("Cleaning data...")
        for l in tqdm(lines):
            cleaned = l

            # Convert line-breaks to spaces
            cleaned = cleaned.replace("\n", f" {NEWLINE} ")

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

        logger.info("Filtering...")
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


# @todo reusable n-gram dataset
class DiscordDataset(Dataset):
    name = "Discord"
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
    def load(cls, sequence_length: int, freq_thresh: int = 100) -> "DiscordDataset":
        lines, vocab = _Generator().get_filtered_lines_vocab(
            freq_thresh, sequence_length
        )

        ds = cls(lines, vocab, sequence_length)
        logger.info(
            f"Loaded Discord dataset with {len(lines):,} sentences and {sum(vocab.values()):,} total words and {len(vocab):,} unique words and {sequence_length=}"
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


if __name__ == "__main__":
    _Scraper().fetch_messages()
