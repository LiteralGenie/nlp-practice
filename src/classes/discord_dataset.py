import json
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from typing import TypedDict

from discord import Intents, TextChannel
from discord.ext import commands
from loguru import logger
from tqdm import tqdm

from config import paths
from utils.misc import load_toml

logger = logger.bind(cat="discord")
logger.add(
    paths.LOG_DIR / "discord_scrape.log",
    filter=lambda d: bool(d["extra"].get("discord")),
)


class _ChRequest(TypedDict):
    channel_id: int
    after: float | None
    before: float | None


class _Message(TypedDict):
    id: int
    id_user: int
    id_channel: int
    id_guild: int | None

    content: str
    date_created: float
    date_edited: float
    date_scraped: float


class _Bot(commands.Bot):
    _tgt_channels: list[_ChRequest] = []
    _messages: list[_Message]

    async def __init__(self, *args, **kwargs):
        intents = Intents.default()
        intents.message_content = True

        super().__init__(*args, intents=intents, **kwargs)

    async def on_ready(self):
        await self._scrape()
        await self.close()

    @classmethod
    def scrape(cls, channels: list[_ChRequest]):
        SECRETS = load_toml(paths.SECRETS_FILE)

        bot = cls()
        bot._tgt_channels = channels
        bot.run(SECRETS["discord_key"])

        return bot._messages

    async def _scrape(self):
        for tgt in self._tgt_channels:
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

            if not isinstance(ch, TextChannel):
                logger.error(
                    f"Channel {ch.__dict__.get('name', '???')} ({ch.id}) should be TextChannel and not {ch.__class__}"
                )
                return

            logger.info(f"Scraping channel {ch.name} ({ch.id})")
            async for msg in ch.history(before=before, after=after):
                self._messages.append(
                    _Message(
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


class _Scraper:
    db_fp = paths.DATASET_DIR / "discord"

    def __init__(self):
        self._init_tables()

    def _init_tables(self):
        with self._get_db() as DB:
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

                    PRIMARY KEY (id)
                ) STRICT;
                """
            )

    def _get_db(self):
        self.db_fp.mkdir(exist_ok=True)

        db = sqlite3.connect(self.db_fp)
        db.row_factory = sqlite3.Row

        return db

    def fetch_messages(self):
        CONFIG = load_toml(paths.CONFIG_FILE)

        requests = []
        for id in CONFIG["scraped_channels"]:
            before = None
            after = None

            with self._get_db() as DB:
                min_date = DB.execute(
                    """
                    SELECT date_created FROM messages
                    WHERE id_channel = ?
                    ORDER BY date_created ASC
                    LIMIT 1
                    """,
                    (id,),
                ).fetchone()
                before = min_date["date_created"] or None

                max_date = DB.execute(
                    """
                    SELECT date_created FROM messages
                    WHERE id_channel = ?
                    ORDER BY date_created DESC
                    LIMIT 1
                    """,
                    (id,),
                ).fetchone()
                after = max_date["date_created"] or None

            req = _ChRequest(channel_id=id, before=before, after=after)
            requests.append(req)

        results = _Bot.scrape(requests)
        with self._get_db() as DB:
            for r in results:
                DB.execute(
                    """
                    INSERT INTO messages 
                    ( id,  id_user,  id_channel,  id_guild,  content,  date_created,  date_edited,  date_scraped) VALUES
                    (:id, :id_user, :id_channel, :id_guild, :content, :date_created, :date_edited, :date_scraped)
                    """,
                    (r,),
                )

        return results

    def get_messages(self) -> list[_Message]:
        with self._get_db() as DB:
            return DB.execute("SELECT * from messages").fetchall()


