from pathlib import Path

import tomlkit


def load_toml(fp: Path | str):
    if isinstance(fp, str):
        fp = Path(fp)
    return tomlkit.parse(fp.read_text(encoding="utf-8"))


def dump_toml(data: tomlkit.TOMLDocument, fp: Path | str) -> None:
    if isinstance(fp, str):
        fp = Path(fp)

    with open(fp, "w") as file:
        tomlkit.dump(data, file)
