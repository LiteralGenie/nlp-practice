import json
import pickle
from pathlib import Path
from typing import IO, Any


class _Cache:
    """
    Reads / writes a meta property when saving data.
    If the meta data does not match what's expected, the cache is ignored
    """

    fp: Path
    dump_mode = "r"
    load_mode = "w+"

    def __init__(self, fp: str | Path):
        self.fp = Path(fp) if not isinstance(fp, Path) else fp

    def dump(
        self,
        data,
        meta: dict | None = None,
    ) -> None:
        self.fp.parent.mkdir(exist_ok=True, parents=True)

        with open(self.fp, self.dump_mode) as file:
            file_data = dict(
                data=data,
                meta=meta or dict(),
            )
            self._dump(file_data, file)

    def _dump(self, data: dict, file: IO) -> None:
        raise NotImplementedError

    def load(
        self,
        meta: dict | None,
    ) -> Any | None:
        if not self.fp.exists():
            return None

        with open(self.fp, self.load_mode) as file:
            raw_data = self._load(file)

            if meta:
                is_meta_changed = self.is_dict_equal(meta, raw_data.get("meta", dict()))
                if is_meta_changed:
                    return None

            return raw_data["data"]

    def _load(self, file: IO):
        raise NotImplementedError

    @staticmethod
    def is_dict_equal(left: dict, right: dict) -> bool:
        """Shallow dict comparison"""

        # Check keys
        is_keys_equal = set(left.keys()) == set(right.keys())
        if not is_keys_equal:
            return False

        # Check vals
        for key in left:
            if left[key] != right[key]:
                return False

        return True


class JsonCache(_Cache):
    dump_mode = "w+"
    load_mode = "r"

    def _dump(self, data: dict, file: IO) -> None:
        json.dump(data, file, indent=2)

    def _load(self, file):
        return json.load(file)


class PickleCache(_Cache):
    dump_mode = "wb+"
    load_mode = "rb"

    def _dump(self, data: dict, file: IO) -> None:
        pickle.dump(data, file)

    def _load(self, file):
        return pickle.load(file)
