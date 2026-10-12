"""Persistent chain storage in SQLite: blocks, the main chain, recent per-block
state diffs, and the flat base state.

Each accepted block is written in one SQLite transaction (block, its state diff,
the main-chain update and any base move), so a crash leaves the database at
the last fully written block, never in between. On restart the node loads the
base state and stacks the recent diffs on it; no block is re-executed.

SQLite is in the Python standard library, so this adds no dependency. Not
consensus: another implementation may store things any way it likes.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager

import numpy as np

from .state import MAPS, State

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocks (hash TEXT PRIMARY KEY, height INTEGER NOT NULL, prev TEXT NOT NULL,
                                   chainwork TEXT NOT NULL, header TEXT NOT NULL, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS blocks_by_height ON blocks (height);
CREATE TABLE IF NOT EXISTS main   (height INTEGER PRIMARY KEY, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS diffs  (hash TEXT PRIMARY KEY, height INTEGER NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state  (map TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                                   PRIMARY KEY (map, key));
CREATE TABLE IF NOT EXISTS blob_marks (hash TEXT PRIMARY KEY, height INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS txindex (txid TEXT PRIMARY KEY, height INTEGER NOT NULL, block TEXT NOT NULL,
                                    pos INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS txindex_by_height ON txindex (height);
CREATE TABLE IF NOT EXISTS addrindex (address TEXT NOT NULL, height INTEGER NOT NULL, pos INTEGER NOT NULL,
                                      ref TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS addrindex_by_address ON addrindex (address, height, pos);
CREATE INDEX IF NOT EXISTS addrindex_by_height ON addrindex (height);
"""


def _default(o):
    if isinstance(o, set):
        return {"$set": sorted(o)}
    raise TypeError(f"cannot store {type(o).__name__}")


def _hook(d):
    return set(d["$set"]) if len(d) == 1 and "$set" in d else d


def enc(value) -> str:
    return json.dumps(value, default=_default, separators=(",", ":"), sort_keys=True)


def dec(text: str):
    return json.loads(text, object_hook=_hook)


def enc_diff(diff: dict) -> str:
    # Pairs instead of objects: some maps have integer keys, which JSON objects would turn into strings.
    return enc({"maps": {name: {"set": [[k, v] for k, v in part["set"].items()], "dead": list(part["dead"])}
                         for name, part in diff["maps"].items()},
                "open_disputes": diff["open_disputes"], "lattice": diff["lattice"]})


def dec_diff(text: str) -> dict:
    raw = dec(text)
    return {"maps": {name: {"set": {k: v for k, v in part["set"]}, "dead": set(part["dead"])}
                     for name, part in raw["maps"].items()},
            "open_disputes": raw["open_disputes"], "lattice": raw["lattice"]}


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.lock = threading.RLock()
        # Must come before the first table is created: lets pruning give space back to the disk.
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._depth = 0

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @contextmanager
    def transaction(self):
        """All writes inside commit together or not at all. Re-entrant."""
        with self.lock:
            if self._depth == 0:
                self.db.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    self.db.execute("ROLLBACK")
                raise
            self._depth -= 1
            if self._depth == 0:
                self.db.execute("COMMIT")

    def _q(self, sql: str, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # -- meta --

    def get_meta(self, key: str, default=None):
        rows = self._q("SELECT value FROM meta WHERE key = ?", (key,))
        return dec(rows[0][0]) if rows else default

    def set_meta(self, key: str, value) -> None:
        self._q("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, enc(value)))

    # -- blocks --

    def put_block(self, bh: str, block: dict, height: int, chainwork: int) -> None:
        self._q("INSERT OR IGNORE INTO blocks (hash, height, prev, chainwork, header, body) VALUES (?, ?, ?, ?, ?, ?)",
                (bh, height, block["header"]["prev"], str(chainwork), enc(block["header"]), enc(block)))

    def get_block(self, bh: str) -> dict | None:
        rows = self._q("SELECT body FROM blocks WHERE hash = ?", (bh,))
        return dec(rows[0][0]) if rows else None

    def get_header(self, bh: str) -> dict | None:
        """A block's header without reading its transactions (headers-first sync serves thousands)."""
        rows = self._q("SELECT header FROM blocks WHERE hash = ?", (bh,))
        return dec(rows[0][0]) if rows else None

    def get_chainwork(self, bh: str) -> int | None:
        rows = self._q("SELECT chainwork FROM blocks WHERE hash = ?", (bh,))
        return int(rows[0][0]) if rows else None

    def prune_bodies(self, from_height: int, below_height: int) -> int:
        """Drop the transactions of every block in [from_height, below_height), keeping
        its header (pruned nodes). Returns how many bodies were dropped."""
        with self.lock:
            cur = self.db.execute("UPDATE blocks SET body = '{\"header\":' || header || ',\"txs\":null}' "
                                  "WHERE height >= ? AND height < ?", (from_height, below_height))
            return cur.rowcount

    def reclaim_space(self, pages: int = 100_000) -> None:
        """Give freed pages back to the file system (needs auto_vacuum, set for new databases)."""
        with self.lock:
            # Each step of this pragma frees one page: the rows must be read for it to run.
            self.db.execute(f"PRAGMA incremental_vacuum({int(pages)})").fetchall()

    def size_pages(self) -> tuple[int, int]:
        """(pages in the file, free pages)."""
        with self.lock:
            return (self.db.execute("PRAGMA page_count").fetchone()[0],
                    self.db.execute("PRAGMA freelist_count").fetchone()[0])

    # -- marks for collecting unused blobs (two phases: mark, then delete after a while) --

    def blob_marks(self) -> dict[str, int]:
        return dict(self._q("SELECT hash, height FROM blob_marks"))

    def set_blob_marks(self, add: dict[str, int], remove) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO blob_marks (hash, height) VALUES (?, ?)", list(add.items()))
            self.db.executemany("DELETE FROM blob_marks WHERE hash = ?", [(h,) for h in remove])

    def has_block(self, bh: str) -> bool:
        return bool(self._q("SELECT 1 FROM blocks WHERE hash = ?", (bh,)))

    def blocks_from(self, min_height: int):
        """(hash, height, chainwork, block) for every stored block at or above min_height, by height."""
        rows = self._q("SELECT hash, height, chainwork, body FROM blocks WHERE height >= ? ORDER BY height",
                       (min_height,))
        return [(bh, height, int(work), dec(body)) for bh, height, work, body in rows]

    # -- main chain --

    def set_main_from(self, height: int, hashes: list[str], index: list | None = None) -> None:
        """Replace the main chain from `height` on. `index` holds the new blocks'
        (txid rows, address rows) for the transaction and address index, which
        thereby always matches the main chain (a reorg rewrites it from the fork)."""
        self._q("DELETE FROM main WHERE height >= ?", (height,))
        with self.lock:
            self.db.executemany("INSERT INTO main (height, hash) VALUES (?, ?)",
                                [(height + i, h) for i, h in enumerate(hashes)])
            self.db.execute("DELETE FROM txindex WHERE height >= ?", (height,))
            self.db.execute("DELETE FROM addrindex WHERE height >= ?", (height,))
            self.add_index(index or [])

    def add_index(self, index: list) -> None:
        with self.lock:
            for tx_rows, addr_rows in index:
                self.db.executemany("INSERT OR REPLACE INTO txindex (txid, height, block, pos) VALUES (?, ?, ?, ?)",
                                    tx_rows)
                self.db.executemany("INSERT INTO addrindex (address, height, pos, ref, role, kind) "
                                    "VALUES (?, ?, ?, ?, ?, ?)", addr_rows)

    def clear_index(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM txindex")
            self.db.execute("DELETE FROM addrindex")

    def tx_location(self, txid: str) -> tuple | None:
        """(height, block hash, position) of a transaction on the main chain."""
        rows = self._q("SELECT height, block, pos FROM txindex WHERE txid = ?", (txid,))
        return rows[0] if rows else None

    def address_history(self, address: str, limit: int = 50, before: int | None = None) -> list[tuple]:
        """(height, position, ref, role, kind) rows of an address, newest first. `ref`
        is the txid, or the block hash for a block reward."""
        if before is None:
            return self._q("SELECT height, pos, ref, role, kind FROM addrindex WHERE address = ? "
                           "ORDER BY height DESC, pos DESC LIMIT ?", (address, limit))
        return self._q("SELECT height, pos, ref, role, kind FROM addrindex WHERE address = ? AND height < ? "
                       "ORDER BY height DESC, pos DESC LIMIT ?", (address, before, limit))

    def main_hashes(self) -> list[str]:
        return [h for (h,) in self._q("SELECT hash FROM main ORDER BY height")]

    # -- state diffs of recent blocks --

    def put_diff(self, bh: str, height: int, diff: dict) -> None:
        self._q("INSERT OR REPLACE INTO diffs (hash, height, body) VALUES (?, ?, ?)", (bh, height, enc_diff(diff)))

    def get_diff(self, bh: str) -> dict | None:
        rows = self._q("SELECT body FROM diffs WHERE hash = ?", (bh,))
        return dec_diff(rows[0][0]) if rows else None

    def drop_diffs_to(self, height: int) -> None:
        self._q("DELETE FROM diffs WHERE height <= ?", (height,))

    def drop_diffs(self, hashes: list[str]) -> None:
        with self.lock:
            self.db.executemany("DELETE FROM diffs WHERE hash = ?", [(h,) for h in hashes])

    # -- the flat base state --

    def write_base(self, diff: dict) -> None:
        """Apply one block's diff to the stored base state."""
        with self.lock:
            for name, part in diff["maps"].items():
                if part["dead"]:
                    self.db.executemany("DELETE FROM state WHERE map = ? AND key = ?",
                                        [(name, enc(k)) for k in part["dead"]])
                if part["set"]:
                    self.db.executemany("INSERT OR REPLACE INTO state (map, key, value) VALUES (?, ?, ?)",
                                        [(name, enc(k), enc(v)) for k, v in part["set"].items()])
            self.set_meta("open_disputes", diff["open_disputes"])
            self.set_meta("lattice", diff["lattice"])

    def replace_base(self, state: State) -> None:
        """Make `state` (flat) the stored base state, replacing whatever was there."""
        with self.lock:
            self.db.execute("DELETE FROM state")
            for name, layer in state.layers():
                self.db.executemany("INSERT INTO state (map, key, value) VALUES (?, ?, ?)",
                                    [(name, enc(k), enc(v)) for k, v in layer.items()])
            self.set_meta("open_disputes", state.open_disputes)
            self.set_meta("lattice", state.lattice.tobytes().hex())

    def load_base(self) -> State:
        base = State()
        for name, key, value in self._q("SELECT map, key, value FROM state"):
            if name in MAPS:
                getattr(base, name).data[dec(key)] = dec(value)
        base.open_disputes = self.get_meta("open_disputes", 0)
        base._set_lattice(np.frombuffer(bytes.fromhex(self.get_meta("lattice")), dtype="<u2").copy())
        return base
