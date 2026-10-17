"""Checking work units in other processes.

Python threads share one interpreter lock, so checking work units in threads
tops out well below the number of cores (measured on 20 cores: 1.35 ms per unit
across 8 threads against 2.3 ms on one; 0.45 ms across 8 processes). A node
syncing thousands of blocks, each with a work unit and its shares, checks them in
a pool of processes instead, started when a long download begins and stopped when
it has been idle for a while. The processes only recompute transcripts; every
decision stays with the node.
"""

import multiprocessing
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

IDLE_SECONDS = 30
# Each worker checks one unit at a time with a single-threaded BLAS: parallelism
# comes from the processes, and a BLAS thread per core in every worker would
# oversubscribe the cores (measured: 8 such workers ran each unit 3x slower).
WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
_env_lock = threading.Lock()


def start_pool(workers: int) -> ProcessPoolExecutor:
    """A pool with all its workers started now, under WORKER_ENV. The BLAS reads its
    thread count when it loads, so a worker must inherit the setting at birth: the
    workers are spawned here, one per submitted wait (none can fall idle before the
    last is spawned), and our own environment is restored afterwards."""
    with _env_lock:
        saved = {k: os.environ.get(k) for k in WORKER_ENV}
        os.environ.update(WORKER_ENV)
        try:
            # spawn everywhere: fork (Linux's default before Python 3.14) copies a process
            # whose other threads may hold locks the child then waits on forever.
            pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            for _ in range(workers):
                pool.submit(time.sleep, 0.2)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return pool


def transcripts(headers: list, blobs_dir: str | None) -> list:
    """Each header's work-unit transcript, recomputed (None where it cannot be: data
    missing, unit outside its job, malformed). Runs in a worker process."""
    from . import work
    from .chain import MAX_JOB_BLOB, BlobStore, InvalidBlock, MissingData, seed_of, work_inputs
    from .tx import InvalidTx
    blobs = BlobStore(blobs_dir)
    out = []
    for hdr in headers:
        try:
            seed = seed_of(hdr)
            a, b = work_inputs(hdr, seed, blobs, MAX_JOB_BLOB)
            out.append(work.compute_unit(a, b, seed, product=False)[0])
        except (MissingData, InvalidBlock, InvalidTx, KeyError, TypeError, ValueError, AttributeError):
            out.append(None)
    return out


class ProcessChecker:
    """A lazily started pool of worker processes for transcripts."""

    def __init__(self, blobs_dir: str | None, workers: int | None = None):
        self.blobs_dir = blobs_dir
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
        self.pool: ProcessPoolExecutor | None = None
        self.lock = threading.Lock()
        self.last_used = 0.0
        self.closed = False

    def submit(self, headers: list) -> Future:
        with self.lock:
            if self.closed:
                raise RuntimeError("checker closed")
            if self.pool is None:
                try:
                    self.pool = start_pool(self.workers)
                except Exception:
                    self.closed = True                   # not retried for every block
                    raise
            self.last_used = time.time()
            try:
                return self.pool.submit(transcripts, headers, self.blobs_dir)
            except BrokenProcessPool:
                # Workers cannot start here (e.g. no importable main module): stay with threads.
                self.closed = True
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = None
                raise

    def stop_if_idle(self) -> None:
        with self.lock:
            if self.pool is not None and time.time() - self.last_used > IDLE_SECONDS:
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = None

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = None
