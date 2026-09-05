"""Persistent sparse cache for the known raw-matrix region of the public H5AD."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import requests
from urllib3.exceptions import HTTPError

BLOCK = 1 << 20
RAW_START, RAW_END = 2_844_786_688, 5_051_514_880


class SparseHTTPFile(io.RawIOBase):
    def __init__(self, url, size, path):
        self.url, self.size, self.path = url, size, Path(path)
        identity = {"url": url, "size": size, "block_size": BLOCK}
        metadata = Path(str(path) + ".json")
        if metadata.exists():
            if json.loads(metadata.read_text()) != identity:
                raise ValueError("cache identity differs")
        else:
            if self.path.exists():
                raise ValueError("refusing an unrecognized existing cache")
            with metadata.open("x") as stream:
                json.dump(identity, stream)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(self.fd).st_size not in (0, size):
            raise ValueError("cache file size differs")
        os.ftruncate(self.fd, size)
        self.journal_path = Path(str(path) + ".blocks")
        self.blocks, self.locks = set(), {}
        self.lock, self.local, self.position = threading.Lock(), threading.local(), 0
        if self.journal_path.exists():
            for line in self.journal_path.read_text().splitlines():
                try:
                    block, digest = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = os.pread(self.fd, min(BLOCK, size - block * BLOCK), block * BLOCK)
                if hashlib.sha256(payload).hexdigest() == digest:
                    self.blocks.add(block)
        self.journal = self.journal_path.open("a")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def fetch_block(self, block):
        with self.lock:
            guard = self.locks.setdefault(block, threading.Lock())
        with guard:
            if block in self.blocks:
                return
            start, end = block * BLOCK, min((block + 1) * BLOCK, self.size)
            if not hasattr(self.local, "session"):
                self.local.session = requests.Session()
            for attempt in range(3):
                try:
                    response = self.local.session.get(
                        self.url, headers={"Range": f"bytes={start}-{end - 1}",
                                           "Accept-Encoding": "identity"}, timeout=(10, 45), stream=True)
                    expected = f"bytes {start}-{end - 1}/{self.size}"
                    with response:
                        if (response.status_code != 206
                                or response.headers.get("Content-Range") != expected):
                            raise ValueError("server returned an invalid byte range")
                        payload = response.raw.read(end - start + 1)
                        if len(payload) != end - start:
                            raise ValueError("server returned an invalid range length")
                    break
                except (requests.RequestException, HTTPError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            if os.pwrite(self.fd, payload, start) != len(payload):
                raise OSError("short cache write")
            digest = hashlib.sha256(payload).hexdigest()
            with self.lock:
                self.journal.write(json.dumps([block, digest]) + "\n")
                self.journal.flush()
                self.blocks.add(block)

    def read(self, length=-1):
        end = self.size if length < 0 else min(self.position + length, self.size)
        if end <= self.position:
            return b""
        for block in range(self.position // BLOCK, (end - 1) // BLOCK + 1):
            self.fetch_block(block)
        result = os.pread(self.fd, end - self.position, self.position)
        self.position += len(result)
        return result

    def readinto(self, buffer):
        payload = self.read(len(buffer))
        buffer[:len(payload)] = payload
        return len(payload)

    def close(self):
        if not self.closed:
            if hasattr(self, "journal"):
                self.journal.close()
            if hasattr(self, "fd"):
                os.close(self.fd)
        super().close()


def prefetch(cache):
    blocks = list(range(RAW_START // BLOCK, (RAW_END - 1) // BLOCK + 1))
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, _ in enumerate(pool.map(cache.fetch_block, blocks), 1):
            if number % 100 == 0 or number == len(blocks):
                print(f"{number}/{len(blocks)} raw blocks; {time.monotonic() - started:.1f}s", flush=True)
    os.fsync(cache.fd)
    print(f"CACHE_READY {cache.path}; allocated={os.stat(cache.path).st_blocks * 512}", flush=True)


if __name__ == "__main__":
    from experiments.development.recover_stephenson_tables import URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with SparseHTTPFile(URL, 7_187_322_881, args.cache) as cache:
        prefetch(cache)
