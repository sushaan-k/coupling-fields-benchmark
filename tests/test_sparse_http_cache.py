from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from experiments.development import sparse_http_cache as cache_module


def test_streamed_timeout_is_retried_before_a_block_is_recorded(tmp_path, monkeypatch):
    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-7/8"}

        def __init__(self):
            self.raw, self.attempts = self, 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return self

        def read(self, length):
            self.attempts += 1
            if self.attempts == 1:
                raise cache_module.HTTPError("synthetic streamed timeout")
            return b"12345678"

    monkeypatch.setattr(cache_module.time, "sleep", lambda _: None)
    with cache_module.SparseHTTPFile("http://example.invalid", 8, tmp_path / "cache") as stream:
        response = Response()
        stream.local.session = response
        assert stream.read(8) == b"12345678"
        assert response.attempts == 2
        assert stream.blocks == {0}


def test_sparse_cache_roundtrip_resume_and_range_validation(tmp_path, monkeypatch):
    payload = bytes(range(251)) * 9000
    requests = []

    class Handler(BaseHTTPRequestHandler):
        bad = False

        def log_message(self, *args):
            pass

        def do_GET(self):
            start, stop = map(int, self.headers["Range"].split("=")[1].split("-"))
            requests.append((start, stop))
            self.send_response(200 if self.bad else 206)
            self.send_header("Content-Range", f"bytes {start}-{stop}/{len(payload)}")
            self.send_header("Content-Length", str(stop - start + 1))
            self.end_headers()
            self.wfile.write(payload[start:stop + 1])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/matrix"
    path = tmp_path / "matrix.h5"
    try:
        with cache_module.SparseHTTPFile(url, len(payload), path) as stream:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(stream.fetch_block, [0, 0, 1, 2]))
            assert len(requests) == 3
            stream.seek(cache_module.BLOCK - 4)
            assert stream.read(19) == payload[cache_module.BLOCK - 4:cache_module.BLOCK + 15]
            stream.seek(-11, 2)
            target = bytearray(11)
            assert stream.readinto(target) == 11
            assert bytes(target) == payload[-11:]
        with cache_module.SparseHTTPFile(url, len(payload), path) as stream:
            assert stream.read() == payload
            assert len(requests) == 3
        with path.open("r+b") as damaged:
            damaged.write(b"wrong")
        with cache_module.SparseHTTPFile(url, len(payload), path) as stream:
            assert stream.read(5) == payload[:5]
            assert len(requests) == 4
        Handler.bad = True
        monkeypatch.setattr(cache_module.time, "sleep", lambda _: None)
        with cache_module.SparseHTTPFile(url, len(payload), tmp_path / "bad.h5") as stream:
            with pytest.raises(ValueError, match="invalid byte range"):
                stream.read(1)
            assert not stream.blocks
        assert len(requests) == 7
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
