from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments import acquire_gse299043_nonheld as acquisition
from experiments import reduce_gse299043_mln as reducer


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEMPLATE = (
    ROOT / "data/confirmation/gse299043_mln/source_manifest_template_v1.json"
)
FROZEN_PREFLIGHT = ROOT / "data/development/gse299043_mln/metadata_preflight_v1.tsv"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    template = json.loads(FROZEN_TEMPLATE.read_text())
    paths = {
        "TEMPLATE": "data/confirmation/source-template.json",
        "OUTPUT": "data/confirmation/source.json",
        "PREFLIGHT": "data/development/preflight.tsv",
        "MEMBER_DIR": "data/development/source-members",
        "PIECE_DIR": "data/development/library-pieces",
        "DONOR_DIR": "data/development/reduced-donors",
        "REDUCED_OUTPUT": "data/development/reduced.json",
        "DEVELOPMENT_ATTEMPT": "data/development/attempt.json",
        "DEVELOPMENT_REFUSAL": "results/development/refusal.json",
    }
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    for name, relative in paths.items():
        monkeypatch.setattr(acquisition, name, tmp_path / relative)
    acquisition.TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    acquisition.TEMPLATE.write_bytes(FROZEN_TEMPLATE.read_bytes())
    acquisition.PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    acquisition.PREFLIGHT.write_bytes(FROZEN_PREFLIGHT.read_bytes())
    monkeypatch.setattr(acquisition, "_assert_family_available", lambda: None)
    monkeypatch.setattr(
        acquisition, "_artifact_bindings", lambda: {"fixture": "a" * 64}
    )
    monkeypatch.setattr(
        acquisition.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 10**12})(),
    )
    return template


def _fake_reducers(template: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    by_filename = {member["filename"]: member for member in template["members"]}

    def reduce_library(
        h5ad_path: Path,
        donor: str,
        output_path: Path,
        *,
        phase: str,
    ) -> dict[str, Any]:
        member = by_filename[h5ad_path.name]
        assert phase == "development"
        digest = hashlib.sha256(member["filename"].encode()).hexdigest()
        payload = {
            "schema": "gse299043-mln-library-reduction/1.0",
            "status": "TARGET_MLN_LIBRARY_REDUCED",
            "donor": donor,
            "role": "development",
            "source_filename": member["filename"],
            "source_bytes": member["bytes"],
            "source_sha256": digest,
        }
        _write_json(output_path, payload)
        return payload

    def finalize_donor(
        piece_paths: list[Path],
        donor: str,
        output_path: Path,
        *,
        phase: str,
    ) -> dict[str, Any]:
        assert phase == "development"
        expected = sum(
            member["donor"] == donor and member["role"] == "development"
            for member in template["members"]
        )
        assert len(piece_paths) == expected
        assert all(path.is_file() for path in piece_paths)
        payload = {
            "schema": "gse299043-mln-reduced-donor/1.0",
            "status": "DONOR_REDUCTION_COMPLETE",
            "donor": donor,
            "role": "development",
            "cells": reducer.CELL_BUDGET,
            "entity_count": len(reducer.MARKERS) ** 2,
            "tables": [[128, 128, 128, 128]] * 81,
        }
        _write_json(output_path, payload)
        return payload

    monkeypatch.setattr(reducer, "reduce_library", reduce_library)
    monkeypatch.setattr(reducer, "finalize_donor", finalize_donor)


def test_frozen_template_binds_exact_preflight_and_development_plan() -> None:
    template, development = acquisition._validated_template()
    bindings = acquisition._artifact_bindings()
    assert acquisition._sha256(FROZEN_TEMPLATE) == acquisition.FROZEN_TEMPLATE_SHA256
    assert bindings["score_authorization_publication_template_sha256"] == (
        acquisition._sha256(acquisition.AUTH_PUBLICATION_TEMPLATE)
    )
    assert template["metadata_manifest"]["sha256"] == acquisition._sha256(
        FROZEN_PREFLIGHT
    )
    assert len(development) == acquisition.DEVELOPMENT_MEMBER_COUNT == 56
    assert sum(member["bytes"] for member in development) == 2_991_542_178
    assert [member["donor"] for member in development] == sorted(
        [member["donor"] for member in development],
        key=reducer.DEVELOPMENT_DONORS.index,
    )
    assert all(member["role"] == "development" for member in development)
    assert template["hashsolo_contract"]["single_tissue_one_hto_exception"] == {
        "donor": "694B",
        "filename": "GSE299043_694B_001.CZI-IA11512689.v2.h5ad",
        "metadata_preflight_sha256": acquisition.FROZEN_PREFLIGHT_SHA256,
        "metadata_preflight_tissue": "pooled:mesenteric lymph node",
        "normalized_hto_id": "694B-MLN-206",
        "rule": (
            "assign every cell only when this exact member contains exactly this "
            "sole normalized donor HTO; every other member requires at least two "
            "donor HTOs"
        ),
    }
    assert not {member["url"] for member in development}.intersection(
        member["url"] for member in template["members"] if member["role"] == "held"
    )


def test_held_member_is_refused_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = json.loads(FROZEN_TEMPLATE.read_text())
    held = next(member for member in template["members"] if member["role"] == "held")
    monkeypatch.setattr(acquisition, "MEMBER_DIR", tmp_path)
    monkeypatch.setattr(
        acquisition,
        "_download",
        lambda *args: (_ for _ in ()).throw(AssertionError("network opened")),
    )
    with pytest.raises(PermissionError, match="forbids every held member"):
        acquisition._download_member(held, tmp_path / held["filename"])


def test_download_requires_frozen_length_and_hashes_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"frozen fixture bytes"

    class Response:
        headers = {"Content-Length": str(len(content))}

        def __init__(self) -> None:
            self.remaining = content

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int) -> bytes:
            del size
            block, self.remaining = self.remaining, b""
            return block

    monkeypatch.setattr(
        acquisition.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    output = tmp_path / "member.h5ad"
    observed, digest = acquisition._download(
        "https://example.invalid/member.h5ad", output, len(content)
    )
    assert observed == len(content)
    assert digest == hashlib.sha256(content).hexdigest()
    assert output.read_bytes() == content


def test_download_retries_only_transient_transport_and_deletes_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"complete frozen fixture"
    calls = 0
    sleeps: list[int] = []

    class Response:
        headers = {"Content-Length": str(len(content))}

        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int) -> bytes:
            del size
            if self.fail:
                if not self.sent:
                    self.sent = True
                    return content[:5]
                raise ConnectionResetError("transient reset")
            if self.sent:
                return b""
            self.sent = True
            return content

    def urlopen(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return Response(calls < acquisition.DOWNLOAD_ATTEMPTS)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(acquisition.time, "sleep", sleeps.append)
    output = tmp_path / "member.h5ad"
    observed, digest = acquisition._download(
        "https://example.invalid/member.h5ad", output, len(content)
    )
    assert calls == acquisition.DOWNLOAD_ATTEMPTS == 3
    assert sleeps == [1, 2]
    assert observed == len(content)
    assert digest == hashlib.sha256(content).hexdigest()
    assert output.read_bytes() == content
    assert not output.with_name(output.name + ".part").exists()


def test_download_does_not_retry_byte_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    class Response:
        headers = {"Content-Length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int) -> bytes:
            del size
            return b""

    def urlopen(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return Response()

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        acquisition.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError(f"slept {seconds}")),
    )
    output = tmp_path / "member.h5ad"
    with pytest.raises(PermissionError, match="remote byte count differs"):
        acquisition._download(
            "https://example.invalid/member.h5ad", output, expected_bytes=7
        )
    assert calls == 1
    assert not output.exists()
    assert not output.with_name(output.name + ".part").exists()


def test_preflight_mutation_refuses_before_attempt_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    acquisition.PREFLIGHT.write_text(acquisition.PREFLIGHT.read_text() + "poison\n")
    monkeypatch.setattr(
        acquisition,
        "_download",
        lambda *args: (_ for _ in ()).throw(AssertionError("network opened")),
    )
    with pytest.raises(PermissionError, match="metadata preflight SHA-256"):
        acquisition.acquire()
    assert not acquisition.DEVELOPMENT_ATTEMPT.exists()


def test_success_requests_only_56_development_members_after_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = _patch_paths(tmp_path, monkeypatch)
    _fake_reducers(template, monkeypatch)
    held_urls = {
        member["url"] for member in template["members"] if member["role"] == "held"
    }
    requested: list[str] = []

    def download(url: str, destination: Path, expected_bytes: int) -> tuple[int, str]:
        assert acquisition.DEVELOPMENT_ATTEMPT.is_file()
        assert url not in held_urls
        assert not list(acquisition.MEMBER_DIR.glob("*.h5ad"))
        requested.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stream fixture")
        return expected_bytes, hashlib.sha256(destination.name.encode()).hexdigest()

    monkeypatch.setattr(acquisition, "_download", download)
    acquisition.acquire()

    assert len(requested) == len(set(requested)) == 56
    attempt = json.loads(acquisition.DEVELOPMENT_ATTEMPT.read_text())
    assert attempt["status"] == "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"
    assert attempt["development_members_planned"] == 56
    assert attempt["held_h5ad_members_requested"] == 0
    source = json.loads(acquisition.OUTPUT.read_text())
    development = [
        member for member in source["members"] if member["role"] == "development"
    ]
    held = [member for member in source["members"] if member["role"] == "held"]
    assert len(development) == 56 and all(member["sha256"] for member in development)
    assert len(held) == 151 and all(member["sha256"] is None for member in held)
    assert all(member["local_path"] is None for member in source["members"])
    assert all(member["retained"] is False for member in source["members"])
    reduced = json.loads(acquisition.REDUCED_OUTPUT.read_text())
    assert reduced["source_manifest_sha256"] == acquisition._sha256(acquisition.OUTPUT)
    assert [record["donor"] for record in reduced["donors"]] == list(
        reducer.DEVELOPMENT_DONORS
    )
    assert reduced["access_audit"] == {
        "development_h5ad_members_decoded": 56,
        "held_h5ad_members_decoded": 0,
        "held_h5ad_members_opened": 0,
        "maximum_concurrent_source_h5ads": 1,
    }
    assert not list(acquisition.MEMBER_DIR.glob("*.h5ad"))
    assert not list(acquisition.MEMBER_DIR.glob("*.part"))
    with pytest.raises(FileExistsError, match="development acquisition artifact"):
        acquisition.acquire()


def test_failure_after_attempt_deletes_h5ad_and_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)

    def download(url: str, destination: Path, expected_bytes: int) -> tuple[int, str]:
        del url
        assert acquisition.DEVELOPMENT_ATTEMPT.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"source poison")
        return expected_bytes, hashlib.sha256(destination.name.encode()).hexdigest()

    monkeypatch.setattr(acquisition, "_download", download)
    monkeypatch.setattr(
        reducer,
        "reduce_library",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("reducer poison")),
    )
    with pytest.raises(ValueError, match="reducer poison"):
        acquisition.acquire()
    refusal = json.loads(acquisition.DEVELOPMENT_REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL"
    assert refusal["rerun_permitted"] is False
    assert refusal["held_h5ad_members_requested"] == 0
    assert not list(acquisition.MEMBER_DIR.glob("*.h5ad"))
    assert not list(acquisition.MEMBER_DIR.glob("*.part"))
    monkeypatch.setattr(
        acquisition,
        "_download",
        lambda *args: (_ for _ in ()).throw(AssertionError("network reopened")),
    )
    with pytest.raises(FileExistsError, match="development acquisition artifact"):
        acquisition.acquire()
