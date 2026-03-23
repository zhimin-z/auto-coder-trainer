"""Tests for SWE-Lego data utilities."""
import json
from pathlib import Path

from trainers.swe_lego.data import (
    _safe_run_sort_key,
    _iter_runs,
    _score_from_fields,
    convert_to_verifier_format,
)


def test_safe_run_sort_key_numeric() -> None:
    assert _safe_run_sort_key("run_1") == (1, "run_1")
    assert _safe_run_sort_key("run_10") == (10, "run_10")


def test_safe_run_sort_key_non_numeric() -> None:
    key = _safe_run_sort_key("run_metadata")
    assert key == (0, "run_metadata")


def test_safe_run_sort_key_no_underscore() -> None:
    key = _safe_run_sort_key("run")
    assert key == (0, "run")


def test_iter_runs_with_run_keys() -> None:
    item = {
        "instance_id": "test-1",
        "run_1": {
            "messages": [{"role": "user", "content": "fix bug"}],
            "patch": "diff ...",
            "score": 1,
        },
        "run_2": {
            "messages": [{"role": "user", "content": "fix bug v2"}],
            "patch": "diff2 ...",
            "score": 0,
        },
    }
    runs = _iter_runs(item)
    assert len(runs) == 2
    assert runs[0]["run"] == "run_1"
    assert runs[1]["run"] == "run_2"
    assert runs[0]["score"] == 1
    assert runs[1]["score"] == 0


def test_iter_runs_with_malformed_keys() -> None:
    """run_metadata should not crash the sort, and should be ignored if no messages."""
    item = {
        "instance_id": "test-2",
        "run_1": {"messages": [{"role": "user", "content": "hi"}], "patch": ""},
        "run_metadata": {"info": "not a real run"},
    }
    runs = _iter_runs(item)
    # run_metadata has no messages, so only run_1 returned
    assert len(runs) == 1
    assert runs[0]["run"] == "run_1"


def test_iter_runs_single_item_fallback() -> None:
    item = {
        "instance_id": "test-3",
        "messages": [{"role": "user", "content": "test"}],
        "patch": "some patch",
        "resolved": True,
    }
    runs = _iter_runs(item)
    assert len(runs) == 1
    assert runs[0]["score"] == 1


def test_score_from_fields() -> None:
    assert _score_from_fields({"score": 0.9}) == 1
    assert _score_from_fields({"score": 0.1}) == 0
    assert _score_from_fields({"resolved": True}) == 1
    assert _score_from_fields({"resolved": False}) == 0
    assert _score_from_fields({}) == 0


def test_convert_to_verifier_format(tmp_path: Path) -> None:
    input_data = [
        {
            "instance_id": "test-1",
            "messages": [
                {"role": "user", "content": "Fix the login bug"},
                {"role": "assistant", "content": "I'll fix it"},
            ],
            "patch": "diff --git a/login.py ...",
            "resolved": True,
        },
        {
            "instance_id": "test-2",
            "messages": [
                {"role": "user", "content": "Add feature"},
                {"role": "assistant", "content": "Done"},
            ],
            "patch": "",
            "resolved": False,
        },
    ]
    input_path = tmp_path / "trajectories.json"
    input_path.write_text(json.dumps(input_data))

    output_path = tmp_path / "verifier_data.jsonl"
    result = convert_to_verifier_format(input_path, output_path)

    assert result == output_path
    assert output_path.exists()

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["instance_id"] == "test-1"
    assert first["score"] == 1
    assert len(first["messages"]) == 3  # system + user + assistant
    assert "YES" in first["messages"][2]["content"]

    second = json.loads(lines[1])
    assert second["score"] == 0
    assert "NO" in second["messages"][2]["content"]
