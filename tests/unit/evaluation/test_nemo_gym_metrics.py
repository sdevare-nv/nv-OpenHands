import json
from pathlib import Path

import pytest

from evaluation.benchmarks.swe_bench.nemo_gym_metrics import (
    update_nemo_gym_metrics,
)


def test_update_is_disabled_without_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEMO_GYM_METRICS_FPATH', raising=False)

    assert update_nemo_gym_metrics({'create_runtime_time': 1.25}) is False


def test_update_merges_non_null_values(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.json'
    path.write_text('{"existing": 7, "drop_me": null}')

    assert update_nemo_gym_metrics(
        {'connect_to_runtime_time': 2.5, 'ignored': None}, metrics_path=path
    )
    assert json.loads(path.read_text()) == {
        'existing': 7,
        'connect_to_runtime_time': 2.5,
    }


def test_update_rejects_malformed_json_without_replacing_it(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.json'
    original = '{not valid json'
    path.write_text(original)

    assert update_nemo_gym_metrics({'create_runtime_time': 1.25}, metrics_path=path) is False
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_update_rejects_invalid_utf8_without_replacing_it(tmp_path: Path) -> None:
    path = tmp_path / 'metrics.json'
    original = b'\xff\xfe\xfa'
    path.write_bytes(original)

    assert update_nemo_gym_metrics({'create_runtime_time': 1.25}, metrics_path=path) is False
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_update_returns_false_when_parent_is_missing(tmp_path: Path) -> None:
    missing_parent = tmp_path / 'missing'
    path = missing_parent / 'metrics.json'

    assert update_nemo_gym_metrics({'create_runtime_time': 1.25}, metrics_path=path) is False
    assert not path.exists()
    assert not missing_parent.exists()


def test_run_infer_uses_safe_metrics_helper() -> None:
    source = Path('evaluation/benchmarks/swe_bench/run_infer.py').read_text()

    assert 'from evaluation.benchmarks.swe_bench.nemo_gym_metrics import' in source
    assert 'def update_metrics(' not in source
    assert 'update_nemo_gym_metrics({"create_runtime_time"' in source
    assert 'update_nemo_gym_metrics({"connect_to_runtime_time"' in source
    assert 'update_nemo_gym_metrics({"initialize_runtime_time"' in source
