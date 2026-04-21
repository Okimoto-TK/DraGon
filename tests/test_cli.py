from __future__ import annotations

import src.cli as cli


def test_train_cli_passes_selected_task(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_training", _fake_run_training)

    cli.main(["train", "--task", "rv", "--name", "smoke"])

    assert captured["task"] == "rv"
    assert captured["name"] == "smoke"


def test_train_cli_allows_name_with_load(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_training", _fake_run_training)

    cli.main(["train", "--task", "ret", "--name", "ret13", "--load", "ret12"])

    assert captured["task"] == "ret"
    assert captured["name"] == "ret13"
    assert captured["load_name"] == "ret12"
