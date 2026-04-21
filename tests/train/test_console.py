from __future__ import annotations

from src.train.console import EpochConsoleLogger


def test_console_logger_starts_lazily_and_displays_one_based_epoch() -> None:
    logger = EpochConsoleLogger(log_every=1, enabled=False)

    assert logger._started is False

    logger.start_phase(epoch=0, phase="train", total_steps=5)

    assert logger._started is True
    assert logger.task_id is not None
    task = logger.progress.tasks[0]
    assert task.description == "train epoch=1"

    logger.close()
