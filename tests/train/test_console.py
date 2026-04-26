from __future__ import annotations

from src.train.console import EpochConsoleLogger, _format_metric_value


def test_console_logger_starts_lazily_and_displays_one_based_epoch() -> None:
    logger = EpochConsoleLogger(log_every=1, enabled=False)

    assert logger._started is False

    logger.start_phase(epoch=0, phase="train", total_steps=5)

    assert logger._started is True
    assert logger.task_id is not None
    task = logger.progress.tasks[0]
    assert task.description == "train epoch=1 field=ret task=mu"

    logger.close()


def test_console_logger_formats_small_losses_compactly() -> None:
    assert _format_metric_value(0.000565, digits=4) == "5.65e-4"
    assert _format_metric_value(0.0123, digits=4) == "0.0123"
    assert _format_metric_value(1.230000, digits=6) == "1.23"
