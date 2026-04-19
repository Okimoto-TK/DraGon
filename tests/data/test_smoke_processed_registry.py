from __future__ import annotations

from config.data import diff_d
from src.data.registry.processed import PROCESSED_PARAM_MAP


def test_processed_registry_exposes_diff_d() -> None:
    for desc in ["macro", "mezzo", "micro", "sidechain"]:
        assert PROCESSED_PARAM_MAP[desc].processor_kwargs["diff_d"] == diff_d
