from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RawProvider(ABC):
    def _validate_query_args(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        if trade_date is not None and (start_date is not None or end_date is not None):
            raise ValueError("trade_date cannot be used together with start_date/end_date")

        if codes is not None:
            if isinstance(codes, str):
                raise TypeError("codes must be a sequence of strings, not a single string")
            for code in codes:
                if not isinstance(code, str):
                    raise TypeError("all items in codes must be strings")
