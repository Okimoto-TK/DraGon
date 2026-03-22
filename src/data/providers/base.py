from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import polars as pl

from src.utils.log import vlog


def validate_query_args(
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    codes: Sequence[str] | None = None,
) -> None:
    if trade_date is not None and (start_date is not None or end_date is not None):
        raise ValueError("trade_date cannot be used together with start_date/end_date")

    if (start_date is None) ^ (end_date is None):
        raise ValueError("start_date and end_date must be provided together")

    if codes is not None:
        if isinstance(codes, str):
            raise TypeError("codes must be a sequence of strings, not a single string")
        for code in codes:
            if not isinstance(code, str):
                raise TypeError("all items in codes must be strings")


class RawProvider(ABC):
    def __init__(self, api):
        self.api = api

    def _raise_not_implemented(self, func):
        raise NotImplementedError(f"{self.api} does not support {func}!")

    def vlog(self, msg: str, level: str | None = None) -> None:
        if level is None:
            vlog(self.api, msg)
        else:
            vlog(self.api, msg, level)

    @abstractmethod
    def get_calendar(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw daily data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw daily data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_5min(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw 5-minute data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_moneyflow(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw moneyflow data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_st(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw ST status data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_suspend(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return raw suspend data in canonical raw schema.
        """
        raise NotImplementedError
