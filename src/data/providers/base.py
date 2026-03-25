from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl

from src.utils.log import vlog
from src.data.models import Query


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
    def get_universe(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw daily data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_calendar(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw daily data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_daily(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw daily data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_adj_factor(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw adj factor data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_5min(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw 5-minute data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_moneyflow(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw moneyflow data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_st(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw ST status data in canonical raw schema.
        """
        raise NotImplementedError

    @abstractmethod
    def get_suspend(
            self,
            query: Query,
    ) -> pl.DataFrame:
        """
        Return raw suspend data in canonical raw schema.
        """
        raise NotImplementedError
