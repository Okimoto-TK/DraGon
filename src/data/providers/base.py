"""Abstract base class for raw data providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from src.data.models import Query
from src.utils.log import vlog


class RawProvider(ABC):
    """Base class for all data source providers (e.g., Tushare, Mairui).

    Each provider must implement methods to fetch various data types
    and return DataFrames conforming to the canonical raw schemas.
    """

    def __init__(self, api: str) -> None:
        """Initialize the provider with an API identifier.

        Args:
            api: Name of the API (e.g., "Tushare", "Mairui").
        """
        self.api = api

    def _raise_not_implemented(self, func: str) -> None:
        """Raise NotImplementedError for unsupported methods."""
        raise NotImplementedError(f"{self.api} does not support {func}!")

    def vlog(self, msg: str, level: str | None = None) -> None:
        """Log a message with this provider's API prefix."""
        if level is None:
            vlog(self.api, msg)
        else:
            vlog(self.api, msg, level)

    @abstractmethod
    def get_universe(self, query: Query) -> pl.DataFrame:
        """Return stock universe data."""
        raise NotImplementedError

    @abstractmethod
    def get_calendar(self, query: Query) -> pl.DataFrame:
        """Return trade calendar data."""
        raise NotImplementedError

    @abstractmethod
    def get_daily(self, query: Query) -> pl.DataFrame:
        """Return daily stock price data."""
        raise NotImplementedError

    @abstractmethod
    def get_adj_factor(self, query: Query) -> pl.DataFrame:
        """Return adjustment factor data."""
        raise NotImplementedError

    @abstractmethod
    def get_5min(self, query: Query) -> pl.DataFrame:
        """Return 5-minute interval stock data."""
        raise NotImplementedError

    @abstractmethod
    def get_moneyflow(self, query: Query) -> pl.DataFrame:
        """Return money flow data."""
        raise NotImplementedError

    @abstractmethod
    def get_namechange(self, query: Query) -> pl.DataFrame:
        """Return stock name change history."""
        raise NotImplementedError

    @abstractmethod
    def get_suspend(self, query: Query) -> pl.DataFrame:
        """Return stock suspension status data."""
        raise NotImplementedError
