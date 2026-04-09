"""API configuration for data providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TushareMode = Literal["official", "private"]


@dataclass(frozen=True)
class MairuiConfig:
    """Configuration for Mairui API client."""

    licence: str = "4A642DD5-3F9D-4EFE-8A68-56919734E95E"
    timeout: int = 3
    retry_timeout: int = 120
    max_retries: int = 15
    time_format: str = "%Y-%m-%d %H:%M:%S"
    semaphore: int = 8


@dataclass(frozen=True)
class TushareConfig:
    """Configuration for Tushare API client."""

    mode: TushareMode = "official"
    token: str = "73a0950ad993c5aa49444b84198f11b920f9d16ac56a816e81644eb5"
    timeout: int = 5
    http_url: str | None = None
