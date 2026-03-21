from dataclasses import dataclass
from typing import Literal

TushareMode = Literal["official", "private"]


@dataclass(frozen=True)
class MairuiConfig:
    licence: str = ""
    timeout: int = 2
    max_retries: int = 5


@dataclass(frozen=True)
class TushareConfig:
    mode: TushareMode = "official"
    token: str = ""
    timeout: int = 5
    http_url: str | None = None
