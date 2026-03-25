from dataclasses import dataclass
from typing import Literal

TushareMode = Literal["official", "private"]


@dataclass(frozen=True)
class MairuiConfig:
    licence: str = "4A642DD5-3F9D-4EFE-8A68-56919734E95E"
    timeout: int = 2
    max_retries: int = 5
    time_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class TushareConfig:
    mode: TushareMode = "private"
    token: str = "nGiMftDngiLxiTZzpDVDxMpgOqrSPblmorglqyvXvLTvTNNhuJPyPNwXWZMzyVWz"
    timeout: int = 5
    http_url: str | None = "http://121.40.135.59:8010"
