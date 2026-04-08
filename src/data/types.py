"""Type aliases and literals for the data pipeline."""
from __future__ import annotations

from typing import Literal

# Mapping from source column names to canonical names
Map = dict[str, str]

# Stock exchange identifiers
Exchange = Literal["SSE", "SZSE", "BSE"]

# Stock status codes: L=listed, D=delisted, P=pre-IPO, G=growth
Status = Literal["L", "D", "P", "G"]

# Pipeline actions
Action = Literal["fetch", "load", "validate"]
