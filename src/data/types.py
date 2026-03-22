from typing import TypeAlias, Tuple, Dict, List
import polars as pl

DailyDF = Dict[Tuple[str], pl.DataFrame]
Map = Dict[str, str]