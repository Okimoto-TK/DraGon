from __future__ import annotations

from typing import Callable
from pathlib import Path

from config.api import MairuiConfig, TushareConfig
from src.data.schemas.raw import TableSchema
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.types import Query
from src.data.registry import PARAM_MAP
from src.data.validators.raw import validate_table


class RawPipeline:
    def __init__(self):
        self.tushare = TushareApi(TushareConfig())
        self.mairui = MairuiApi(MairuiConfig())

    def _fetch_data(
            self,
            api: Callable,
            provider: Callable,
            writer: Callable,
            path: Path,
            schema: TableSchema,
            query: Query,
            **kwargs
    ):
        df = provider(api(self), query)
        writer(df=df, dir_path=path, schema=schema, desc=query.desc)

    @staticmethod
    def _validate_date(
            self,
            reader: Callable,
            path: Path,
            schema: TableSchema,
            desc: str,
            **kwargs
    ):
        df = reader(path=path, desc=desc)
        validate_table(df=df, schema=schema)

    def run_raw_pipeline(
            self,
            query: Query,
    ):
        params = PARAM_MAP[query.desc]
        self._fetch_data(query=query, **params.model_dump())

    def run_raw_validation(self):
        for desc in PARAM_MAP.keys():
            self._validate_date(**PARAM_MAP[desc])
