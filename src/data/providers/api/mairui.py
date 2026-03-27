from __future__ import annotations

import asyncio
import httpx
import polars as pl
import polars.selectors as ps
import inspect
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt
import random
from tqdm.asyncio import tqdm as tqdm

import config.config as config
from config.api import MairuiConfig
from src.data.providers.api.registry import FETCH_FIELD_5MIN
from src.data.providers.base import RawProvider
from src.data.schemas.raw import RAW_5MIN_SCHEMA
from src.data.validators.raw import validate_table
from src.data.providers.api.registry import FIELD_MAP_5MIN
from src.data.models import Query
from src.data.utils.raw import align_df


class MairuiApi(RawProvider):
    def __init__(self, api_config: MairuiConfig):
        super().__init__("Mairui")

        self.vlog(f"Creating Mairui API Instance...")

        self.licence = api_config.licence
        self.retry = api_config.max_retries
        self.retry_timeout = api_config.retry_timeout
        self.time_format = api_config.time_format
        self.client = httpx.AsyncClient(
            timeout=api_config.timeout,
            headers={
                "Accept-Encoding": "gzip, deflate, br",
                "User-Agent": "MairuiQuant/1.0"
            }
        )
        self.semaphore = asyncio.Semaphore(api_config.semaphore)

    async def _request(self, code: str, start_date: str, end_date: str) -> pl.DataFrame:
        self.vlog(f"Requesting JSON for {code}...")

        url = f'https://api.mairuiapi.com/hsstock/history/{code}/5/n/{self.licence}'
        params = {
            "st": start_date,
            "et": end_date
        }
        try:
            async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.retry),
                    wait=wait_exponential(multiplier=1, min=2, max=self.retry_timeout),
                    reraise=True
            ):
                with attempt:
                    response = await self.client.get(url=url, params=params)
                    response.raise_for_status()
                    result = response.json()
                    try:
                        df = pl.from_dicts(result, schema=FETCH_FIELD_5MIN)
                        if df.is_empty():
                            return pl.DataFrame(schema=FETCH_FIELD_5MIN)
                        else:
                            df = df.with_columns(
                                code=pl.lit(code)
                            ).with_columns(
                                ps.all().exclude(["code", "t"]).map_batches(lambda s: s.cast(pl.Float64))
                            )
                        return df
                    except Exception:
                        return pl.DataFrame(schema=FETCH_FIELD_5MIN)
        except Exception as e:
            self.vlog(f"Failed to request JSON for {code}: {e}", level="ERROR")
            raise e

    async def _controller(self, code, start_date: str, end_date: str):
        async with self.semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            return await self._request(code=code, start_date=start_date, end_date=end_date)

    async def _async_runner(self, query: Query, codes: pl.DataFrame):
        tasks = [self._controller(code, query.start_date, query.end_date) for code in codes.get_column("code").to_list()]
        results = await tqdm.gather(
            *tasks,
            desc=f"Fetching {query.desc}...",
            disable=config.debug
        )
        return results

    def get_universe(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_calendar(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_daily(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_adj_factor(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_5min(
            self,
            query: Query,
            codes: pl.DataFrame | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        self.vlog(f"Fetching {query.desc} data...")

        results = asyncio.run(self._async_runner(query, codes))
        results = [r for r in results if not r.is_empty()]
        df = (pl.concat(results)
              .select(FETCH_FIELD_5MIN)
              .rename(FIELD_MAP_5MIN))

        if df.is_empty():
            self.vlog(f"No data received, building empty dataframe...", level="WARNING")

            df = pl.DataFrame(schema=RAW_5MIN_SCHEMA.column_names_and_types)
        else:
            df = df.with_columns(
                pl.col("trade_time").str.to_datetime(self.time_format, strict=config.debug).dt.date().alias("trade_date"),
                pl.col("trade_time").str.to_datetime(self.time_format, strict=config.debug).dt.time().alias("time")
            ).drop("trade_time").cast(RAW_5MIN_SCHEMA.column_names_and_types)

        if codes is not None:
            df = df.filter(
                pl.col("code").is_in(codes.get_column("code").to_list())
            )
        df = align_df(df,
                      codes=codes,
                      calendar=calendar,
                      start_date=query.start_date,
                      end_date=query.end_date)
        df = df.sort(["code", "trade_date", "time"])
        validate_table(df, RAW_5MIN_SCHEMA)

        self.vlog(f"Done, exiting.")
        return df

    def get_moneyflow(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_namechange(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_suspend(
            self,
            **_kwargs
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)
