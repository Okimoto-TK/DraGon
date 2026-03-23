from __future__ import annotations

from typing import Sequence

import polars as pl
import requests
import inspect
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from tqdm import tqdm

import config.conf as conf
from config.api import MairuiConfig
from src.data.providers.base import RawProvider
from src.data.schemas.raw import RAW_5MIN_SCHEMA
from src.data.validators.raw import validate_table
from src.data.providers.api.registry import FIELD_MAP_5MIN
from src.data.types import Query


class MairuiApi(RawProvider):
    def __init__(self, config: MairuiConfig):
        super().__init__("Mairui")

        self.vlog(f"Creating Mairui API Instance...")

        self.licence = config.licence
        self.retry = Retry(
            total=config.max_retries,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=['HEAD', 'GET', 'OPTIONS'],
            backoff_factor=1
        )
        self.timeout = config.timeout
        self.time_format = config.time_format
        self.session = self._get_session()

    def _get_session(self):
        adapter = HTTPAdapter(max_retries=self.retry)
        session = requests.Session()
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _request_json(self, session: requests.Session, code: str, start_date: str, end_date: str):
        self.vlog(f"Requesting JSON for {code}...")

        url = f'https://api.mairuiapi.com/hsstock/history/{code}/5/n/{self.licence}'
        params = {
            "st": start_date,
            "et": end_date
        }
        try:
            response = session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            return result
        except Exception as e:
            self.vlog(f"Failed to request JSON for {code}: {e}", level="ERROR")
            raise e

    def get_calendar(
            self,
            query: Query,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_daily(
            self,
            query: Query,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_5min(
            self,
            query: Query,
    ) -> pl.DataFrame:
        self.vlog(f"Fetching {query.desc} data...")

        results = []
        for code in tqdm(query.codes, desc=f"Fetching {query.desc}:", disable=conf.debug):

            if query.trade_date is not None:
                self.vlog(f"trade_date exists, asof-date=trade_date.")

                _df = pl.DataFrame(self._request_json(
                    session=self.session,
                    code=code,
                    start_date=query.trade_date,
                    end_date=query.trade_date
                ))
            else:
                self.vlog(f"No trade_date provided, using start/end_date as default.")
                _df = pl.DataFrame(self._request_json(
                    session=self.session,
                    code=code,
                    start_date=query.start_date,
                    end_date=query.end_date
                ))
            _df = _df.rename(FIELD_MAP_5MIN).with_columns(
                code=pl.lit(code)
            )
            results.append(_df)

        if len(results) == 0:
            self.vlog(f"No data received, building empty dataframe...", level="WARNING")

            df = pl.DataFrame(schema=RAW_5MIN_SCHEMA.column_names_and_types)
        else:
            df = pl.concat(results).with_columns(
                pl.col("trade_time").str.to_datetime(self.time_format, strict=conf.debug).dt.date().alias("trade_date"),
                pl.col("trade_time").str.to_datetime(self.time_format, strict=conf.debug).dt.time().alias("time")
            ).drop("trade_time")

        validate_table(df, RAW_5MIN_SCHEMA)

        if query.codes is not None:
            df = df.with_columns(
                pl.col("code").is_in(query.codes)
            )

        self.vlog(f"Done, exiting.")
        return df

    def get_moneyflow(
            self,
            query: Query
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_st(
            self,
            query: Query,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_suspend(
            self,
            query: Query,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)
