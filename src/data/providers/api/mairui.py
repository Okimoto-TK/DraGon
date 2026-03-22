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
from src.data.providers.base import RawProvider, validate_query_args
from src.data.schemas.raw import RAW_5MIN_SCHEMA
from src.data.validators.raw import validate_table
from src.data.providers.api.mapping import FIELD_MAP_5MIN


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
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_5min(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        validate_query_args(trade_date=trade_date, start_date=start_date, end_date=end_date, codes=codes)

        self.vlog(f"Fetching 5min data...")

        results = []
        for code in tqdm(codes, desc="Fetching 5min:", disable=conf.debug):

            if trade_date is not None:
                self.vlog(f"trade_date exists, asof-date=trade_date.")

                _df = pl.DataFrame(self._request_json(
                    session=self.session,
                    code=code,
                    start_date=trade_date,
                    end_date=trade_date
                ))
            else:
                self.vlog(f"No trade_date provided, using start/end_date as default.")
                _df = pl.DataFrame(self._request_json(
                    session=self.session,
                    code=code,
                    start_date=start_date,
                    end_date=end_date
                ))
            _df = _df.rename(FIELD_MAP_5MIN).with_columns(
                code=pl.lit(code)
            )
            results.append(_df)

        if len(results) == 0:
            self.vlog(f"No data received, building empty dataframe...", level="Warning")

            df = pl.DataFrame(schema=RAW_5MIN_SCHEMA.column_names_and_types)
        else:
            df = pl.concat(results).with_columns(
                pl.col("trade_time").str.to_datetime(self.time_format, strict=conf.debug).dt.date().alias("trade_date"),
                pl.col("trade_time").str.to_datetime(self.time_format, strict=conf.debug).dt.time().alias("time")
            ).drop("trade_time")

        validate_table(df, RAW_5MIN_SCHEMA)

        self.vlog(f"Done, exiting.")
        return df

    def get_moneyflow(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_st(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_suspend(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)
