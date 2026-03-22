from __future__ import annotations

from typing import Sequence

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from tqdm import tqdm
from typing import Dict, Tuple

from config.api import MairuiConfig
from src.data.providers.base import RawProvider
from src.data.schemas.raw import RAW_5MIN_SCHEMA
from src.data.validators.raw import validate_table, validate_time_format
from src.data.providers.api.mapping import FIELD_MAP_5MIN
from src.data.utils.raw import partition_by


class MairuiApi(RawProvider):
    def __init__(self, config:MairuiConfig | None = None):
        self.licence = config.licence
        self.retry = Retry(
            total=config.max_retries,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=['HEAD', 'GET', 'OPTIONS'],
            backoff_factor=1
        )
        self.timeout = config.timeout

    def _get_session(self):
        adapter = HTTPAdapter(max_retries=self.retry)
        session = requests.Session()
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _request_json(self, session: requests.Session, code: str, start_date: str, end_date: str):
        url = f'https://api.mairuiapi.com/hsstock/history/{code}/5/n/{self.licence}'
        params = {
            "st": start_date,
            "et": end_date
        }
        try:
            response = session.get(url, params=params, timeout=self.timeout)
            result = response.json()
            return result
        except Exception as e:
            raise e

    def get_5min(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> Dict[Tuple, pl.DataFrame]:
        self._validate_query_args(trade_date=trade_date, start_date=start_date, end_date=end_date, codes=codes)
        session = self._get_session()
        results = []
        for code in tqdm(codes, desc="Fetching 5min:"):
            if trade_date is not None:
                result = pl.DataFrame(self._request_json(
                    session=session,
                    code=code,
                    start_date=trade_date,
                    end_date=trade_date
                ))
            else:
                result = pl.DataFrame(self._request_json(
                    session=session,
                    code=code,
                    start_date=start_date,
                    end_date=end_date
                ))
            result.rename(FIELD_MAP_5MIN)
            result = result.with_columns(
                code=pl.lit(code)
            )
            results.append(result)
        df = pl.concat(results)
        validate_table(df, RAW_5MIN_SCHEMA)
        validate_time_format(df, RAW_5MIN_SCHEMA.get_column("time"))
        return partition_by(df, by="date")
