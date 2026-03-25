from __future__ import annotations

from src.data.pipelines import RawPipeline
from src.data.models import Query


START_DATE = "20260319"
END_DATE = "20260324"


if __name__ == '__main__':
    pipeline = RawPipeline()

    daily = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="daily", start_date=START_DATE, end_date=END_DATE)
    )

    adj_factor = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="adj_factor", start_date=START_DATE, end_date=END_DATE)
    )

    r5min = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="5min", start_date=START_DATE, end_date=END_DATE)
    )

    moneyflow = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="moneyflow", start_date=START_DATE, end_date=END_DATE)
    )

    limit = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="limit", start_date=START_DATE, end_date=END_DATE)
    )

    st = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="st", start_date=START_DATE, end_date=END_DATE)
    )

    suspend = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="suspend", start_date=START_DATE, end_date=END_DATE)
    )

    print(daily)
    print(adj_factor)
    print(r5min)
    print(moneyflow)
    print(limit)
    print(st)
    print(suspend)

