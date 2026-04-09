"""Entry point for running the raw data pipeline."""
from __future__ import annotations

from src.data.models import Query
from src.data.pipelines import RawPipeline

# Date range for data fetching
START_DATE = "20120101"
END_DATE = "20260408"


def main() -> None:
    """Run the raw data pipeline."""
    pipeline = RawPipeline()

    # Uncomment below to fetch and load data
    daily = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="daily", start_date=START_DATE, end_date=END_DATE),
    )

    adj_factor = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="adj_factor", start_date=START_DATE, end_date=END_DATE),
    )

    r5min = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="5min", start_date=START_DATE, end_date=END_DATE),
    )

    moneyflow = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="moneyflow", start_date=START_DATE, end_date=END_DATE),
    )

    limit = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="limit", start_date=START_DATE, end_date=END_DATE),
    )

    namechange = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="namechange", start_date=START_DATE, end_date=END_DATE),
    )
    suspend = pipeline.run(
        action={"fetch", "load"},
        query=Query(desc="suspend", start_date=START_DATE, end_date=END_DATE),
    )


if __name__ == "__main__":
    main()

